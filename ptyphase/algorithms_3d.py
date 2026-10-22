"""
Projections and kernels for the 3D-coupled model (see phase_abstract_3d).

Model set M (pupil P held fixed):
    Psi_j(K) = P[pup_idx0_j, pup_idx1_j](K) * S(q_z,j(K), K)
with S(q_z) linearly interpolated between nodes j0 and j0+1 of the q_z grid.

Each sample touches two q_z nodes in its own K column, so the normal matrix of the
least-squares fit  min_S sum |P S(q(K)) - Psi|^2  is TRIDIAGONAL along q_z for every K
column. `fit_object` solves it exactly (vectorised Thomas algorithm), with a small
Tikhonov term pulling unmeasured voxels towards the previous estimate, so P_M is an
exact projection (for a fixed pupil and no support constraint).
"""
import numpy as np
from .utils_pr import time_it
from .algorithms import AlgorithmKernel, project_data, ifft_images, fft_images


# ~~~___________________ Sampling operators ___________________~~~

def pupil_stack(geom, pupil):
    """Pupil patch of every image, (N, n0, n1)."""
    return pupil[geom.pup_idx0[:, :, None], geom.pup_idx1[:, None, :]]


def gather(geom, S):
    """S evaluated at every (image, K) sample, (N, n0, n1)."""
    Sf = S.ravel()
    return (1 - geom.w) * Sf[geom.node0] + geom.w * Sf[geom.node0 + geom.n_lat]


def _bincount_c(idx, vals, n):
    return (np.bincount(idx, weights=vals.real, minlength=n)
            + 1j * np.bincount(idx, weights=vals.imag, minlength=n))


def normal_equations(geom, PSI, pupil):
    """Right-hand side B^H Psi and tridiagonal B^H B (diag, off-diag) along q_z.

    Returns rhs (Nz, M) complex, diag (Nz, M), off (Nz-1, M) real; M = n0*n1.
    """
    Nz = geom.obj_shape[0]
    M = geom.n_lat
    n = Nz * M
    P = pupil_stack(geom, pupil)
    p2 = np.abs(P)**2
    w = geom.w
    i0 = geom.node0.ravel()
    i1 = i0 + M

    y = (np.conj(P) * PSI).ravel()
    wf = w.ravel()
    rhs = _bincount_c(i0, (1 - wf) * y, n) + _bincount_c(i1, wf * y, n)

    p2f = p2.ravel()
    diag = (np.bincount(i0, weights=p2f * (1 - wf)**2, minlength=n)
            + np.bincount(i1, weights=p2f * wf**2, minlength=n))
    off = np.bincount(i0, weights=p2f * wf * (1 - wf), minlength=n)

    return rhs.reshape(Nz, M), diag.reshape(Nz, M), off.reshape(Nz, M)[:-1]


def solve_tridiagonal(diag, off, rhs):
    """Batched symmetric tridiagonal solve along axis 0 (Thomas algorithm)."""
    Nz = diag.shape[0]
    c = np.zeros_like(off)
    d = np.zeros_like(rhs)
    denom = diag[0].copy()
    c[0] = off[0] / denom
    d[0] = rhs[0] / denom
    for i in range(1, Nz):
        denom = diag[i] - off[i - 1] * c[i - 1]
        if i < Nz - 1:
            c[i] = off[i] / denom
        d[i] = (rhs[i] - off[i - 1] * d[i - 1]) / denom
    x = np.empty_like(rhs)
    x[-1] = d[-1]
    for i in range(Nz - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def coverage(geom, pupil, rel=None):
    """Bool (Nz, n0, n1): voxels the data actually constrains."""
    rel = geom.eps if rel is None else rel
    Nz, M = geom.obj_shape[0], geom.n_lat
    p2 = np.abs(pupil_stack(geom, pupil)).ravel()**2
    i0 = geom.node0.ravel()
    wf = geom.w.ravel()
    den = (np.bincount(i0, weights=p2 * (1 - wf), minlength=Nz * M)
           + np.bincount(i0 + M, weights=p2 * wf, minlength=Nz * M))
    return (den > rel * den.max()).reshape(geom.obj_shape)


def apply_support(geom, S):
    if geom.support is None:
        return S
    obj = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(S))) * geom.support
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj)))


def fit_object(geom, PSI, pupil, S_prev):
    """Least-squares 3D spectrum for the exit waves PSI (pupil fixed):

        argmin_S  sum |P gather(S) - PSI|^2 + mu |S - S_prev|^2,  mu = eps * max(diag)

    With a support, alternates the data fit with the real-space support
    (Gerchberg-Papoulis); then P_M is an approximation of the joint projection.
    """
    rhs, diag, off = normal_equations(geom, PSI, pupil)
    mu = geom.eps * diag.max()
    shape = geom.obj_shape
    S_prev2 = S_prev.reshape(shape[0], -1)
    S = solve_tridiagonal(diag + mu, off, rhs + mu * S_prev2).reshape(shape)

    if geom.support is not None:
        for _ in range(geom.support_iters):
            S_sup = apply_support(geom, S).reshape(shape[0], -1)
            S = solve_tridiagonal(diag + mu, off, rhs + mu * S_sup).reshape(shape)
        S = apply_support(geom, S)
    return S


# ~~~___________________ Pupil helpers ___________________~~~

def pupil_coverage(geom, pupil_shape):
    """Bool (pupil_shape): pupil pixels that at least one (image, K) sample touches.
    Only these can be recovered from the data; the rest keeps its initial value."""
    flat = (geom.pup_idx0[:, :, None] * pupil_shape[1] + geom.pup_idx1[:, None, :])
    counts = np.bincount(flat.ravel(), minlength=pupil_shape[0] * pupil_shape[1])
    return (counts > 0).reshape(pupil_shape)


def normalise_pupil_scale(pupil, objectFT, cover):
    """P and S only enter as P·S, so their relative scale is free and drifts during
    alternating updates. Rescale so that rms|P| = 1 on the covered pupil (P·S unchanged)."""
    c = np.sqrt(np.mean(np.abs(pupil[cover])**2))
    if not np.isfinite(c) or c == 0:
        return pupil, objectFT
    return pupil / c, objectFT * c


# ~~~___________________ Projections ___________________~~~

def project_model(geom, pupil, objectFT):
    """Exit-wave spectra predicted by (pupil, 3D spectrum), (N, n0, n1)."""
    return pupil_stack(geom, pupil) * gather(geom, objectFT)


def project_model_fit(geom, PSI, pupil, objectFT):
    """P_M(PSI): refit S to PSI, then re-predict. Returns (Psi_M, S_fit)."""
    S_fit = fit_object(geom, PSI, pupil, objectFT)
    return project_model(geom, pupil, S_fit), S_fit


# ~~~___________________ Kernels ___________________~~~

class AlgorithmKernel3D(AlgorithmKernel):
    """Kernel interface for the 3D engine. Kernels get the geometry `geom`
    instead of (pupil_slices, object_slices)."""

    def step(self, PSI, geom, objectFT, pupil_func, images, **kwargs):
        raise NotImplementedError

    @time_it
    def update_object_fts(self, PSI, geom, pupil, objectFT):
        return fit_object(geom, PSI, pupil, objectFT)

    @time_it
    def update_pupil(self, PSI, geom, objectFT, pupil_func, ctf=None, eps=1e-3):
        """LSQ pupil for a fixed object; pixels with little object power keep
        their previous value. Constrained to the aperture `ctf` if given."""
        O = gather(geom, objectFT)
        flat = (geom.pup_idx0[:, :, None] * pupil_func.shape[1] + geom.pup_idx1[:, None, :])
        flat = np.broadcast_to(flat, O.shape).ravel()
        n = pupil_func.size
        numer = _bincount_c(flat, (np.conj(O) * PSI).ravel(), n)
        denom = np.bincount(flat, weights=(np.abs(O)**2).ravel(), minlength=n)
        mu = eps * denom.max()
        P = ((numer + mu * pupil_func.ravel()) / (denom + mu)).reshape(pupil_func.shape)
        if ctf is not None:
            P = P * (np.abs(ctf) > 0)
        return P

    def compute_gap(self, PSI, geom, pupil, objectFT, images):
        """|| P_D(PSI) - P_M(PSI) || / || PSI ||"""
        Psi_M, _ = project_model_fit(geom, PSI, pupil, objectFT)
        Psi_D = project_data(images, PSI)
        return np.linalg.norm(Psi_D - Psi_M) / np.linalg.norm(PSI)

    def compute_data_misfit(self, geom, pupil, objectFT, images):
        """sum || |ifft(P S)| - sqrt(I) ||^2 / sum I  for the current model."""
        psi = ifft_images(project_model(geom, pupil, objectFT))
        err = np.sum((np.abs(psi) - np.sqrt(images))**2, axis=(-2, -1))
        return np.sum(err) / np.sum(images), err / np.sum(images, axis=(-2, -1))


class DM3D(AlgorithmKernel3D):
    """Difference Map (Elser 2003) with A = data, B = model:

        f_D = P_D(Ψ) - (P_D(Ψ) - Ψ)/β
        f_M = P_M(Ψ) + (P_M(Ψ) - Ψ)/β
        Ψ  <- Ψ + β [ P_D(f_M) - P_M(f_D) ]

    β = 1 gives the usual ptychographic DM (Thibault 2008):
        Ψ <- Ψ + P_D(2 P_M Ψ - Ψ) - P_M Ψ.
    The solution is read from P_M(f_D): the engine fits the object to f_D.
    """

    def __init__(self, beta=1.0, beta_decay=None, beta_min=0.5):
        self.beta = beta
        self.beta_decay = beta_decay
        self.beta_min = beta_min

    def step(self, PSI, geom, objectFT, pupil_func, images, **kwargs):
        b = self.beta
        Psi_D = project_data(images, PSI)
        Psi_M, _ = project_model_fit(geom, PSI, pupil_func, objectFT)

        f_D = Psi_D - (Psi_D - PSI) / b
        f_M = Psi_M + (Psi_M - PSI) / b

        if b == 1.0:
            P_M_fD = Psi_M                          # f_D == Ψ
        else:
            P_M_fD, _ = project_model_fit(geom, f_D, pupil_func, objectFT)

        Psi_n = PSI + b * (project_data(images, f_M) - P_M_fD)

        if self.beta_decay is not None:
            self.beta = max(self.beta * self.beta_decay, self.beta_min)
        return Psi_n

    def solution_estimate(self, PSI, images):
        Psi_D = project_data(images, PSI)
        return Psi_D - (Psi_D - PSI) / self.beta


class ER3D(AlgorithmKernel3D):
    """Error reduction: Ψ <- P_D(P_M(Ψ)). Use to polish after DM."""

    def step(self, PSI, geom, objectFT, pupil_func, images, **kwargs):
        Psi_M, _ = project_model_fit(geom, PSI, pupil_func, objectFT)
        return project_data(images, Psi_M)


class RAAR3D(AlgorithmKernel3D):
    """RAAR (Luke 2004), data set inner:
        Ψ <- β Ψ + β P_M(R_D Ψ) + (1-2β) P_D(Ψ); solution from P_D(Ψ)."""

    def __init__(self, beta=0.9, beta_decay=None, beta_min=0.5):
        self.beta = beta
        self.beta_decay = beta_decay
        self.beta_min = beta_min

    def step(self, PSI, geom, objectFT, pupil_func, images, **kwargs):
        Psi_D = project_data(images, PSI)
        P_M_RD, _ = project_model_fit(geom, 2 * Psi_D - PSI, pupil_func, objectFT)
        Psi_n = self.beta * PSI + self.beta * P_M_RD + (1 - 2 * self.beta) * Psi_D
        if self.beta_decay is not None:
            self.beta = max(self.beta * self.beta_decay, self.beta_min)
        return Psi_n

    def solution_estimate(self, PSI, images):
        return project_data(images, PSI)
