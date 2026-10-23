"""
Curved-slice <-> 3D object mapping, 3D support and kernels for the per-streak engine
(see phase_abstract_ewald).

Kernels are the 2D ones from algorithms.py with `EwaldSupportMixin` in front: every
object fit (update_object_fts, hence every project_model_fit inside DM / RAAR_standard
and compute_gap) becomes

    per-streak LSQ  ->  slices -> 3D object  ->  * support  ->  3D object -> slices

Support after LSQ approximates the joint projection onto {P*O : O from a supported
3D object}; it is exact where the pupil weight is uniform. Slice pixels the data does
not constrain (including zero-padded streaks) are filled in by the support
(Gerchberg-Papoulis).
"""
import numpy as np
from scipy.ndimage import gaussian_filter

from .algorithms import (AlgorithmKernel, DM, RAAR, RAAR_standard, ER, fft_images, ifft_images,
                         project_data, project_model)

ifft_z = lambda x: np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=0), axis=0), axes=0)
fft_z = lambda x: np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=0), axis=0), axes=0)


# ~~~___________________ Slice <-> 3D mapping ___________________~~~

def slices_to_object(geom, F):
    """Stack of per-streak spectra F[s, K0, K1] -> 3D object [z, x, y].
    Slice s sits at q_s + c(K): IFFT along s gives exp(i q_s z), the ramp adds exp(i c(K) z)."""
    f = ifft_z(F)
    if geom.ramp is not None:
        f = f * geom.ramp
    return ifft_images(f)


def object_to_slices(geom, obj):
    """Exact inverse of slices_to_object."""
    f = fft_images(obj)
    if geom.ramp is not None:
        f = f * np.conj(geom.ramp)
    return fft_z(f)


def project_support_3d(geom, F):
    """Apply the 3D real-space support to the slice stack F (unchanged if no support)."""
    if geom.support is None:
        return F
    return object_to_slices(geom, slices_to_object(geom, F) * geom.support)


def shrinkwrap_support(obj, sigma=1.0, threshold=0.1):
    """Bool support from the blurred |obj| thresholded at `threshold` * max."""
    blurred = gaussian_filter(np.abs(obj), sigma)
    return blurred > threshold * blurred.max()


# ~~~___________________ Kernels ___________________~~~

class EwaldSupportMixin:
    """Put in front of a 2D kernel: the object fit is followed by the 3D support.
    The engine sets `geom` (EwaldSliceGeometry) before stepping."""
    geom = None

    def update_object_fts(self, PSI, pupil, pupil_slices, object_slices, objectFTs=None, eps=1e-6):
        if self.geom is None:
            return super().update_object_fts(PSI, pupil, pupil_slices, object_slices,
                                             objectFTs=objectFTs, eps=eps)
        if objectFTs is None:
            # same as no previous estimate (O_prev = 0), but sized to all streaks,
            # including trailing ones without images
            objectFTs = np.zeros(self.geom.obj_shape, dtype=complex)
        O = super().update_object_fts(PSI, pupil, pupil_slices, object_slices,
                                      objectFTs=objectFTs, eps=eps)
        return project_support_3d(self.geom, O)


class DMEwald(EwaldSupportMixin, DM):
    pass


class RAAREwald(EwaldSupportMixin, RAAR):
    pass


class RAAR_standardEwald(EwaldSupportMixin, RAAR_standard):
    pass


class EREwald(EwaldSupportMixin, ER):
    pass


# ~~~___________________ Divide and concur ___________________~~~

def project_overlap(PSI, F, pupil, pupil_slices, object_slices, lam):
    """Exact projection of (PSI, F) onto {PSI = P*F} for a fixed pupil, in the metric
    |dPSI|^2 + lam |dF|^2. Per streak and pixel:

        F' = (sum_j conj(P_j) psi_j + lam F) / (sum_j |P_j|^2 + lam),   psi'_j = P_j F'

    Returns (PSI', F')."""
    denom = np.zeros(F.shape, dtype=np.float64)
    numer = lam * F.astype(np.complex128)

    for pup_sl, obj_sl, psi_i in zip(pupil_slices, object_slices, PSI):
        pupil_patch = pupil[pup_sl]
        denom[obj_sl] += np.abs(pupil_patch)**2
        numer[obj_sl] += np.conjugate(pupil_patch) * psi_i

    F_new = numer / (denom + lam)
    return project_model(pupil_slices, object_slices, pupil, F_new), F_new


class DnCEwald(AlgorithmKernel):
    """Divide and concur (Gravel & Elser 2008) with three constraints on the pair
    (PSI, F) = (exit-wave spectra, curved-slice stack). Each constraint holds replicas
    of only the variables it involves:

        data     D : PSI_D          -> modulus replacement
        overlap  O : (PSI_O, F_O)   -> PSI = P*F (project_overlap, metric weight lam on F)
        support  S : F_S            -> 3D real-space support on the curved slices

    x = [PSI_D, PSI_O, F_O, F_S]. Divide P_A projects every replica onto its own
    constraint; concur P_B replaces the PSI replicas by their mean and the F replicas by
    theirs. Iteration: Elser's DM with A = divide, B = concur (as DM3D):

        f_A = P_A(x) - (P_A(x) - x)/beta,   f_B = P_B(x) + (P_B(x) - x)/beta
        x  <- x + beta [ P_A(f_B) - P_B(f_A) ]

    beta = 1 is the Gravel-Elser DnC. The estimate (PSI, F) is read from P_B(f_A);
    `residual` = ||P_A(x) - P_B(x)|| / ||x|| is reported as the gap.
    Use with FourierPtychoEwaldEngine, which keeps the replicas across iterations and
    updates the pupil from the estimate.
    """
    geom = None

    def __init__(self, beta=1.0, lam=1.0, beta_decay=None, beta_min=0.5):
        if lam <= 0:
            raise ValueError("lam must be > 0")
        self.beta = beta
        self.lam = lam
        self.beta_decay = beta_decay
        self.beta_min = beta_min
        self.x = None
        self._iter = None
        self.residual = np.nan

    def reset(self, PSI, F):
        self.x = [PSI.copy(), PSI.copy(), F.copy(), F.copy()]

    def rescale(self, c):
        """F -> F*c in every replica (after the pupil was rescaled P -> P/c)."""
        self.x[2] = self.x[2] * c
        self.x[3] = self.x[3] * c

    def _divide(self, x, pupil, pupil_slices, object_slices, images):
        psi_O, F_O = project_overlap(x[1], x[2], pupil, pupil_slices, object_slices, self.lam)
        return [project_data(images, x[0]), psi_O, F_O, project_support_3d(self.geom, x[3])]

    @staticmethod
    def _concur(x):
        psi = 0.5 * (x[0] + x[1])
        F = 0.5 * (x[2] + x[3])
        return [psi, psi, F, F]

    def step(self, pupil_func, pupil_slices, object_slices, images, **kwargs):
        b = self.beta
        x = self.x
        divide = lambda v: self._divide(v, pupil_func, pupil_slices, object_slices, images)

        P_A = divide(x)
        P_B = self._concur(x)
        self.residual = np.sqrt(sum(np.linalg.norm(a - c)**2 for a, c in zip(P_A, P_B))
                                / sum(np.linalg.norm(xi)**2 for xi in x))

        if b == 1.0:
            P_B_fA = P_B                                          # f_A == x
        else:
            P_B_fA = self._concur([a - (a - xi) / b for a, xi in zip(P_A, x)])
        del P_A

        P_A_fB = divide([c + (c - xi) / b for c, xi in zip(P_B, x)])
        self.x = [xi + b * (a - c) for xi, a, c in zip(x, P_A_fB, P_B_fA)]

        if self.beta_decay is not None:
            self.beta = max(self.beta * self.beta_decay, self.beta_min)

        return P_B_fA[0], P_B_fA[2]
