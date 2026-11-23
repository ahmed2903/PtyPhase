"""
Differentiable forward model of the 3D-coupled Bragg Fourier-ptychography problem.

For image j (k_in0,j = k_out,j - G) and scan frequency K on the image K-grid:

    k_in(K)  = k_in0,j + K                                  (ptyphase convention, [x, y])
    q_z(K)   = k_out,z - sqrt(k^2 - |k_in(K)|^2) - G_z
    Psi~_j(K) = P(k_in(K)) * S(q_z(K), K)                    S linear in q_z between two nodes
    I_j      = |IFFT2c(Psi~_j)|^2

This is `projection.algorithms_3d.project_model` (pupil_stack * gather) in torch, using the
same integer geometry (`EwaldGeometry`: pup_idx0/1, node0, w), so it is validated against
reciprocity_sim.BraggSimulator in the same way. Everything here is differentiable w.r.t.
S and P.
"""
import torch


def fft2c(x):
    """Centred 2D FFT over the last two axes (same as np fftshift(fft2(ifftshift)))."""
    d = (-2, -1)
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=d), dim=d), dim=d)


def ifft2c(x):
    d = (-2, -1)
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=d), dim=d), dim=d)


def ifft3c(x):
    """Centred 3D iFFT: spectrum [qz, K0, K1] -> object [z, x, y]."""
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(x)))


def fft3c(x):
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(x)))


def abs2(z):
    """|z|^2 without the gradient singularity of abs() at z = 0."""
    return z.real**2 + z.imag**2


class EwaldSampler:
    """Fixed sampling geometry as torch tensors.

    geom : projection.phase_abstract_3d.EwaldGeometry
    """

    def __init__(self, geom, device, real_dtype=torch.float64):
        self.obj_shape = tuple(geom.obj_shape)
        self.n_lat = int(geom.n_lat)
        self.pup_idx0 = torch.as_tensor(geom.pup_idx0, dtype=torch.long, device=device)   # (N, n0)
        self.pup_idx1 = torch.as_tensor(geom.pup_idx1, dtype=torch.long, device=device)   # (N, n1)
        self.node0 = torch.as_tensor(geom.node0, dtype=torch.long, device=device)         # (N, n0, n1)
        self.w = torch.as_tensor(geom.w, dtype=real_dtype, device=device)                 # (N, n0, n1)
        self.num_images = self.node0.shape[0]

    def _sel(self, t, idx):
        return t if idx is None else t[idx]

    def gather(self, S, idx=None):
        """S at every (image, K) sample of the images `idx` (None = all): (B, n0, n1)."""
        Sf = S.reshape(-1)
        node0 = self._sel(self.node0, idx)
        w = self._sel(self.w, idx)
        return (1 - w) * Sf[node0] + w * Sf[node0 + self.n_lat]

    def pupil_patches(self, P, idx=None):
        """Pupil patch of every image: P(k_in0 + K), (B, n0, n1)."""
        i0 = self._sel(self.pup_idx0, idx)
        i1 = self._sel(self.pup_idx1, idx)
        return P[i0[:, :, None], i1[:, None, :]]

    def fields(self, S, P, idx=None):
        """Exit-wave spectra Psi~ = P * S (the coherent-image spectra), (B, n0, n1)."""
        return self.pupil_patches(P, idx) * self.gather(S, idx)

    def intensities(self, S, P, idx=None):
        """Coherent-image intensities |IFFT2c(P * S)|^2, (B, n0, n1)."""
        return abs2(ifft2c(self.fields(S, P, idx)))
