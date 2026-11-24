"""
Learnable quantities: the 3D object spectrum and the pupil.

Complex quantities are stored as (real, imag) float parameters, not (amplitude, phase):
the phase gradient is singular where |S| = 0, and most of the 3D spectrum is near zero.

The forward model only sees the product P * S, so (P, S) -> (P/c, S*c) leaves it unchanged.
`scale_(c)` applies that gauge in place (no grad); the engine uses it to keep rms|P| = 1
on the covered pupil.
"""
import numpy as np
import torch
import torch.nn as nn

from ..zernike import SquarePolynomials

N_POLY = 29


def _complex_param(z, real_dtype, device):
    z = np.asarray(z)
    return (nn.Parameter(torch.as_tensor(z.real.copy(), dtype=real_dtype, device=device)),
            nn.Parameter(torch.as_tensor(z.imag.copy(), dtype=real_dtype, device=device)))


class Spectrum3D(nn.Module):
    """S(q_z, K0, K1) = scale * (re + i im), shape (Nz, n0, n1).

    scale : fixed natural magnitude of S, so the parameters are O(1) and a learning rate
            (e.g. Adam's, which sets the step size per parameter) is relative to it.
    mask  : optional bool (Nz, n0, n1); voxels outside it are held at 0.
    """

    def __init__(self, S0, scale=1.0, mask=None, real_dtype=torch.float64, device="cpu"):
        super().__init__()
        self.scale = float(scale)
        self.re, self.im = _complex_param(np.asarray(S0) / self.scale, real_dtype, device)
        if mask is not None:
            self.register_buffer("mask", torch.as_tensor(np.asarray(mask), dtype=real_dtype, device=device))
        else:
            self.mask = None

    def forward(self):
        S = self.scale * torch.complex(self.re, self.im)
        return S if self.mask is None else S * self.mask

    @torch.no_grad()
    def scale_(self, c):
        self.re.mul_(c)
        self.im.mul_(c)


class PixelPupil(nn.Module):
    """Free complex pupil on the pupil grid, multiplied by the aperture `ctf`.

    freeze_amp : only the phase is learnable; the amplitude stays |P0|.
    """

    def __init__(self, P0, ctf, freeze_amp=False, real_dtype=torch.float64, device="cpu"):
        super().__init__()
        P0 = np.asarray(P0)
        self.freeze_amp = freeze_amp
        self.register_buffer("ctf", torch.as_tensor(np.abs(np.asarray(ctf)) > 0, dtype=real_dtype, device=device))
        if freeze_amp:
            self.register_buffer("amp", torch.as_tensor(np.abs(P0), dtype=real_dtype, device=device))
            self.phase = nn.Parameter(torch.as_tensor(np.angle(P0), dtype=real_dtype, device=device))
        else:
            self.re, self.im = _complex_param(P0, real_dtype, device)

    def forward(self):
        if self.freeze_amp:
            P = self.amp * torch.exp(1j * self.phase)
        else:
            P = torch.complex(self.re, self.im)
        return P #* self.ctf

    @torch.no_grad()
    def scale_(self, c):
        """P -> P / c."""
        if self.freeze_amp:
            self.amp.div_(c)
        else:
            self.re.div_(c)
            self.im.div_(c)


def square_poly_basis(k0_axis, k1_axis, u_max, centre=(0.0, 0.0)):
    """S1..S29 (Mahajan & Dai square polynomials) on the pupil grid, (29, P0, P1).

    Same normalisation as reciprocity_sim.beam.SquareAberrations: x = (k0 - c0)/u_max / sqrt(2),
    y = (k1 - c1)/u_max / sqrt(2), so the aperture edge |k| = u_max maps to the edge of the
    basis's square [-1/sqrt2, 1/sqrt2]. Axis 0 of the pupil is k[0] (ptyphase [x, y] layout),
    so the sim's true coefficients can be used directly.
    """
    a = 1 / np.sqrt(2)
    X, Y = np.meshgrid((np.asarray(k0_axis) - centre[0]) / u_max * a,
                       (np.asarray(k1_axis) - centre[1]) / u_max * a, indexing="ij")
    return np.stack([getattr(SquarePolynomials, f"S{i}")([X, Y], 1.0) * np.ones_like(X)
                     for i in range(1, N_POLY + 1)])


def fit_poly_coeffs(phase, basis, mask):
    """Least-squares S1..S29 coefficients of an (unwrapped) phase on the mask."""
    A = basis[:, mask].T
    coeffs, *_ = np.linalg.lstsq(A, phase[mask], rcond=None)
    return coeffs


class PolyPupil(nn.Module):
    """P = amp * exp(i * sum_i c_i S_i) * ctf.

    basis    : (29, P0, P1) from `square_poly_basis`.
    coeffs0  : initial coefficients (29,).
    amp0     : initial amplitude (P0, P1).
    free_amp : learn the amplitude per pixel too (otherwise it stays amp0).
    """

    def __init__(self, basis, coeffs0, amp0, ctf, free_amp=False, real_dtype=torch.float64, device="cpu"):
        super().__init__()
        self.register_buffer("basis", torch.as_tensor(basis, dtype=real_dtype, device=device))
        self.register_buffer("ctf", torch.as_tensor(np.abs(np.asarray(ctf)) > 0, dtype=real_dtype, device=device))
        self.coeffs = nn.Parameter(torch.as_tensor(np.asarray(coeffs0, dtype=float), dtype=real_dtype, device=device))
        amp0 = torch.as_tensor(np.asarray(amp0, dtype=float), dtype=real_dtype, device=device)
        self.free_amp = free_amp
        if free_amp:
            self.amp = nn.Parameter(amp0)
        else:
            self.register_buffer("amp", amp0)

    def phase(self):
        return torch.einsum("i,ijk->jk", self.coeffs, self.basis)

    def forward(self):
        return self.amp * torch.exp(1j * self.phase()) #* self.ctf

    @torch.no_grad()
    def scale_(self, c):
        self.amp.div_(c)
