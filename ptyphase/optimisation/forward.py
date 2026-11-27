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

`ShiftableEwaldSampler` places the pupil patches continuously (bilinear, no rounding of the
patch start) and lets every image move by a lateral k_in shift (e.g. a per-streak offset), with
q_z recomputed; it is differentiable w.r.t. the shift as well.
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


class ShiftableEwaldSampler(EwaldSampler):
    """EwaldSampler with continuous (sub-pixel) and differentiable placement of every image.

    Image j can be moved by a lateral shift dkin_j (rad/nm, [x, y]) of its pupil position k_in0,j:

      pupil : P(k_in0 + dkin + K) by bilinear interpolation on the pupil grid. The patch start is
              not rounded, so even dkin = 0 places the patch exactly (EwaldSampler rounds it to a
              whole pupil pixel; the two agree when k_in0 sits on a pupil pixel).
      q_z   : recomputed for every sample.
              mode 'kout' : k_out moves with k_in (a detector-mapping error, one global G), so
                            k_out,z changes on the Ewald sphere and q_z follows (~tan2θ·|dkin|);
              mode 'kin'  : only k_in moves (a per-image G); q_z changes through k_in,z only.
              With dkin = None the precomputed geometry (geom.node0, geom.w) is used, so q_z is
              exactly geom.qz.

    The pupil-grid convention is that of PhaseRetrievalBase: pupil index p <-> k = k_min + p*dk,
    patch pixel i <-> k_in0 + (i - n//2)*dk, i.e. the same K-grid as _qz_samples. (For odd image
    sizes the rounded integer patch start of EwaldSampler differs from this by half a pixel.)

    geom      : EwaldGeometry
    kins      : (N, 3) k_in (rad/nm); kouts : (N, 3) k_out; G : (3,); k : 2π/λ
    k_min, dk : (k0_min, k1_min), (dk0, dk1) of the pupil grid; img_shape : (n0, n1)
    """

    def __init__(self, geom, kins, kouts, G, k, k_min, dk, img_shape, device,
                 real_dtype=torch.float64, mode="kout", placement="bilinear"):
        super().__init__(geom, device, real_dtype)
        if mode not in ("kout", "kin"):
            raise ValueError("mode must be 'kout' or 'kin'")
        if placement not in ("bilinear", "nearest"):
            raise ValueError("placement must be 'bilinear' or 'nearest'")
        f64 = dict(dtype=torch.float64, device=device)
        self.mode = mode
        self.placement = placement      # 'nearest': patch start rounded (EwaldSampler at dkin = 0; no pupil gradient)
        self.rdtype = real_dtype
        self.kin = torch.as_tensor(kins[:, :2], **f64)              # (N, 2)
        self.kout_lat = torch.as_tensor(kouts[:, :2], **f64)        # (N, 2)
        self.kout_z = torch.as_tensor(kouts[:, 2], **f64)           # (N,)
        self.Gz = float(G[2])
        self.k = float(k)
        self.k_min = (float(k_min[0]), float(k_min[1]))
        self.dk = (float(dk[0]), float(dk[1]))
        n0, n1 = img_shape
        self.c = (n0 // 2, n1 // 2)
        self.K0 = (torch.arange(n0, **f64) - n0 // 2) * self.dk[0]
        self.K1 = (torch.arange(n1, **f64) - n1 // 2) * self.dk[1]
        self.ar0 = torch.arange(n0 + 1, dtype=torch.long, device=device)
        self.ar1 = torch.arange(n1 + 1, dtype=torch.long, device=device)
        self.Nz = int(self.obj_shape[0])
        self.dqz = float(geom.dqz)
        self.lat = torch.arange(n0 * n1, dtype=torch.long, device=device).reshape(n0, n1)
        self.n_clamped_pupil = 0          # samples whose patch fell off the pupil grid (clamped)
        self.n_clamped_qz = 0             # samples whose q_z fell off the q_z grid (clamped)

    def _kin(self, idx, dkin):
        kin = self._sel(self.kin, idx)
        return kin if dkin is None else kin + dkin.to(torch.float64)

    def pupil_patches(self, P, idx=None, dkin=None):
        """P(k_in0 + dkin + K), bilinear on the pupil grid, (B, n0, n1). Differentiable w.r.t.
        P and dkin. One gather of an (n0+1, n1+1) patch, blended from its four corners."""
        kin = self._kin(idx, dkin)
        f0 = (kin[:, 0] - self.k_min[0]) / self.dk[0] - self.c[0]
        f1 = (kin[:, 1] - self.k_min[1]) / self.dk[1] - self.c[1]
        if self.placement == "nearest":
            # the integer rule of PhaseRetrievalBase._compute_patch_bounds (round half to even)
            I0 = torch.round(f0).long()[:, None] + self.ar0[None, :-1]
            I1 = torch.round(f1).long()[:, None] + self.ar1[None, :-1]
            P0, P1 = P.shape[-2:]
            bad = (I0[:, 0] < 0) | (I0[:, -1] > P0 - 1) | (I1[:, 0] < 0) | (I1[:, -1] > P1 - 1)
            if bool(bad.any()):
                self.n_clamped_pupil += int(bad.sum())
                I0, I1 = I0.clamp(0, P0 - 1), I1.clamp(0, P1 - 1)
            return P[I0[:, :, None], I1[:, None, :]]
        i0, i1 = torch.floor(f0).detach(), torch.floor(f1).detach()
        t0 = (f0 - i0).to(self.rdtype)[:, None, None]
        t1 = (f1 - i1).to(self.rdtype)[:, None, None]
        I0 = i0.long()[:, None] + self.ar0[None, :]                  # (B, n0+1)
        I1 = i1.long()[:, None] + self.ar1[None, :]                  # (B, n1+1)
        P0, P1 = P.shape[-2:]
        bad = (I0[:, 0] < 0) | (I0[:, -1] > P0 - 1) | (I1[:, 0] < 0) | (I1[:, -1] > P1 - 1)
        if bool(bad.any()):
            self.n_clamped_pupil += int(bad.sum())
            I0, I1 = I0.clamp(0, P0 - 1), I1.clamp(0, P1 - 1)
        Q = P[I0[:, :, None], I1[:, None, :]]                       # (B, n0+1, n1+1)
        top = (1 - t1) * Q[:, :-1, :-1] + t1 * Q[:, :-1, 1:]
        bot = (1 - t1) * Q[:, 1:, :-1] + t1 * Q[:, 1:, 1:]
        return (1 - t0) * top + t0 * bot

    def qz(self, idx=None, dkin=None):
        """Continuous q_z of every sample of the images idx with k_in0 shifted by dkin, (B, n0, n1),
        float64. mode 'kout': k_out shifted by dkin too (on the Ewald sphere)."""
        kin = self._kin(idx, dkin)
        k0 = kin[:, 0, None, None] + self.K0[None, :, None]
        k1 = kin[:, 1, None, None] + self.K1[None, None, :]
        kin_z = torch.sqrt(torch.clamp(self.k**2 - k0**2 - k1**2, min=0.0))
        kout_z = self._sel(self.kout_z, idx)
        if dkin is not None and self.mode == "kout":
            d = dkin.to(torch.float64)
            ko = self._sel(self.kout_lat, idx)
            r0 = self.k**2 - torch.sum(ko**2, dim=1)
            r1 = self.k**2 - torch.sum((ko + d)**2, dim=1)
            # sqrt(r1) - sqrt(r0) without cancellation
            kout_z = kout_z - (2 * torch.sum(ko * d, dim=1) + torch.sum(d**2, dim=1)) / (
                torch.sqrt(torch.clamp(r1, min=0.0)) + torch.sqrt(torch.clamp(r0, min=0.0)))
        return kout_z[:, None, None] - kin_z - self.Gz

    def gather(self, S, idx=None, dkin=None):
        """S at every sample (linear in q_z between two nodes), (B, n0, n1). dkin None: the
        precomputed geometry (exactly geom.qz)."""
        if dkin is None:
            return super().gather(S, idx)
        f = self.qz(idx, dkin) / self.dqz + self.Nz // 2
        j0 = torch.floor(f).detach()
        w = (f - j0).to(self.rdtype)
        j0 = j0.long()
        bad = (j0 < 0) | (j0 > self.Nz - 2)
        if bool(bad.any()):
            self.n_clamped_qz += int(bad.sum())
            j0 = j0.clamp(0, self.Nz - 2)
            w = torch.where(bad, w.clamp(0.0, 1.0), w)
        node0 = j0 * self.n_lat + self.lat[None]
        Sf = S.reshape(-1)
        return (1 - w) * Sf[node0] + w * Sf[node0 + self.n_lat]

    def fields(self, S, P, idx=None, dkin=None):
        """Exit-wave spectra P(k_in0 + dkin + K) * S(q_z(K; dkin), K), (B, n0, n1)."""
        return self.pupil_patches(P, idx, dkin) * self.gather(S, idx, dkin)

    def intensities(self, S, P, idx=None, dkin=None):
        return abs2(ifft2c(self.fields(S, P, idx, dkin)))

    def coverage(self, pupil_shape, dkin=None):
        """Bool (pupil_shape): pupil pixels the bilinear patches touch (with weight possibly 0
        on the far edge) at shifts dkin (N, 2) or none."""
        with torch.no_grad():
            kin = self._kin(None, dkin)
            i0 = torch.floor((kin[:, 0] - self.k_min[0]) / self.dk[0] - self.c[0]).long()
            i1 = torch.floor((kin[:, 1] - self.k_min[1]) / self.dk[1] - self.c[1]).long()
            n0, n1 = len(self.ar0) - 1, len(self.ar1) - 1
            # 2D difference array: +1 at the patch start, -1 past its end, then cumulative sums
            a0, b0 = i0.clamp(0, pupil_shape[0]), (i0 + n0 + 1).clamp(0, pupil_shape[0])
            a1, b1 = i1.clamp(0, pupil_shape[1]), (i1 + n1 + 1).clamp(0, pupil_shape[1])
            d = torch.zeros((pupil_shape[0] + 1, pupil_shape[1] + 1), dtype=torch.int64, device=kin.device)
            ones = torch.ones_like(a0, dtype=torch.int64)
            for r, c, s in ((a0, a1, 1), (a0, b1, -1), (b0, a1, -1), (b0, b1, 1)):
                d.index_put_((r, c), s * ones, accumulate=True)
            m = torch.cumsum(torch.cumsum(d, 0), 1)[:pupil_shape[0], :pupil_shape[1]]
            return (m > 0).cpu().numpy()
