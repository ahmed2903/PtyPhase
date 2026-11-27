"""
Gradient-based 3D reconstruction: fit one 3D spectrum S(q_z, K0, K1) (and the pupil) to the
measured coherent-image intensities by minimising

    L = loss(|IFFT2c(P(k_in0 + K) * S(q_z(K), K))|^2, I_meas) + sum_r w_r * R_r

with torch autograd and a torch optimiser (Adam by default). The forward model and its
geometry are the 3D-coupled one of `projection.phase_abstract_3d` (every (image, K) sample at
its own Ewald q_z), so the data interface, `prepare()` and the saved h5 layout are identical to
`FourierPtycho3DEngine`.

    eng = OptimFourierPtycho3D(images=..., kins=..., kouts=..., pupil_func=..., ks_pupil=...,
                               lr_psize=..., wavelength=..., G=...)
    eng.prepare(extend='triple', dqz=dqz, init='amplitude', device='cuda',
                pupil_model='pixel')                 # or 'poly' (S1..S29 phase)
    eng.set_optimiser(lr_spectrum=1e-2, lr_pupil=1e-3)
    eng.run(epochs=200, pupil_start=50)
    eng.save_reconsturction('out.h5')
    eng.obj_3D                                        # [z, x, y]

Per-streak position refinement (optional): prepare(streak_shifts='streaks') adds one offset per
streak along the in-pupil normal of the streaks u (= direction of grad q_z). In mode 'kout'
(default) the offset is a k_out (detector-mapping) error: k_in and k_out move together and q_z
follows (one global G). In mode 'kin' only k_in moves (a per-streak G). The offsets are learnt
with their own optimiser (set_optimiser(lr_shift=...), run(shift_start=...)) and/or set by a
grid scan of the per-streak loss (scan_streak_shifts, run(shift_scan_epochs=...)).

Units rad/nm; layout images/pupil [x, y], spectrum [qz, K0, K1], object [z, x, y].
"""
import time

import h5py
import numpy as np
import torch

from ..projection.phase_abstract_3d import PhaseRetrieval3DBase, Plot3D
from ..projection.algorithms_3d import pupil_coverage
from .forward import EwaldSampler, ShiftableEwaldSampler, abs2, ifft2c, ifft3c
from .losses import get_loss
from .models import Spectrum3D, PixelPupil, PolyPupil, StreakShifts, square_poly_basis, fit_poly_coeffs
from .regularisers import RegContext

_DTYPES = {"float64": torch.float64, "float32": torch.float32}


class OptimFourierPtycho3D(PhaseRetrieval3DBase, Plot3D):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reg_losses = {}        # name -> per-epoch penalty value (unweighted)
        self.reg_weights = {}       # name -> per-epoch weight
        self.reg_info = {}          # name -> description (saved as attrs)
        self.total_losses = []      # data loss + sum_r weight * penalty, per epoch
        self.epoch_times = []
        self.spectrum_optimiser = self.pupil_optimiser = self.shift_optimiser = None
        self.spectrum_scheduler = self.pupil_scheduler = self.shift_scheduler = None
        # per-streak shifts
        self.shifts = None
        self.shift_mode = None
        self.continuous = False     # sub-pixel (ShiftableEwaldSampler) forward model
        self.shift_history, self.shift_history_epochs = [], []
        self.shift_scans = []       # one dict per scan (epoch, offsets, L, best, delta before/after)
        self._shift_cfg = None
        self._init_spectrum = None
        self._qz_oversample = 1

    # ------------------------------------------------------------- prepare
    def prepare(self, device="cuda", dtype="float64", pupil_model="pixel",
                freeze_pupil_amp=False, pupil_coeffs=None, u_max=None,
                spectrum_mask=None, pixel_threshold=None, threshold_mode="mask",
                streak_shifts=None, shift_mode="kout", shift_max=None, subpixel=False,
                placement="bilinear", init_spectrum=None, init_pupil=None, **kwargs):
        """
        kwargs go to PhaseRetrieval3DBase.prepare (extend, dqz, qz_oversample, nz_pad, eps,
        support, z_extent_nm, support_iters, init). Then:

        device, dtype     : torch device ('cuda'/'cpu') and real precision ('float64'/'float32').
        pupil_model       : 'pixel' - free complex pupil inside the aperture;
                            'poly'  - phase = sum c_i S_i (square polynomials S1..S29).
        freeze_pupil_amp  : keep the pupil amplitude fixed at its initial value and learn only
                            the phase (pixel phases, or the poly coefficients).
        pupil_coeffs      : poly: initial S1..S29 coefficients. None -> LSQ fit of the
                            unwrapped phase of the initial pupil on the aperture.
        u_max             : poly: aperture half-width (rad/nm). None -> from ks_pupil.
        spectrum_mask     : optional bool (Nz, n0, n1); S is held at 0 outside it.
                            'measured' -> measured_mask(): only voxels some data sample
                            interpolates from are free (unmeasured q_z padding stays 0).
        pixel_threshold   : intensity threshold, in the units of the input images (same value for
                            every image). Pixels with I_meas < threshold are handled by
        threshold_mode    : 'mask' - excluded from the data loss and the misfit (weight 0): the
                                     model is free there;
                            'zero' - set to 0 in the data the model is fitted to (kept in the loss:
                                     the model is pushed to be dark there).
                            The initial spectrum (init='amplitude') still uses the raw images.
        streak_shifts     : None (off), 'streaks' (one offset per input streak; the data must be
                            given per streak), 'qz' (q_z bins of width dqz, for a flat stack) or an
                            int array (N,) with the group of every image. Enables the sub-pixel
                            forward model.
        shift_mode        : 'kout' - k_out and k_in move together, q_z follows (default);
                            'kin'  - only k_in moves (a per-streak G).
        shift_max         : bound on |offset| in pupil pixels. None -> 1.5 streak spacings.
                            The q_z grid and the 'measured' mask are widened to cover it.
        subpixel          : sub-pixel (bilinear) pupil placement without any shifts.
        placement         : pupil placement of the shiftable model: 'bilinear' (exact, sub-pixel,
                            differentiable w.r.t. the shifts) or 'nearest' (patch start rounded to a
                            pupil pixel, exactly as without shifts; only for scanning results that
                            were trained with the rounded placement, in whole-pixel steps; the shifts
                            then get no pupil gradient, so lr_shift is refused).
                            A pixel pupil trained with rounded placement can carry pixel-scale texture
                            tied to that rounding: evaluated with 'bilinear' it may fit much worse.
        init_spectrum     : (S, qz_axis) from a previous result: S (Nz', n0, n1) in input-image
                            units on the q_z axis qz_axis (rad/nm); placed on this grid (same dqz,
                            aligned nodes; zero-padded or cropped in q_z).
        init_pupil        : complex pupil on the full pupil grid (as saved), replacing pupil_func
                            after the grid is built (pupil_func is zoomed into the centre instead).
        """
        if shift_mode not in ("kout", "kin"):
            raise ValueError("shift_mode must be 'kout' or 'kin'")
        if placement not in ("bilinear", "nearest"):
            raise ValueError("placement must be 'bilinear' or 'nearest'")
        self.shift_mode = shift_mode
        self.placement = placement
        self._shift_cfg = None if streak_shifts is None else dict(groups=streak_shifts, max=shift_max)
        self._init_spectrum = init_spectrum
        self._qz_oversample = kwargs.get("qz_oversample", 1)

        super().prepare(**kwargs)
        self.PSI = None                                   # not used here; free the memory
        if init_pupil is not None:
            init_pupil = np.asarray(init_pupil, dtype=complex)
            if init_pupil.shape != self.pupil_func.shape:
                raise ValueError(f"init_pupil has shape {init_pupil.shape}, the pupil grid is "
                                 f"{self.pupil_func.shape} (same --extend and pupil file as the old run?)")
            self.pupil_func = init_pupil.copy()
            print(f"Initial pupil from file {init_pupil.shape}")
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        if self.device.type == "cuda":
            cc = torch.cuda.get_device_capability(self.device)
            built = [a for a in torch.cuda.get_arch_list() if a.startswith("sm_")]
            if built and min(int(a[3:]) for a in built) > 10 * cc[0] + cc[1]:
                raise RuntimeError(
                    f"{torch.cuda.get_device_name(self.device)} has compute capability {cc[0]}.{cc[1]}, "
                    f"but this torch build only supports {built}. Use a newer GPU (e.g. Slurm "
                    f"--constraint='A100|A6000|H100|L40S') or device='cpu'.")
        self.rdtype = _DTYPES[dtype]
        self.pupil_model = pupil_model

        # ---- internal intensity scale (undone on output) ----
        self.img_scale = float(np.mean(self.images))
        S0 = self.rec_fourier_images / np.sqrt(self.img_scale)

        # ---- gauge: rms|P| = 1 on the covered pupil ----
        cover = self.pupil_cover_mask()
        P0 = np.asarray(self.pupil_func, dtype=complex)
        c = np.sqrt(np.mean(np.abs(P0[cover])**2)) if cover.any() else 1.0
        P0, S0 = P0 / c, S0 * c

        # ---- torch objects ----
        dev, rd = self.device, self.rdtype
        self.continuous = bool(subpixel) or self._shift_cfg is not None
        if self.continuous:
            self.sampler = ShiftableEwaldSampler(self.geom, self.kins, self.kouts, self.G, self.k,
                                                 (self.kx_min_n, self.ky_min_n), (self.dkx, self.dky),
                                                 self.img_shape, dev, rd, mode=shift_mode, placement=placement)
        else:
            self.sampler = EwaldSampler(self.geom, dev, rd)
        if self._shift_cfg is not None:
            sg = self._shift_geom
            self.shifts = StreakShifts(sg["groups"], sg["n_groups"], sg["u"], sg["dk"], sg["max_shift"],
                                       real_dtype=rd, device=dev)
        self.images_t = torch.as_tensor(self.images / self.img_scale, dtype=rd, device=dev)
        self.pixel_weights_t = None
        self.pixel_threshold, self.threshold_mode = pixel_threshold, threshold_mode
        p = np.percentile(self.images, [1, 10, 25, 50, 75, 90, 99])
        print("Image intensities (input units) percentiles 1/10/25/50/75/90/99: "
              + " / ".join(f"{v:.3g}" for v in p))
        if pixel_threshold is not None:
            if threshold_mode not in ("mask", "zero"):
                raise ValueError("threshold_mode must be 'mask' or 'zero'")
            below = torch.as_tensor(self.images < pixel_threshold, device=dev)
            self.masked_pixel_fraction = float(below.float().mean())
            per_img = below.float().mean(dim=(1, 2)).cpu().numpy()
            if threshold_mode == "mask":
                self.pixel_weights_t = (~below).to(rd)
            else:
                self.images_t = torch.where(below, torch.zeros((), dtype=rd, device=dev), self.images_t)
            print(f"Pixel threshold {pixel_threshold:g} ({threshold_mode}): "
                  f"{100 * self.masked_pixel_fraction:.1f}% of all pixels below; per image "
                  f"min/median/max {100 * per_img.min():.1f}/{100 * np.median(per_img):.1f}/"
                  f"{100 * per_img.max():.1f}%, {int((per_img == 1).sum())} images fully below")
        self.cover_t = torch.as_tensor(cover, dtype=torch.bool, device=dev)
        # natural |S|: images have mean 1 and rms|P| = 1, so by Parseval (ifft divides by
        # n0*n1) rms|P S| ~ sqrt(n0*n1). Parameters are S / this, i.e. O(1).
        self.spectrum_scale = float(np.sqrt(np.prod(self.img_shape)))
        if isinstance(spectrum_mask, str):
            if spectrum_mask != "measured":
                raise ValueError("spectrum_mask must be None, 'measured' or a bool array")
            dil = self._shift_geom["qz_nodes"] if self._shift_cfg is not None else 0
            spectrum_mask = self.measured_mask(dilate=dil)
            print(f"Spectrum mask (measured voxels{f', widened by ±{dil} q_z nodes for the shifts' if dil else ''}): "
                  f"{100 * spectrum_mask.mean():.1f}% of the spectrum, "
                  f"q_z slices {np.flatnonzero(spectrum_mask.any(axis=(1, 2)))[[0, -1]].tolist()} "
                  f"of 0..{spectrum_mask.shape[0] - 1}")
        self.spectrum_mask = None if spectrum_mask is None else np.asarray(spectrum_mask, dtype=bool)
        self.spectrum = Spectrum3D(S0, scale=self.spectrum_scale, mask=spectrum_mask, real_dtype=rd, device=dev)

        if pupil_model == "pixel":
            self.pupil = PixelPupil(P0, self.ctf, freeze_amp=freeze_pupil_amp, real_dtype=rd, device=dev)
        elif pupil_model == "poly":
            k0 = self.kx_min_n + np.arange(P0.shape[0]) * self.dkx
            k1 = self.ky_min_n + np.arange(P0.shape[1]) * self.dky
            ks = np.asarray(self.ks_pupil)
            centre = (0.5 * (ks[:, 0].max() + ks[:, 0].min()), 0.5 * (ks[:, 1].max() + ks[:, 1].min()))
            if u_max is None:
                u_max = 0.5 * max(ks[:, 0].max() - ks[:, 0].min(), ks[:, 1].max() - ks[:, 1].min())
            self.u_max, self.pupil_centre = u_max, centre
            basis = square_poly_basis(k0, k1, u_max, centre)
            ap = np.abs(self.ctf) > 0
            if pupil_coeffs is None:
                from skimage.restoration import unwrap_phase
                sx0, sx1, sy0, sy1 = self.pupil_coords
                ph = np.zeros(P0.shape)
                ph[sx0:sx1, sy0:sy1] = unwrap_phase(np.angle(P0[sx0:sx1, sy0:sy1]))
                pupil_coeffs = fit_poly_coeffs(ph, basis, ap)
            amp0 = np.where(ap, np.abs(P0), 0.0)
            self.pupil = PolyPupil(basis, pupil_coeffs, amp0, self.ctf, free_amp=not freeze_pupil_amp,
                                   real_dtype=rd, device=dev)
        else:
            raise ValueError("pupil_model must be 'pixel' or 'poly'")

        self._sync_numpy()
        print(f"Optimiser engine on {self.device} ({dtype}): {self.num_images} images "
              f"{tuple(self.img_shape)}, spectrum {self.geom.obj_shape}, pupil {P0.shape} ({pupil_model})"
              f"{(', ' + ('sub-pixel' if placement == 'bilinear' else 'rounded') + ' pupil placement, continuous q_z') if self.continuous else ''}")

    # ------------------------------------------------------- shift geometry
    def _qz_margin(self, qz, dqz):
        """Called by _build_geometry: set up the streak-shift geometry and return the extra q_z
        range the shifted samples can reach."""
        if self._shift_cfg is None:
            return 0.0
        self._setup_shift_geometry(qz, dqz)
        return self._shift_geom["qz_excursion"]

    def _setup_shift_geometry(self, qz, dqz):
        cfg, N = self._shift_cfg, self.num_images
        n0, n1 = self.img_shape
        kin, kout = np.asarray(self.kins, float), np.asarray(self.kouts, float)

        g = cfg["groups"]
        if isinstance(g, str):
            if g == "streaks":
                if self.streak_idx is None:
                    raise ValueError("streak_shifts='streaks' needs the data given per streak (lists of "
                                     "images/kins/kouts); use 'qz' for a flat stack")
                raw, how = np.asarray(self.streak_idx), "input streaks"
            elif g == "qz":
                # bins of the un-oversampled spacing, offset to the circular mean of the centres
                q0 = qz[:, n0 // 2, n1 // 2]
                dq = dqz * self._qz_oversample
                frac = q0 / dq - np.round(q0 / dq)
                off = dq * np.angle(np.mean(np.exp(2j * np.pi * frac))) / (2 * np.pi)
                raw, how = np.round((q0 - off) / dq).astype(np.int64), f"q_z bins of {dq:.3e} rad/nm"
            else:
                raise ValueError("streak_shifts must be None, 'streaks', 'qz' or an int array")
        else:
            raw, how = np.asarray(g), "given groups"
            if raw.shape != (N,):
                raise ValueError(f"streak_shifts array must have shape ({N},), got {raw.shape}")
        ids, groups = np.unique(raw.astype(np.int64), return_inverse=True)
        n_groups = len(ids)

        # u: direction of grad q_z for a joint lateral shift of k_in0 and k_out (at K = 0)
        kin_z = np.sqrt(np.clip(self.k**2 - np.sum(kin[:, :2]**2, axis=1), 0, None))
        kout_z = np.sqrt(np.clip(self.k**2 - np.sum(kout[:, :2]**2, axis=1), 0, None))
        grad_kout = -kout[:, :2] / kout_z[:, None] + kin[:, :2] / kin_z[:, None]
        grad_kin = kin[:, :2] / kin_z[:, None]
        u = grad_kout.mean(axis=0)
        u = u / np.linalg.norm(u)
        dk = float(np.sqrt(self.dkx * self.dky))

        pos = kin[:, :2] @ u / dk
        centres = np.array([pos[groups == s].mean() for s in range(n_groups)])
        spacing = float(np.median(np.diff(np.sort(centres)))) if n_groups > 1 else np.nan
        from scipy.spatial import cKDTree
        pts = kin[:, :2] / dk
        nn = float(np.median(cKDTree(pts).query(pts, k=2)[0][:, 1])) if N > 1 else np.nan
        ref = spacing if np.isfinite(spacing) else nn
        max_shift = float(cfg["max"]) if cfg["max"] is not None else 1.5 * ref
        if not np.isfinite(max_shift) or max_shift <= 0:
            raise ValueError("cannot set a default shift_max (one group, one image); pass shift_max")

        # largest |dq_z / d(shift)| over the samples (rad/nm per rad/nm), for the grid margin
        Kmax = np.hypot(n0 // 2 * self.dkx, n1 // 2 * self.dky)
        kl = float(np.max(np.linalg.norm(kin[:, :2], axis=1))) + Kmax
        g_kin = kl / np.sqrt(max(self.k**2 - kl**2, 1e-30))
        g_max = g_kin + (float(np.max(np.linalg.norm(kout[:, :2] / kout_z[:, None], axis=1)))
                         if self.shift_mode == "kout" else 0.0)
        excursion = max_shift * dk * g_max
        qz_nodes = int(np.ceil(excursion / dqz)) + 1
        per_px = (np.mean(grad_kout @ u) if self.shift_mode == "kout" else np.mean(grad_kin @ u)) * dk / dqz

        self._shift_geom = dict(groups=groups, n_groups=n_groups, ids=ids, u=u, dk=dk, centres=centres,
                                spacing=spacing, nn_spacing=nn, max_shift=max_shift,
                                qz_excursion=excursion, qz_nodes=qz_nodes, dqz_per_px=per_px, how=how)
        counts = np.bincount(groups, minlength=n_groups)
        print(f"Streak shifts ({self.shift_mode} mode): {n_groups} groups ({how}), {counts.min()}-{counts.max()} "
              f"images each | u = ({u[0]:+.4f}, {u[1]:+.4f}) [x, y], {np.degrees(np.arctan2(u[1], u[0])):.1f} deg | "
              f"1 px = {dk:.4e} rad/nm, streak spacing {spacing:.2f} px, nearest-pixel spacing {nn:.2f} px | "
              f"q_z change {per_px:+.3f} dqz per px | max shift ±{max_shift:.2f} px -> q_z margin "
              f"{excursion:.2e} rad/nm ({qz_nodes} nodes)")

    # ------------------------------------------------------------- spectrum init
    def _initiate_recons_images(self, init='ones'):
        if self._init_spectrum is not None:
            self.rec_fourier_obj = self._regrid_spectrum(*self._init_spectrum)
        super()._initiate_recons_images(init)

    def _regrid_spectrum(self, S_old, qz_old):
        """Put a spectrum saved on another q_z grid (same dqz, aligned nodes) on this one."""
        S_old = np.asarray(S_old, dtype=complex)
        qz_old = np.asarray(qz_old, dtype=float)
        g = self.geom
        if S_old.shape[1:] != tuple(g.obj_shape[1:]) or S_old.shape[0] != len(qz_old):
            raise ValueError(f"init spectrum {S_old.shape} / q_z axis {qz_old.shape} do not fit the grid {g.obj_shape}")
        dq_old = float(np.median(np.diff(qz_old))) if len(qz_old) > 1 else g.dqz
        if abs(dq_old - g.dqz) > 1e-6 * g.dqz:
            raise ValueError(f"init spectrum dqz {dq_old:.6e} differs from the grid dqz {g.dqz:.6e} "
                             f"(same --dqz / --qz_oversample as the old run?)")
        off = (qz_old[0] - g.qz_axis[0]) / g.dqz
        o = int(round(off))
        if abs(off - o) > 1e-3:
            raise ValueError(f"init spectrum q_z nodes are not on this grid (offset {off:.4f} nodes)")
        S = np.zeros(g.obj_shape, dtype=complex)
        lo, hi = max(o, 0), min(o + len(qz_old), g.obj_shape[0])
        S[lo:hi] = S_old[lo - o:hi - o]
        e_all = np.sum(np.abs(S_old)**2)
        lost = 1 - np.sum(np.abs(S)**2) / e_all if e_all > 0 else 0.0
        print(f"Initial spectrum from file: Nz {len(qz_old)} -> {g.obj_shape[0]} (node offset {o})"
              f"{f', {100 * lost:.3g}% of its energy outside the new grid' if lost > 1e-9 else ''}")
        return S

    # ------------------------------------------------------------- masks
    def measured_mask(self, dilate=0):
        """Bool (Nz, n0, n1): voxels that at least one data sample interpolates from with a
        non-zero weight (independent of the pupil). All others never enter the forward model.
        dilate: widen by this many q_z nodes on each side (room for shifted samples)."""
        g = self.geom
        Nz, M = g.obj_shape[0], g.n_lat
        i0, w = g.node0.ravel(), g.w.ravel()
        cnt = (np.bincount(i0, weights=(w < 1).astype(float), minlength=Nz * M)
               + np.bincount(i0 + M, weights=(w > 0).astype(float), minlength=Nz * M))
        m = (cnt > 0).reshape(g.obj_shape)
        if dilate > 0:
            out = m.copy()
            for s in range(1, int(dilate) + 1):
                out[s:] |= m[:-s]
                out[:-s] |= m[s:]
            m = out
        return m

    def pupil_cover_mask(self):
        """Bool mask of the pupil pixels the data constrains (inside the aperture), at the
        initial geometry (it sets the P/S gauge)."""
        if getattr(self, "pupil_cover", None) is None:
            self.pupil_cover = pupil_coverage(self.geom, self.pupil_func.shape) & (np.abs(self.ctf) > 0)
        return self.pupil_cover

    def current_pupil_coverage(self):
        """Pupil pixels the patches touch at the current shifts (sub-pixel model), in the aperture."""
        if not self.continuous:
            return self.pupil_cover_mask()
        with torch.no_grad():
            dkin = None if self.shifts is None else self.shifts()
        return self.sampler.coverage(self.pupil_func.shape, dkin) & (np.abs(self.ctf) > 0)

    @torch.no_grad()
    def pupil_centre_ratio(self):
        """Per image: |P| where the image takes its K=0 sample (the pixel on the integer path,
        the bilinear value with sub-pixel placement), divided by the median |P| of the 8 pupil
        pixels around it. Returns a numpy array (N,), NaN where that median is 0.

        1 means the image centres are not special. A free-amplitude pupil can lower |P| exactly
        there while S(q_z, K=0) grows, and every image's mean level stays the same. The median
        dropping well below 1 is the precursor of the pixel-pupil blow-up (see PupilSmooth)."""
        P = self.pupil()
        P0, P1 = P.shape
        sp = self.sampler
        if self.continuous:
            dkin = None if self.shifts is None else self.shifts()
            kin = sp._kin(None, dkin)
            x0 = (kin[:, 0] - sp.k_min[0]) / sp.dk[0]
            x1 = (kin[:, 1] - sp.k_min[1]) / sp.dk[1]
            if sp.placement == "nearest":          # the patch rule: round(start) + n//2
                c0 = (torch.round(x0 - sp.c[0]).long() + sp.c[0]).clamp(0, P0 - 1)
                c1 = (torch.round(x1 - sp.c[1]).long() + sp.c[1]).clamp(0, P1 - 1)
                Pc = P[c0, c1]
            else:
                i0 = torch.floor(x0).long().clamp(0, P0 - 2)
                i1 = torch.floor(x1).long().clamp(0, P1 - 2)
                t0 = (x0 - i0).clamp(0, 1).to(self.rdtype)
                t1 = (x1 - i1).clamp(0, 1).to(self.rdtype)
                Pc = ((1 - t0) * ((1 - t1) * P[i0, i1] + t1 * P[i0, i1 + 1])
                      + t0 * ((1 - t1) * P[i0 + 1, i1] + t1 * P[i0 + 1, i1 + 1]))
                c0 = torch.round(x0).long().clamp(0, P0 - 1)
                c1 = torch.round(x1).long().clamp(0, P1 - 1)
        else:
            n0, n1 = self.img_shape
            c0, c1 = sp.pup_idx0[:, n0 // 2], sp.pup_idx1[:, n1 // 2]
            Pc = P[c0, c1]
        nb = torch.stack([torch.sqrt(abs2(P[(c0 + a).clamp(0, P0 - 1), (c1 + b).clamp(0, P1 - 1)]))
                          for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)])
        ref = torch.median(nb, dim=0).values
        r = torch.sqrt(abs2(Pc)) / ref
        r = torch.where(ref > 0, r, torch.full_like(r, float("nan")))
        return r.cpu().numpy()

    # ----------------------------------------------------------- optimisers
    def set_optimiser(self, lr_spectrum, lr_pupil=None, optimiser=torch.optim.Adam, lr_shift=None, **kwargs):
        """Separate optimisers for the spectrum, the pupil and the streak shifts (so a frozen part
        keeps its optimiser state untouched). kwargs go to all (e.g. betas, weight_decay).
        lr_shift is in pupil pixels per step (Adam); it needs prepare(streak_shifts=...)."""
        self.spectrum_optimiser = optimiser(self.spectrum.parameters(), lr=lr_spectrum, **kwargs)
        if lr_pupil is not None:
            self.pupil_optimiser = optimiser(self.pupil.parameters(), lr=lr_pupil, **kwargs)
        if lr_shift is not None:
            if self.shifts is None:
                raise ValueError("lr_shift needs prepare(streak_shifts=...)")
            if getattr(self, "placement", "bilinear") == "nearest":
                raise ValueError("placement='nearest' gives the shifts no pupil gradient: use scans only, "
                                 "or placement='bilinear' to learn them")
            self.shift_optimiser = optimiser(self.shifts.parameters(), lr=lr_shift, **kwargs)

    def set_scheduler(self, scheduler, which="spectrum", **kwargs):
        """torch.optim.lr_scheduler class for 'spectrum', 'pupil' or 'shift'; stepped once per epoch."""
        if which not in ("spectrum", "pupil", "shift"):
            raise ValueError("which must be 'spectrum', 'pupil' or 'shift'")
        opt = getattr(self, f"{which}_optimiser")
        setattr(self, f"{which}_scheduler", scheduler(opt, **kwargs))

    # ------------------------------------------------------------- forward
    def _dkin(self, idx=None, extra=None):
        return None if self.shifts is None else self.shifts(idx, extra)

    def _fields(self, S, P, idx=None, dkin=None):
        if self.continuous:
            return self.sampler.fields(S, P, idx, dkin)
        return self.sampler.fields(S, P, idx)

    # ------------------------------------------------------------- main loop
    def run(self, epochs, batch_size=None, pupil_start=0, alternate=None, regularisers=(),
            loss="l2_intensity", normalise_pupil=True, log_every=1, seed=None,
            shift_start=0, shift_scan_epochs=(), shift_scan_range=None, shift_scan_step=1.0):
        """
        epochs       : passes over all images.
        batch_size   : images per optimiser step (None = all, i.e. full-batch gradient).
        pupil_start  : epoch (counted over all run() calls) from which the pupil is learnt.
        alternate    : (n_spec, n_pup) - cycle n_spec spectrum-only epochs with n_pup
                       pupil-only epochs (after pupil_start). None = both every step.
        regularisers : iterable of Regulariser; each added as weight(epoch) * penalty.
        loss         : 'l2_intensity', 'l2_amplitude' or 'poisson' (see losses.py).
        normalise_pupil : after each pupil step, rescale P -> P/c, S -> S*c (rms|P| = 1 on the
                       covered pupil). The model output is unchanged.
        shift_start  : epoch from which the streak shifts are learnt (needs set_optimiser(lr_shift=..));
                       they are then updated at every step, independent of `alternate`.
        shift_scan_epochs : epochs at which scan_streak_shifts runs (before that epoch's steps; an
                       epoch equal to the final epoch count runs after the last one, so epochs=0
                       with scan epoch = current epoch only scans).
        shift_scan_range, shift_scan_step : trial offsets -range..range in steps (pupil px);
                       range None = the shift bound.
        """
        if self.spectrum_optimiser is None:
            raise RuntimeError("call set_optimiser() first")
        scan_epochs = sorted(set(int(e) for e in shift_scan_epochs))
        if scan_epochs and self.shifts is None:
            raise ValueError("shift scans need prepare(streak_shifts=...)")
        self._loss_name = loss
        loss_fn, norm_fn = get_loss(loss)
        I_all = self.images_t
        W_all = self.pixel_weights_t                      # None, or (N, n0, n1) 0/1
        norm = norm_fn(I_all, W_all)
        norm_amp = torch.sum(I_all) if W_all is None else torch.sum(W_all * I_all)
        N = self.num_images
        B = N if batch_size is None else min(int(batch_size), N)
        self._scan_batch = B
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        regularisers = list(regularisers)
        names = [r.name for r in regularisers]
        if len(set(names)) != len(names):
            raise ValueError(f"regulariser names must be unique, got {names}")
        for r in regularisers:
            self.reg_losses.setdefault(r.name, [])
            self.reg_weights.setdefault(r.name, [])
            if hasattr(r, "setup"):
                r.setup(self)
            self.reg_info[r.name] = r.describe() if hasattr(r, "describe") else r.name
            print(f"Regulariser {self.reg_info[r.name]}")
        self._regularisers = regularisers

        def scan_now():
            R = self.shifts.max_shift if shift_scan_range is None else float(shift_scan_range)
            m = int(np.ceil(R / shift_scan_step - 1e-9))
            offsets = shift_scan_step * np.arange(-m, m + 1)          # symmetric, contains 0
            self.scan_streak_shifts(offsets, apply=True, batch_size=B)

        for _ in range(epochs):
            ep = self.iters_passed
            t0 = time.time()
            if ep in scan_epochs:
                scan_now()
            for r in regularisers:
                if hasattr(r, "on_epoch"):
                    r.on_epoch(ep, self)        # e.g. shrinkwrap
            learn_pupil = self.pupil_optimiser is not None and ep >= pupil_start
            do_spec, do_pup = True, learn_pupil
            if learn_pupil and alternate is not None:
                n_s, n_p = alternate
                do_spec = ((ep - pupil_start) % (n_s + n_p)) < n_s
                do_pup = not do_spec
            do_shift = self.shift_optimiser is not None and ep >= shift_start
            for p in self.spectrum.parameters():
                p.requires_grad_(do_spec)
            for p in self.pupil.parameters():
                p.requires_grad_(do_pup)
            if self.shifts is not None:
                for p in self.shifts.parameters():
                    p.requires_grad_(do_shift)

            order = torch.randperm(N, generator=gen) if B < N else torch.arange(N)
            ep_loss, ep_misfit, ep_total = 0.0, 0.0, 0.0
            ep_regs = {r.name: 0.0 for r in regularisers}
            weights = {r.name: float(r.weight_at(ep)) for r in regularisers}
            # weighted terms first: they must build obj with grad before it is cached
            active = [r for r in regularisers if weights[r.name]]
            tracked = [r for r in regularisers if not weights[r.name]]

            for b0 in range(0, N, B):
                idx = order[b0:b0 + B].to(self.device)
                frac = len(idx) / N
                if do_spec:
                    self.spectrum_optimiser.zero_grad(set_to_none=True)
                if do_pup:
                    self.pupil_optimiser.zero_grad(set_to_none=True)
                if do_shift:
                    self.shift_optimiser.zero_grad(set_to_none=True)

                S = self.spectrum()
                P = self.pupil()
                I_pred = abs2(ifft2c(self._fields(S, P, idx, self._dkin(idx))))
                I_meas = I_all[idx]
                W = None if W_all is None else W_all[idx]
                total = loss_fn(I_pred, I_meas, norm, W)
                ep_loss += total.item()

                if regularisers:
                    ctx = RegContext(S, P)
                    for r in active:
                        pen = r.penalty(ctx)
                        total = total + frac * weights[r.name] * pen
                        ep_regs[r.name] += frac * pen.item()
                    with torch.no_grad():                  # weight 0: value tracked only
                        for r in tracked:
                            ep_regs[r.name] += frac * r.penalty(ctx).item()
                ep_total += total.item()

                with torch.no_grad():
                    r2 = (torch.sqrt(I_pred) - torch.sqrt(I_meas))**2
                    ep_misfit += ((torch.sum(r2) if W is None else torch.sum(W * r2)) / norm_amp).item()

                if not torch.isfinite(total):
                    raise FloatingPointError(f"non-finite loss at epoch {ep}")
                total.backward()

                if do_spec:
                    self.spectrum_optimiser.step()
                if do_pup:
                    self.pupil_optimiser.step()
                    if normalise_pupil:
                        self._normalise_gauge()
                if do_shift:
                    self.shift_optimiser.step()
                    self.shifts.gauge_()
                    self.shifts.clamp_()

            if do_spec and self.spectrum_scheduler is not None:
                self.spectrum_scheduler.step()
            if do_pup and self.pupil_scheduler is not None:
                self.pupil_scheduler.step()
            if do_shift and self.shift_scheduler is not None:
                self.shift_scheduler.step()

            self.losses.append(ep_loss)
            self.total_losses.append(ep_total)
            self.data_misfits.append(ep_misfit)
            for r in regularisers:
                self.reg_losses[r.name].append(ep_regs[r.name])
                self.reg_weights[r.name].append(weights[r.name])
            self.epoch_times.append(time.time() - t0)
            self.iters_passed += 1

            if log_every and (ep % log_every == 0):
                if self.shifts is not None:
                    self.shift_history.append(self.shifts.numpy())
                    self.shift_history_epochs.append(ep)
                regs = "".join(f", {k} = {v:.4e} (w {weights[k]:.2e})" for k, v in ep_regs.items())
                tot = f", total = {ep_total:.4e}" if regularisers else ""
                what = "+".join(n for n, on in (("S", do_spec), ("P", do_pup), ("shift", do_shift)) if on)
                sh = ""
                if self.shifts is not None:
                    d = self.shifts.numpy()
                    sh = f", shifts rms {np.sqrt(np.mean(d**2)):.2f} max {np.abs(d).max():.2f} px"
                print(f"Epoch {ep} [{what}]: loss = {ep_loss:.4e}{tot}, data misfit = {ep_misfit:.4e}"
                      f"{regs}{sh} ({self.epoch_times[-1]:.2f} s)")

        ep = self.iters_passed
        if ep in scan_epochs and not any(s["epoch"] == ep for s in self.shift_scans):
            scan_now()
        if self.continuous and (self.sampler.n_clamped_pupil or self.sampler.n_clamped_qz):
            print(f"Warning: {self.sampler.n_clamped_pupil} pupil patches and {self.sampler.n_clamped_qz} q_z "
                  f"samples fell off their grid and were clamped (larger --extend / --nz_pad, smaller shift_max)")
        self._sync_numpy()

    @torch.no_grad()
    def _normalise_gauge(self):
        P = self.pupil()
        c = torch.sqrt(torch.mean(abs2(P[self.cover_t]))).item()
        if np.isfinite(c) and c > 0:
            self.pupil.scale_(c)
            self.spectrum.scale_(c)

    # ------------------------------------------------------- shift scan
    @torch.no_grad()
    def scan_streak_shifts(self, offsets, apply=True, batch_size=None, loss=None, min_energy_frac=0.5):
        """Grid search of the streak offsets with S and P fixed.

        For every trial t in `offsets` (pupil px, relative to the current offsets) the data loss of
        every image is computed with its streak moved by t, and summed per streak: L[t, s]. Each
        image depends only on its own streak's offset, so one pass over the images covers all
        streaks. The best t per streak (parabolic sub-step around the minimum) is applied if
        `apply` (then the gauge and the bound). Returns (offsets, L, best); the scan is stored in
        self.shift_scans and saved by save_reconsturction.

        A best offset at the end of the range is never applied (not a bracketed minimum).
        min_energy_frac : a streak's best offset is only applied if its predicted intensity there
                          is at least this fraction of that at the current offset. With S and P
                          fixed, a streak the model fits badly can 'improve' by moving where the
                          model predicts little (out of the aperture or the measured q_z range):
                          that is not a registration. Rejected streaks keep their offset.
        """
        if self.shifts is None:
            raise ValueError("scan_streak_shifts needs prepare(streak_shifts=...)")
        offsets = np.asarray(offsets, dtype=float)
        loss_fn, norm_fn = get_loss(loss or getattr(self, "_loss_name", "l2_intensity"))
        I_all, W_all = self.images_t, self.pixel_weights_t
        norm = norm_fn(I_all, W_all)
        S, P = self.spectrum(), self.pupil()
        N, G = self.num_images, self.shifts.n_groups
        B = N if batch_size is None else min(int(batch_size), N)
        groups = self.shifts.groups
        L = torch.zeros((len(offsets), G), dtype=torch.float64, device=self.device)
        E = torch.zeros((len(offsets), G), dtype=torch.float64, device=self.device)   # predicted intensity
        t0 = time.time()
        for it, t in enumerate(offsets):
            for b0 in range(0, N, B):
                idx = torch.arange(b0, min(N, b0 + B), device=self.device)
                I = abs2(ifft2c(self.sampler.fields(S, P, idx, self.shifts(idx, extra=float(t)))))
                W = None if W_all is None else W_all[idx]
                li = loss_fn(I, I_all[idx], norm, W, per_image=True)
                L[it].index_add_(0, groups[idx], li.to(torch.float64))
                E[it].index_add_(0, groups[idx], torch.sum(I, dim=(-2, -1)).to(torch.float64))
        L, E = L.cpu().numpy(), E.cpu().numpy()

        j = np.argmin(L, axis=0)
        best = offsets[j].copy()
        step = offsets[1] - offsets[0] if len(offsets) > 1 else 0.0
        for s in range(G):
            if 0 < j[s] < len(offsets) - 1:
                a, b, c = L[j[s] - 1, s], L[j[s], s], L[j[s] + 1, s]
                den = a - 2 * b + c
                if den > 0:
                    best[s] += 0.5 * (a - c) / den * step
        at_edge = (j == 0) | (j == len(offsets) - 1)
        i0 = int(np.argmin(np.abs(offsets)))
        has0 = abs(offsets[i0]) < 1e-9
        gain = (L[i0] - L.min(axis=0)) / np.maximum(L[i0], 1e-300) if has0 else np.full(G, np.nan)
        # energy guard: the predicted intensity at the best offset vs at the current one
        e_ratio = E[j, np.arange(G)] / np.maximum(E[i0], 1e-300) if has0 else np.ones(G)
        accept = (e_ratio >= min_energy_frac) & ~at_edge        # an edge minimum is not bracketed
        step_applied = np.where(accept, best, 0.0)
        before = self.shifts.numpy()
        if apply:
            self.shifts.set_(before + step_applied)
            self.shifts.gauge_()
            self.shifts.clamp_()
        after = self.shifts.numpy()
        self.shift_scans.append(dict(epoch=self.iters_passed, offsets=offsets, L=L, E=E, best=best,
                                     accepted=accept, delta_before=before, delta_after=after, applied=bool(apply)))
        rej = np.flatnonzero(~accept)
        print(f"Shift scan at epoch {self.iters_passed}: {len(offsets)} offsets {offsets[0]:+g}..{offsets[-1]:+g} px, "
              f"{time.time() - t0:.1f} s | best offset per streak rms {np.sqrt(np.mean(best**2)):.2f}, "
              f"max |.| {np.abs(best).max():.2f} px, {int(at_edge.sum())}/{G} at the range edge | "
              f"loss gain vs 0: median {100 * np.nanmedian(gain):.3g}%, max {100 * np.nanmax(gain):.3g}%"
              f"{f' | not applied (minimum at the range edge, or predicted intensity < {min_energy_frac:g} of current): streaks {rej.tolist()}' if len(rej) else ''}"
              f"{' | applied, shifts now rms %.2f px' % np.sqrt(np.mean(after**2)) if apply else ''}")
        return offsets, L, best

    # ------------------------------------------------------------- results
    @torch.no_grad()
    def _sync_numpy(self):
        """Copy the current model to the numpy attributes used by plotting and saving,
        in the units of the input images."""
        S = self.spectrum().cpu().numpy() * np.sqrt(self.img_scale)
        self.rec_fourier_images = S
        self.pupil_func = self.pupil().cpu().numpy()
        self.obj_3D = self.inverse_fft_3d(S)
        self.rec_obj_images = self.obj_3D
        if self.shifts is not None:
            self.kins_refined, self.kouts_refined = self.refined_k_vectors()

    @torch.no_grad()
    def refined_k_vectors(self):
        """(kins, kouts) (N, 3) with the current streak shifts applied (on the Ewald sphere);
        kouts only move in 'kout' mode."""
        kin, kout = np.array(self.kins, dtype=float), np.array(self.kouts, dtype=float)
        if self.shifts is None:
            return kin, kout
        d = self.shifts().cpu().numpy().astype(float)
        kin[:, :2] += d
        kin[:, 2] = np.sqrt(np.clip(self.k**2 - np.sum(kin[:, :2]**2, axis=1), 0, None))
        if self.shift_mode == "kout":
            ko = kout[:, :2].copy()
            r0 = np.sqrt(np.clip(self.k**2 - np.sum(ko**2, axis=1), 0, None))
            r1 = np.sqrt(np.clip(self.k**2 - np.sum((ko + d)**2, axis=1), 0, None))
            kout[:, 2] -= (2 * np.sum(ko * d, axis=1) + np.sum(d**2, axis=1)) / (r0 + r1)
            kout[:, :2] += d
        return kin, kout

    @torch.no_grad()
    def current_object(self):
        """Current 3D object [z, x, y] (internal scale), numpy."""
        return ifft3c(self.spectrum()).cpu().numpy()

    @torch.no_grad()
    def predict_intensities(self, idx=None):
        """Model intensities in the input-image units, (N or B, n0, n1) numpy."""
        if idx is not None:
            idx = torch.as_tensor(idx, device=self.device)
        I = abs2(ifft2c(self._fields(self.spectrum(), self.pupil(), idx, self._dkin(idx))))
        return I.cpu().numpy() * self.img_scale

    def save_reconsturction(self, file_path, metadata={}):
        metadata = dict(metadata, engine="OptimFourierPtycho3D", pupil_model=self.pupil_model,
                        image_scale=self.img_scale,
                        pixel_threshold=str(getattr(self, "pixel_threshold", None)),
                        threshold_mode=str(getattr(self, "threshold_mode", None)),
                        masked_pixel_fraction=getattr(self, "masked_pixel_fraction", 0.0),
                        subpixel_pupil=bool(self.continuous and self.placement == "bilinear"),
                        pupil_placement=str(self.placement if self.continuous else "rounded (integer path)"),
                        streak_shifts=self.shifts is not None,
                        shift_mode=str(self.shift_mode if self.shifts is not None else None))
        if self.continuous:
            self.pupil_cover = self.current_pupil_coverage()     # saved as Pupil_coverage
        super().save_reconsturction(file_path, metadata)
        with h5py.File(file_path, "a") as h5f:
            losses = h5f["Losses"]
            losses["main_loss_values"].attrs["note"] = "data loss per epoch (normalised, see losses.py)"
            losses.create_dataset("epoch_times_s", data=np.array(self.epoch_times))
            losses.create_dataset("total_loss_values", data=np.array(self.total_losses))
            losses["total_loss_values"].attrs["note"] = "data loss + sum_r weight_r * reg_r, per epoch"
            for name, vals in self.reg_losses.items():
                d = losses.create_dataset(f"reg_{name}", data=np.array(vals))
                d.attrs["note"] = "unweighted penalty per epoch (also tracked while its weight is 0)"
                d.attrs["regulariser"] = self.reg_info.get(name, name)
                losses.create_dataset(f"reg_{name}_weight", data=np.array(self.reg_weights[name]))
            if getattr(self, "spectrum_mask", None) is not None:
                ds = h5f["Reconstructed_data"].create_dataset("Spectrum_mask", data=self.spectrum_mask,
                                                              compression="gzip")
                ds.attrs["note"] = "S held at 0 outside (q_z, Kx, Ky)"
            for r in getattr(self, "_regularisers", []):
                for key, arr in (r.save_arrays().items() if hasattr(r, "save_arrays") else []):
                    if arr is not None:
                        ds = h5f["Reconstructed_data"].create_dataset(key, data=arr, compression="gzip")
                        ds.attrs["regulariser"] = self.reg_info.get(r.name, r.name)
                        if hasattr(r, "n_updates"):
                            ds.attrs["shrinkwrap_updates"] = r.n_updates
            if self.pupil_model == "poly":
                g = h5f["Reconstructed_data"]
                g.create_dataset("Pupil_poly_coeffs", data=self.pupil.coeffs.detach().cpu().numpy())
                g["Pupil_poly_coeffs"].attrs["note"] = ("S1..S29 square-polynomial phase coefficients (rad); "
                                                       "x = (k0-c0)/u_max/sqrt2, y = (k1-c1)/u_max/sqrt2")
                g["Pupil_poly_coeffs"].attrs["u_max_rad_nm_inv"] = self.u_max
                g["Pupil_poly_coeffs"].attrs["centre_rad_nm_inv"] = self.pupil_centre
            if self.shifts is not None:
                self._save_shifts(h5f)

    def _save_shifts(self, h5f):
        sg = self._shift_geom
        g = h5f["Reconstructed_data"]
        ds = g.create_dataset("Streak_shifts", data=self.shifts.numpy())
        ds.attrs["units"] = "pupil pixels along u (1 px = dk rad/nm)"
        ds.attrs["dk_rad_nm_inv"] = sg["dk"]
        ds.attrs["u_xy"] = sg["u"]
        ds.attrs["mode"] = self.shift_mode
        ds.attrs["groups"] = sg["how"]
        ds.attrs["streak_spacing_px"] = sg["spacing"]
        ds.attrs["nearest_pixel_spacing_px"] = sg["nn_spacing"]
        ds.attrs["dqz_per_px"] = sg["dqz_per_px"]
        ds.attrs["max_shift_px"] = sg["max_shift"]
        ds.attrs["note"] = ("dkin_j = dk * Streak_shifts[Streak_groups[j]] * u; 'kout' mode: k_out moves too "
                            "(q_z follows), 'kin' mode: only k_in moves. The weighted mean is held at 0 (gauge).")
        g.create_dataset("Streak_groups", data=sg["groups"].astype(np.int32), compression="gzip")
        g["Streak_groups"].attrs["note"] = "group (index into Streak_shifts) of every image, in input order"
        g.create_dataset("Streak_ids", data=sg["ids"])
        g["Streak_ids"].attrs["note"] = "input streak index (or q_z bin) of every group"
        g.create_dataset("Streak_centres", data=sg["centres"])
        g["Streak_centres"].attrs["note"] = "initial mean position of each group along u (pupil px)"
        kin, kout = self.refined_k_vectors()
        g.create_dataset("Kins_refined", data=kin, compression="gzip")
        g.create_dataset("Kouts_refined", data=kout, compression="gzip")
        g["Kins_refined"].attrs["note"] = "k_in (rad/nm) of every image with the streak shifts applied"
        if self.shift_history:
            lg = h5f["Losses"]
            lg.create_dataset("streak_shift_history", data=np.array(self.shift_history))
            lg.create_dataset("streak_shift_history_epochs", data=np.array(self.shift_history_epochs))
        if self.shift_scans:
            sc = g.create_group("Streak_shift_scans")
            for i, s in enumerate(self.shift_scans):
                d = sc.create_dataset(f"scan_{i}", data=s["L"])
                d.attrs["note"] = "per-streak data loss [offset, group] with S and P fixed"
                d.attrs["epoch"] = s["epoch"]
                d.attrs["offsets_px"] = s["offsets"]
                d.attrs["best_px"] = s["best"]
                d.attrs["accepted"] = s["accepted"]
                e = sc.create_dataset(f"scan_{i}_predicted_intensity", data=s["E"])
                e.attrs["note"] = "sum of predicted intensities per [offset, group] (energy guard)"
                d.attrs["delta_before_px"] = s["delta_before"]
                d.attrs["delta_after_px"] = s["delta_after"]
                d.attrs["applied"] = s["applied"]
