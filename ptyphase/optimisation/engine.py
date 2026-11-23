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

Units rad/nm; layout images/pupil [x, y], spectrum [qz, K0, K1], object [z, x, y].
"""
import time

import h5py
import numpy as np
import torch

from ..projection.phase_abstract_3d import PhaseRetrieval3DBase, Plot3D
from ..projection.algorithms_3d import pupil_coverage
from .forward import EwaldSampler, abs2, ifft2c
from .losses import get_loss
from .models import Spectrum3D, PixelPupil, PolyPupil, square_poly_basis, fit_poly_coeffs
from .regularisers import RegContext

_DTYPES = {"float64": torch.float64, "float32": torch.float32}


class OptimFourierPtycho3D(PhaseRetrieval3DBase, Plot3D):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reg_losses = {}
        self.epoch_times = []
        self.spectrum_optimiser = self.pupil_optimiser = None
        self.spectrum_scheduler = self.pupil_scheduler = None

    # ------------------------------------------------------------- prepare
    def prepare(self, device="cuda", dtype="float64", pupil_model="pixel",
                freeze_pupil_amp=False, pupil_coeffs=None, u_max=None,
                spectrum_mask=None, **kwargs):
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
        """
        super().prepare(**kwargs)
        self.PSI = None                                   # not used here; free the memory
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
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
        self.sampler = EwaldSampler(self.geom, dev, rd)
        self.images_t = torch.as_tensor(self.images / self.img_scale, dtype=rd, device=dev)
        self.cover_t = torch.as_tensor(cover, dtype=torch.bool, device=dev)
        # natural |S|: images have mean 1 and rms|P| = 1, so by Parseval (ifft divides by
        # n0*n1) rms|P S| ~ sqrt(n0*n1). Parameters are S / this, i.e. O(1).
        self.spectrum_scale = float(np.sqrt(np.prod(self.img_shape)))
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
              f"{tuple(self.img_shape)}, spectrum {self.geom.obj_shape}, pupil {P0.shape} ({pupil_model})")

    def pupil_cover_mask(self):
        """Bool mask of the pupil pixels the data constrains (inside the aperture)."""
        if getattr(self, "pupil_cover", None) is None:
            self.pupil_cover = pupil_coverage(self.geom, self.pupil_func.shape) & (np.abs(self.ctf) > 0)
        return self.pupil_cover

    # ----------------------------------------------------------- optimisers
    def set_optimiser(self, lr_spectrum, lr_pupil=None, optimiser=torch.optim.Adam, **kwargs):
        """Separate optimisers for the spectrum and the pupil (so a frozen pupil keeps its
        optimiser state untouched). kwargs go to both (e.g. betas, weight_decay)."""
        self.spectrum_optimiser = optimiser(self.spectrum.parameters(), lr=lr_spectrum, **kwargs)
        if lr_pupil is not None:
            self.pupil_optimiser = optimiser(self.pupil.parameters(), lr=lr_pupil, **kwargs)

    def set_scheduler(self, scheduler, which="spectrum", **kwargs):
        """torch.optim.lr_scheduler class for 'spectrum' or 'pupil'; stepped once per epoch."""
        opt = self.spectrum_optimiser if which == "spectrum" else self.pupil_optimiser
        setattr(self, f"{which}_scheduler", scheduler(opt, **kwargs))

    # ------------------------------------------------------------- main loop
    def run(self, epochs, batch_size=None, pupil_start=0, alternate=None, regularisers=(),
            loss="l2_intensity", normalise_pupil=True, log_every=1, seed=None):
        """
        epochs       : passes over all images.
        batch_size   : images per optimiser step (None = all, i.e. full-batch gradient).
        pupil_start  : epoch (counted over all run() calls) from which the pupil is learnt.
        alternate    : (n_spec, n_pup) - cycle n_spec spectrum-only epochs with n_pup
                       pupil-only epochs (after pupil_start). None = both every step.
        regularisers : iterable of Regulariser; each added as weight(epoch) * penalty.
        loss         : 'l2_intensity' or 'l2_amplitude'.
        normalise_pupil : after each pupil step, rescale P -> P/c, S -> S*c (rms|P| = 1 on the
                       covered pupil). The model output is unchanged.
        """
        if self.spectrum_optimiser is None:
            raise RuntimeError("call set_optimiser() first")
        loss_fn, norm_fn = get_loss(loss)
        I_all = self.images_t
        norm = norm_fn(I_all)
        norm_amp = torch.sum(I_all)
        N = self.num_images
        B = N if batch_size is None else min(int(batch_size), N)
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        regularisers = list(regularisers)
        for r in regularisers:
            self.reg_losses.setdefault(r.name, [])

        for _ in range(epochs):
            ep = self.iters_passed
            t0 = time.time()
            learn_pupil = self.pupil_optimiser is not None and ep >= pupil_start
            do_spec, do_pup = True, learn_pupil
            if learn_pupil and alternate is not None:
                n_s, n_p = alternate
                do_spec = ((ep - pupil_start) % (n_s + n_p)) < n_s
                do_pup = not do_spec
            for p in self.spectrum.parameters():
                p.requires_grad_(do_spec)
            for p in self.pupil.parameters():
                p.requires_grad_(do_pup)

            order = torch.randperm(N, generator=gen) if B < N else torch.arange(N)
            ep_loss, ep_misfit = 0.0, 0.0
            ep_regs = {r.name: 0.0 for r in regularisers}

            for b0 in range(0, N, B):
                idx = order[b0:b0 + B].to(self.device)
                frac = len(idx) / N
                if do_spec:
                    self.spectrum_optimiser.zero_grad(set_to_none=True)
                if do_pup:
                    self.pupil_optimiser.zero_grad(set_to_none=True)

                S = self.spectrum()
                P = self.pupil()
                I_pred = abs2(ifft2c(self.sampler.fields(S, P, idx)))
                I_meas = I_all[idx]
                total = loss_fn(I_pred, I_meas, norm)
                ep_loss += total.item()

                if regularisers:
                    ctx = RegContext(S, P)
                    for r in regularisers:
                        w = r.weight_at(ep)
                        if w:
                            pen = r.penalty(ctx)
                            total = total + frac * w * pen
                            ep_regs[r.name] += frac * pen.item()

                with torch.no_grad():
                    ep_misfit += (torch.sum((torch.sqrt(I_pred) - torch.sqrt(I_meas))**2) / norm_amp).item()

                if not torch.isfinite(total):
                    raise FloatingPointError(f"non-finite loss at epoch {ep}")
                total.backward()

                if do_spec:
                    self.spectrum_optimiser.step()
                if do_pup:
                    self.pupil_optimiser.step()
                    if normalise_pupil:
                        self._normalise_gauge()

            if do_spec and self.spectrum_scheduler is not None:
                self.spectrum_scheduler.step()
            if do_pup and self.pupil_scheduler is not None:
                self.pupil_scheduler.step()

            self.losses.append(ep_loss)
            self.data_misfits.append(ep_misfit)
            for r in regularisers:
                self.reg_losses[r.name].append(ep_regs[r.name])
            self.epoch_times.append(time.time() - t0)
            self.iters_passed += 1

            if log_every and (ep % log_every == 0):
                regs = "".join(f", {k} = {v:.4e}" for k, v in ep_regs.items())
                what = ("S+P" if do_spec and do_pup else "S" if do_spec else "P")
                print(f"Epoch {ep} [{what}]: loss = {ep_loss:.4e}, data misfit = {ep_misfit:.4e}"
                      f"{regs} ({self.epoch_times[-1]:.2f} s)")

        self._sync_numpy()

    @torch.no_grad()
    def _normalise_gauge(self):
        P = self.pupil()
        c = torch.sqrt(torch.mean(abs2(P[self.cover_t]))).item()
        if np.isfinite(c) and c > 0:
            self.pupil.scale_(c)
            self.spectrum.scale_(c)

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

    @torch.no_grad()
    def predict_intensities(self, idx=None):
        """Model intensities in the input-image units, (N or B, n0, n1) numpy."""
        I = self.sampler.intensities(self.spectrum(), self.pupil(), idx)
        return I.cpu().numpy() * self.img_scale

    def save_reconsturction(self, file_path, metadata={}):
        metadata = dict(metadata, engine="OptimFourierPtycho3D", pupil_model=self.pupil_model,
                        image_scale=self.img_scale)
        super().save_reconsturction(file_path, metadata)
        with h5py.File(file_path, "a") as h5f:
            losses = h5f["Losses"]
            losses["main_loss_values"].attrs["note"] = "data loss per epoch (normalised, see losses.py)"
            losses.create_dataset("epoch_times_s", data=np.array(self.epoch_times))
            for name, vals in self.reg_losses.items():
                losses.create_dataset(f"reg_{name}", data=np.array(vals))
            if self.pupil_model == "poly":
                g = h5f["Reconstructed_data"]
                g.create_dataset("Pupil_poly_coeffs", data=self.pupil.coeffs.detach().cpu().numpy())
                g["Pupil_poly_coeffs"].attrs["note"] = ("S1..S29 square-polynomial phase coefficients (rad); "
                                                       "x = (k0-c0)/u_max/sqrt2, y = (k1-c1)/u_max/sqrt2")
                g["Pupil_poly_coeffs"].attrs["u_max_rad_nm_inv"] = self.u_max
                g["Pupil_poly_coeffs"].attrs["centre_rad_nm_inv"] = self.pupil_centre
