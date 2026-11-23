"""
Regularisers. A regulariser is a weighted penalty on the current model, added to the data loss:

    total = loss_data + sum_r  w_r(epoch) * r.penalty(ctx)

`ctx` (RegContext) gives the spectrum S (Nz, n0, n1), the object obj = IFFT3c(S) [z, x, y]
(computed lazily, once per step, shared by all regularisers) and the pupil P, all as torch
tensors attached to the autograd graph.

Weight schedule: constant `weight`, switched on at epoch `start` and ramped linearly to `weight`
over `ramp` epochs; or any `scheduler` with get_alpha(epoch) (e.g. ptyphase.schedulers.WeightScheduler),
which then replaces that rule.

The engine evaluates every regulariser at every step, also while its weight is 0, so its value
is tracked per epoch from the start (Losses/reg_<name> in the saved h5).

Subclass and implement `penalty`:

    class L1Object(Regulariser):
        name = "l1_obj"
        def penalty(self, ctx):
            return ctx.obj.abs().mean()
"""
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from .forward import ifft3c


class RegContext:
    def __init__(self, S, P):
        self.S = S
        self.P = P
        self._obj = None

    @property
    def obj(self):
        if self._obj is None:
            self._obj = ifft3c(self.S)
        return self._obj


class Regulariser:
    name = "reg"

    def __init__(self, weight=1.0, start=0, ramp=0, scheduler=None):
        self.weight = float(weight)
        self.start = int(start)
        self.ramp = int(ramp)
        self.scheduler = scheduler

    def weight_at(self, epoch):
        if self.scheduler is not None:
            return self.scheduler.get_alpha(epoch)
        if epoch < self.start:
            return 0.0
        if self.ramp > 0 and epoch < self.start + self.ramp:
            return self.weight * (epoch - self.start + 1) / self.ramp
        return self.weight

    def penalty(self, ctx):
        raise NotImplementedError

    def describe(self):
        return f"{self.name}: weight {self.weight:g}, start {self.start}, ramp {self.ramp}"


class TV3D(Regulariser):
    """Isotropic 3D total variation of the object o(z, x, y):

        TV = mean_voxels sqrt( sum_a  (w_a * D_a f)^2 + delta^2 )  /  rms(f)

    f = |o| (mode 'amplitude', default: the phase - the displacement field - is left free) or
    o itself (mode 'complex': penalises phase variations too, so phase wraps and strain cost).
    D_a are forward differences along z, x, y (no wrap-around; the box edge is not a jump).

    axis_weights : (w_z, w_x, w_y) multiply each difference. Use e.g. (1/dz, 1/dx, 1/dy) * dx to
                   penalise per nm instead of per voxel when dz != dx. Default (1, 1, 1).
    delta        : smoothing, relative to rms(f); keeps the gradient finite where D f = 0.

    Dividing by rms(f) makes the penalty scale-invariant (TV(c f) = TV(f)): its value is ~ the
    mean relative jump between neighbouring voxels, so weights carry over between datasets and
    units, and it cannot be lowered by shrinking the whole object (plain TV pulls the object's
    scale towards 0, which the data term then has to fight). Only the shape is penalised.
    """
    name = "tv"

    def __init__(self, weight=1.0, start=0, ramp=0, scheduler=None, mode="amplitude",
                 axis_weights=(1.0, 1.0, 1.0), delta=1e-3):
        super().__init__(weight, start, ramp, scheduler)
        if mode not in ("amplitude", "complex"):
            raise ValueError("TV3D mode must be 'amplitude' or 'complex'")
        self.mode = mode
        self.axis_weights = tuple(float(a) for a in axis_weights)
        self.delta = float(delta)

    def penalty(self, ctx):
        o = ctx.obj
        if self.mode == "amplitude":
            f = torch.sqrt(o.real**2 + o.imag**2 + 1e-30)
        else:
            f = o
        ref = torch.sqrt(torch.mean(f.real**2 + (f.imag**2 if f.is_complex() else 0)) + 1e-60)
        base = f[:-1, :-1, :-1]
        wz, wx, wy = self.axis_weights
        g2 = 0.0
        for w, nb in ((wz, f[1:, :-1, :-1]), (wx, f[:-1, 1:, :-1]), (wy, f[:-1, :-1, 1:])):
            d = (nb - base) * w
            g2 = g2 + (d.real**2 + d.imag**2 if d.is_complex() else d**2)
        return torch.mean(torch.sqrt(g2 + (self.delta * ref)**2)) / ref

    def describe(self):
        return (super().describe() + f", mode {self.mode}, axis_weights {self.axis_weights}, "
                f"delta {self.delta:g}")


class Support3D(Regulariser):
    """Real-space support penalty: the fraction of the object's energy outside the support,

        pen = sum |(1 - supp) * o|^2  /  sum |o|^2        (0 = all energy inside, 1 = none)

    Scale-invariant like TV3D, so weights carry over between datasets.

    Initial support: a box centred on the origin (z = x = y = 0) of size `size_nm`:
        (sz,)          -> |z| <= sz/2, all x and y
        (sz, sx, sy)   -> |z| <= sz/2, |x| <= sx/2, |y| <= sy/2
    (the axes are the saved Axes/z_nm, x_nm, y_nm).

    Shrinkwrap (only when `sw_threshold` is given): every `sw_every` epochs from `sw_start`, the
    support becomes  gaussian_filter(|o|, sw_sigma voxels) > sw_threshold * max. With
    `sw_in_box` (default) the new support is intersected with the initial box, so the box (e.g.
    the z extent) keeps being enforced; otherwise it replaces the support. If the new support
    would have fewer than `sw_min_frac` of the voxels, the old one is kept (shrinkwrap can
    collapse when the object is still poor).
    """
    name = "support"

    def __init__(self, size_nm, weight=1.0, start=0, ramp=0, scheduler=None,
                 sw_threshold=None, sw_sigma=1.0, sw_every=100, sw_start=0, sw_in_box=True,
                 sw_min_frac=1e-3):
        super().__init__(weight, start, ramp, scheduler)
        size_nm = tuple(float(s) for s in size_nm)
        if len(size_nm) not in (1, 3):
            raise ValueError("support size_nm must be (sz,) or (sz, sx, sy)")
        self.size_nm = size_nm
        self.sw_threshold = None if sw_threshold is None else float(sw_threshold)
        self.sw_sigma = float(sw_sigma)
        self.sw_every = max(int(sw_every), 1)
        self.sw_start = int(sw_start)
        self.sw_in_box = bool(sw_in_box)
        self.sw_min_frac = float(sw_min_frac)
        self.box = self.support = None      # numpy bool (Nz, n0, n1)
        self._mask_t = None                 # torch, 1 outside the support
        self.n_updates = 0

    # ---- engine hooks
    def setup(self, engine):
        """Build the box on the engine's grid (called by run(); kept across run() calls)."""
        if self.box is not None:
            return
        axes = (engine.z_axis, engine.x_axis, engine.y_axis)
        sizes = self.size_nm if len(self.size_nm) == 3 else (self.size_nm[0], np.inf, np.inf)
        m = [np.abs(a) <= s / 2 for a, s in zip(axes, sizes)]
        self.box = m[0][:, None, None] & m[1][None, :, None] & m[2][None, None, :]
        if not self.box.any():
            raise ValueError(f"support box {self.size_nm} nm contains no voxel "
                             f"(dz = {engine.dz:.2f} nm)")
        self._set(self.box.copy(), engine)
        print(f"Support box {self.size_nm} nm: {self.box.any(axis=(1, 2)).sum()} of {self.box.shape[0]} "
              f"z-slices, {100 * self.box.mean():.1f}% of the volume")

    def on_epoch(self, epoch, engine):
        if self.sw_threshold is None or epoch < self.sw_start or (epoch - self.sw_start) % self.sw_every:
            return
        amp = np.abs(engine.current_object())
        blur = gaussian_filter(amp, self.sw_sigma) if self.sw_sigma > 0 else amp
        new = blur > self.sw_threshold * blur.max()
        if self.sw_in_box:
            new &= self.box
        frac = new.mean()
        if frac < self.sw_min_frac:
            print(f"  shrinkwrap at epoch {epoch}: {100 * frac:.3f}% < {100 * self.sw_min_frac:g}%, kept old support")
            return
        self._set(new, engine)
        self.n_updates += 1
        print(f"  shrinkwrap at epoch {epoch}: support {100 * frac:.1f}% of the box")

    def _set(self, supp, engine):
        self.support = supp
        self._mask_t = torch.as_tensor(~supp, device=engine.device, dtype=engine.rdtype)

    # ---- penalty
    def penalty(self, ctx):
        o = ctx.obj
        e = o.real**2 + o.imag**2
        return torch.sum(self._mask_t * e) / (torch.sum(e) + 1e-60)

    def save_arrays(self):
        return {"Support": self.support, "Support_box": self.box}

    def describe(self):
        s = super().describe() + f", box {self.size_nm} nm"
        if self.sw_threshold is not None:
            s += (f", shrinkwrap thr {self.sw_threshold:g} sigma {self.sw_sigma:g} vox every {self.sw_every} "
                  f"from {self.sw_start}{' in box' if self.sw_in_box else ''}")
        return s
