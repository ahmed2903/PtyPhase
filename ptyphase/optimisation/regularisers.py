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

from .forward import ifft3c, abs2


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


# ---------------------------------------------------------------------------------- pupil smoothness
def _stencil_masks(mask, order):
    """Per-axis bool masks of the finite-difference stencils lying entirely inside `mask`."""
    m = np.asarray(mask, dtype=bool)
    if order == 2:
        return m[2:, :] & m[1:-1, :] & m[:-2, :], m[:, 2:] & m[:, 1:-1] & m[:, :-2]
    if order == 1:
        return m[1:, :] & m[:-1, :], m[:, 1:] & m[:, :-1]
    raise ValueError("order must be 1 or 2")


def _differences(P, order):
    """Finite differences of P along axis 0 and axis 1 (numpy or torch)."""
    if order == 2:
        return P[2:, :] - 2 * P[1:-1, :] + P[:-2, :], P[:, 2:] - 2 * P[:, 1:-1] + P[:, :-2]
    return P[1:, :] - P[:-1, :], P[:, 1:] - P[:, :-1]


def _bbox(mask):
    i0, i1 = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    return slice(i0[0], i0[-1] + 1), slice(i1[0], i1[-1] + 1)


def pupil_roughness(P, mask, order=2):
    """The PupilSmooth value of a numpy pupil: mean |finite difference|^2 over the stencils
    inside `mask`, divided by the mean |P|^2 on `mask`. NaN if there is no stencil."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return float("nan")
    sl = _bbox(m)
    P, m = np.asarray(P)[sl], m[sl]
    v0, v1 = _stencil_masks(m, order)
    n = int(v0.sum() + v1.sum())
    if n == 0:
        return float("nan")
    d0, d1 = _differences(P, order)
    num = (np.sum(np.abs(d0[v0])**2) + np.sum(np.abs(d1[v1])**2)) / n
    return float(num / max(np.mean(np.abs(P[m])**2), 1e-300))


class PupilSmooth(Regulariser):
    """Roughness of the pupil on the pixels the data constrain:

        pen = mean_stencils |D P|^2  /  mean_mask |P|^2

    D P are finite differences along each pupil axis (no diagonal terms):
      order 2 (default): P[i+1] - 2 P[i] + P[i-1]   (curvature)
      order 1          : P[i+1] - P[i]              (gradient)
    The mean runs over the stencils that lie entirely in the mask. By default the mask is the
    current data coverage intersected with the aperture (engine.current_pupil_coverage()). So
    uncovered pixels never enter, nothing is extrapolated, and the aperture edge (a real step in
    |P|) is not smoothed. With learnable streak shifts the coverage moves, so the mask is rebuilt
    every `mask_every` epochs.

    What it costs (order 2, |P| ~ 1):
      checkerboard of amplitude eps        16 eps^2 per stencil
      phase slope g rad/px (tilt)          4 (1 - cos g)^2 ~ g^4 per stencil
      single-pixel dip of depth d          12 d^2 in total (4d^2 + 2 d^2 per axis)
    The dips matter: a free pupil can lower |P| at the pixels where each image takes its K=0
    sample, while S(q_z, K=0) grows and every image's mean level stays the same. The weight on
    those pixels is small, so this runs away under a drive (bilinear blending, or the ratio
    gradient of Support3D) and ends in the pixel-pupil blow-up. order 1 discriminates much less
    (checkerboard 8 eps^2 against g^2 for a slope).

    Scale-invariant (P and S trade a common scale), so weights carry over between runs.
    detach_norm (default True): the denominator is detached, so the gradient only lowers the
    roughness. With the plain ratio, the gradient also pushes mean |P|^2 up, and that can be
    gamed the same way Support3D's ratio is (raise the denominator instead of lowering the
    numerator). The value is the same either way.

    mask       : bool pupil-shaped array to use instead of the coverage (offline evaluation, tests).
    mask_every : epochs between mask rebuilds when the engine has streak shifts.
    """
    name = "pupil_smooth"

    def __init__(self, weight=1.0, start=0, ramp=0, scheduler=None, order=2, mask=None,
                 detach_norm=True, mask_every=100):
        super().__init__(weight, start, ramp, scheduler)
        if order not in (1, 2):
            raise ValueError("PupilSmooth order must be 1 or 2")
        self.order = int(order)
        self.detach_norm = bool(detach_norm)
        self.mask_every = max(int(mask_every), 1)
        self._user_mask = None if mask is None else np.asarray(mask, dtype=bool)
        self._dynamic = False
        self.mask = None
        self._device, self._rdtype = "cpu", torch.float64
        if self._user_mask is not None:
            self._build(self._user_mask)

    # ---- engine hooks
    def setup(self, engine):
        self._device, self._rdtype = engine.device, engine.rdtype
        self._dynamic = self._user_mask is None and getattr(engine, "shifts", None) is not None
        self._build(self._user_mask if self._user_mask is not None else engine.current_pupil_coverage())
        print(f"Pupil smoothness: order {self.order}, {self._n_mask} pupil px "
              f"({'given mask' if self._user_mask is not None else 'covered aperture'}), "
              f"{self._n_terms} stencils{', mask rebuilt every %d epochs (streak shifts)' % self.mask_every if self._dynamic else ''}")

    def on_epoch(self, epoch, engine):
        if self._dynamic and epoch > 0 and epoch % self.mask_every == 0:
            self._build(engine.current_pupil_coverage())

    def _build(self, mask):
        m = np.asarray(mask, dtype=bool)
        if not m.any():
            raise ValueError("PupilSmooth: empty mask")
        self._sl = _bbox(m)
        mc = m[self._sl]
        v0, v1 = _stencil_masks(mc, self.order)
        self._n_terms, self._n_mask = int(v0.sum() + v1.sum()), int(mc.sum())
        if self._n_terms == 0:
            raise ValueError("PupilSmooth: no finite-difference stencil lies inside the mask")
        t = lambda a: torch.as_tensor(a, dtype=self._rdtype, device=self._device)
        self._m, self._v0, self._v1 = t(mc), t(v0), t(v1)
        self.mask = m

    # ---- penalty
    def penalty(self, ctx):
        P = ctx.P[self._sl]
        d0, d1 = _differences(P, self.order)
        num = (torch.sum(self._v0 * abs2(d0)) + torch.sum(self._v1 * abs2(d1))) / self._n_terms
        den = torch.sum(self._m * abs2(P)) / self._n_mask
        if self.detach_norm:
            den = den.detach()
        return num / (den + 1e-60)

    def describe(self):
        return (super().describe() + f", order {self.order} ({'2nd' if self.order == 2 else '1st'} differences), "
                f"{self._n_mask if self.mask is not None else '?'} px mask, "
                f"{'detached' if self.detach_norm else 'ratio'} normalisation")
