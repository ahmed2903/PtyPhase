"""
3D-coupled Fourier-ptychography base class.

Same data interface as `PhaseRetrievalBase` (coherent images + k_in per image,
grouped per streak or flat), plus the k_out vectors and the wavelength. Instead of
one independent 2D object spectrum per streak, there is ONE 3D spectrum

    S(q_z, K_0, K_1)      shape (Nz, n0, n1)

and every (image j, scan frequency K) sample is placed at its own Ewald q_z:

    k_in(K) = k_in0,j + K                       (ptyphase convention, P(k_in0 + K))
    q_z(K)  = k_out,z - sqrt(k^2 - |k_in(K)|^2) - G_z

The lateral axes of S are exactly the K-grid of the coherent images, so only q_z is
interpolated (linear, between two nodes). This couples the streaks: a cap tilts
through neighbouring q_z bins, so streaks share voxels and their relative phases
are no longer free.

Units: rad/nm everywhere (as ptyxplore / ptyphase). Layout: images [x, y]
(axis 0 <-> k[0]); object returned as [z, x, y] in the object frame.
"""
from dataclasses import dataclass
import numpy as np
import h5py

from .phase_abstract import PhaseRetrievalBase
from .plotting import plot_images_side_by_side, two_lists_one_slider
from .utils_pr import time_it, prepare_dims, calc_obj_freq_bandwidth, split_psize
import matplotlib.pyplot as plt
from IPython.display import display, clear_output


@dataclass
class EwaldGeometry:
    """Iteration-invariant sampling geometry of the 3D model (see algorithms_3d)."""
    obj_shape: tuple        # (Nz, n0, n1)
    qz_axis: np.ndarray     # (Nz,) rad/nm, q_z = 0 at index Nz//2
    dqz: float
    pup_idx0: np.ndarray    # (N, n0) int  - pupil row for each image / K_0
    pup_idx1: np.ndarray    # (N, n1) int  - pupil col for each image / K_1
    node0: np.ndarray       # (N, n0, n1) int - flat index of the lower q_z node
    w: np.ndarray           # (N, n0, n1) float - weight of the upper node (lower: 1-w)
    qz: np.ndarray          # (N, n0, n1) float - continuous q_z of every sample
    eps: float = 1e-3       # Tikhonov weight, relative to max of the normal-matrix diagonal
    support: np.ndarray = None   # (Nz, n0, n1) bool real-space support, or None
    support_iters: int = 3

    @property
    def n_lat(self):
        return self.obj_shape[1] * self.obj_shape[2]


class PhaseRetrieval3DBase(PhaseRetrievalBase):
    """
    Args:
        images      : list (one per streak) of (n_i, n0, n1) intensity stacks, or one
                      (N, n0, n1) array. Streak grouping is only used to estimate dqz.
        kins        : same structure, (n_i, 3) k_in vectors (rad/nm) - pupil positions,
                      exactly as for the 2D engine.
        kouts       : same structure, (n_i, 3) k_out vectors (rad/nm).
        wavelength  : nm. If None, k = median |k_out|.
        G           : (3,) reference reciprocal-lattice vector (rad/nm), the same for
                      every pixel. From ptyxplore pass roi.G (compute_kins_global_G) with
                      kins=roi.kins, kouts=roi.kouts. If None, mean(k_out - k_in); with
                      elastic-sphere kins that is off only by a constant in G_z (a phase
                      ramp along z). q_z is measured relative to it.
        pupil_func, ks_pupil, lr_psize : as for PhaseRetrievalBase.
        rec_fourier_obj : optional initial 3D spectrum (Nz, n0, n1).
    """

    def __init__(self, images, kins, kouts, pupil_func, ks_pupil, lr_psize,
                 wavelength=None, G=None, rec_fourier_obj=None,
                 num_jobs=4, backend="threading"):

        super().__init__(images=images, pupil_func=pupil_func, kins=kins,
                         ks_pupil=ks_pupil, lr_psize=lr_psize,
                         num_jobs=num_jobs, backend=backend)
        self.kouts = kouts
        self.wavelength = wavelength
        self.G = None if G is None else np.asarray(G, dtype=float)
        self.rec_fourier_obj = rec_fourier_obj
        self.obj_3D = None

    # ---------------------------------------------------------------- prepare
    @time_it
    def prepare(self, extend=None, dqz=None, qz_oversample=1, nz_pad=2,
                eps=1e-3, support=None, z_extent_nm=None, support_iters=3,
                init='ones', **kwargs):
        """
        extend        : pupil-grid padding, as in the 2D engine.
        dqz           : q_z grid spacing (rad/nm). None -> median spacing of the
                        streak q_z values (needs images/kins/kouts given per streak).
        qz_oversample : divide dqz by this (larger z field of view, 2π/dqz).
        nz_pad        : extra q_z nodes on each side of the sampled range.
        eps           : Tikhonov weight for the object fit (unmeasured voxels keep
                        their previous value).
        support       : (Nz, n0, n1) bool real-space support, or
        z_extent_nm   : build a slab support |z| <= z_extent_nm/2.
        init          : 'ones' (unit spectrum on measured voxels) or 'amplitude'
                        (fit of sqrt(I) with zero phase).
        """
        print("Preparing 3D")
        self.extend = extend

        self._prep_images()
        self.bounds_x, self.bounds_y, self.dks = prepare_dims(
            self.img_shape, self.ks_pupil, self.lr_psize, extend=self.extend)
        self.kx_min_n, self.kx_max_n = self.bounds_x
        self.ky_min_n, self.ky_max_n = self.bounds_y
        self.dkx, self.dky = self.dks
        self.omega_obj_x, self.omega_obj_y = calc_obj_freq_bandwidth(self.lr_psize)

        self._load_pupil()
        self.pupil_shape = self.pupil_func.shape

        self.compute_bounds()
        self._build_geometry(dqz, qz_oversample, nz_pad, eps, support_iters)

        if z_extent_nm is not None:
            support = np.broadcast_to(
                (np.abs(self.z_axis) <= z_extent_nm / 2)[:, None, None], self.geom.obj_shape).copy()
        self.geom.support = support

        self._initiate_recons_images(init)
        self.PSI = self.get_psi()

    def _prep_images(self):
        """Flatten per-streak lists; remember the streak index of every image."""
        if isinstance(self.images, (list, tuple)):
            self.streak_idx = np.concatenate([np.full(len(im), s) for s, im in enumerate(self.images)])
            self.images = np.concatenate([np.asarray(im) for im in self.images])
            self.kins = np.concatenate([np.asarray(k) for k in self.kins])
            self.kouts = np.concatenate([np.asarray(k) for k in self.kouts])
        else:
            self.images = np.asarray(self.images)
            as_flat = lambda v: np.concatenate(v) if isinstance(v, (list, tuple)) else np.asarray(v)
            self.kins = as_flat(self.kins)
            self.kouts = as_flat(self.kouts)
            self.streak_idx = None
        self.num_streaks = 1 if self.streak_idx is None else int(self.streak_idx.max()) + 1
        self.num_images = len(self.images)
        self.img_shape = self.images.shape[1:]

        if self.wavelength is not None:
            self.k = 2 * np.pi / self.wavelength
        else:
            self.k = float(np.median(np.linalg.norm(self.kouts, axis=1)))
        if self.G is None:
            self.G = np.mean(self.kouts - self.kins, axis=0)

    @time_it
    def compute_bounds(self):
        """Pupil patch for every image (same rule as the 2D engine)."""
        self.patch_bounds, self.pupil_slices = [], []
        for kin in self.kins:
            (lx, hx, ly, hy), rest = self._compute_patch_bounds(kin)
            self.patch_bounds.append(((lx, hx, ly, hy), rest))
            self.pupil_slices.append((slice(lx, hx), slice(ly, hy)))

    # ------------------------------------------------------------- geometry
    def _qz_samples(self):
        """Continuous q_z for every (image, K) sample, shape (N, n0, n1)."""
        n0, n1 = self.img_shape
        K0 = (np.arange(n0) - n0 // 2) * self.dkx
        K1 = (np.arange(n1) - n1 // 2) * self.dky
        kin0 = self.kins[:, 0, None, None] + K0[None, :, None]
        kin1 = self.kins[:, 1, None, None] + K1[None, None, :]
        kin_z = np.sqrt(np.clip(self.k**2 - kin0**2 - kin1**2, 0.0, None))
        return self.kouts[:, 2, None, None] - kin_z - self.G[2]

    def _estimate_dqz(self, qz0):
        if self.streak_idx is None or self.num_streaks < 2:
            raise ValueError("dqz=None needs the data given per streak; pass dqz explicitly.")
        qs = np.sort([np.median(qz0[self.streak_idx == s]) for s in range(self.num_streaks)])
        return float(np.median(np.diff(qs)))

    @time_it
    def _build_geometry(self, dqz, qz_oversample, nz_pad, eps, support_iters):
        n0, n1 = self.img_shape
        N = self.num_images

        qz = self._qz_samples()
        n0c, n1c = n0 // 2, n1 // 2
        if dqz is None:
            dqz = self._estimate_dqz(qz[:, n0c, n1c])
        dqz = dqz / qz_oversample

        Nz = 2 * int(np.ceil(np.abs(qz).max() / dqz)) + 2 * nz_pad + 2
        Nz += Nz % 2
        qz_axis = (np.arange(Nz) - Nz // 2) * dqz

        f = qz / dqz + Nz // 2
        j0 = np.floor(f).astype(np.int64)
        w = f - j0
        lat = np.arange(n0 * n1).reshape(n0, n1)
        node0 = j0 * (n0 * n1) + lat[None]

        pup_idx0 = np.array([np.arange(sl[0].start, sl[0].stop) for sl in self.pupil_slices])
        pup_idx1 = np.array([np.arange(sl[1].start, sl[1].stop) for sl in self.pupil_slices])

        self.geom = EwaldGeometry(obj_shape=(Nz, n0, n1), qz_axis=qz_axis, dqz=dqz,
                                  pup_idx0=pup_idx0, pup_idx1=pup_idx1,
                                  node0=node0, w=w, qz=qz, eps=eps,
                                  support_iters=support_iters)

        psize_x, psize_y = split_psize(self.lr_psize)
        self.dz = 2 * np.pi / (Nz * dqz)
        self.z_axis = (np.arange(Nz) - Nz // 2) * self.dz
        self.x_axis = (np.arange(n0) - n0 // 2) * psize_x
        self.y_axis = (np.arange(n1) - n1 // 2) * psize_y

        tilt = qz - qz[:, n0c:n0c + 1, n1c:n1c + 1]
        print(f"3D grid: Nz={Nz}, dqz={dqz:.3e} rad/nm, dz={self.dz:.2f} nm, "
              f"z FOV={Nz*self.dz:.0f} nm | q_z spread within a cap: "
              f"rms {np.std(tilt)/dqz:.2f}, max {np.abs(tilt).max()/dqz:.2f} dqz")

    # --------------------------------------------------------- initialisation
    def _initiate_recons_images(self, init='ones'):
        from .algorithms_3d import fit_object, coverage

        measured = coverage(self.geom, self.pupil_func)
        if self.rec_fourier_obj is not None:
            S = np.asarray(self.rec_fourier_obj, dtype=complex)
            assert S.shape == self.geom.obj_shape, f"rec_fourier_obj must be {self.geom.obj_shape}"
        elif init == 'amplitude':
            amp = self.forward_fft_images(np.sqrt(self.images).astype(complex))
            S = fit_object(self.geom, amp, self.pupil_func, np.zeros(self.geom.obj_shape, complex))
        else:
            S = measured.astype(complex)
        self.rec_fourier_images = S
        self.rec_obj_images = self.inverse_fft_3d(S)

    def get_psi(self):
        from .algorithms_3d import project_model
        return project_model(self.geom, self.pupil_func, self.rec_fourier_images)

    # ------------------------------------------------------------------- I/O
    def save_reconsturction(self, file_path, metadata={}):
        super().save_reconsturction(file_path, metadata)
        with h5py.File(file_path, "a") as h5f:
            if hasattr(self, "pupil_cover_mask"):
                # pupil pixels the data constrains; the rest keeps the initial guess
                h5f["Reconstructed_data"].create_dataset("Pupil_coverage", data=self.pupil_cover_mask(),
                                                         compression="gzip")
            ax = h5f.create_group("Axes")
            ax.create_dataset("qz_rad_nm_inv", data=self.geom.qz_axis)
            ax.create_dataset("z_nm", data=self.z_axis)
            ax.create_dataset("x_nm", data=self.x_axis)
            ax.create_dataset("y_nm", data=self.y_axis)
            ax.attrs["G_rad_nm_inv"] = self.G
            ax.attrs["layout"] = "object [z, x, y]; spectrum [qz, Kx, Ky]"

    # ---------- Fourier Utilities ----------
    @staticmethod
    def forward_fft_images(x):
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))

    @staticmethod
    def inverse_fft_3d(x):
        """Centered 3D iFFT: spectrum [qz, K0, K1] -> object [z, x, y]."""
        return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x)))

    @staticmethod
    def forward_fft_3d(x):
        return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x)))


class LivePlot3D:
    def _initialize_live_plot(self):
        fig = plt.figure(figsize=(9, 11))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
        ax_xy = fig.add_subplot(gs[0, 0]); ax_xz = fig.add_subplot(gs[0, 1])
        ax_ph = fig.add_subplot(gs[1, 0]); ax_pup = fig.add_subplot(gs[1, 1])
        ax_loss = fig.add_subplot(gs[2, :])

        obj = self.inverse_fft_3d(self.rec_fourier_images)
        zc, xc = obj.shape[0] // 2, obj.shape[1] // 2
        self._lp_xy = ax_xy.imshow(np.abs(obj[zc]), origin='lower'); ax_xy.set_title("|obj| z = 0 (x, y)")
        self._lp_xz = ax_xz.imshow(np.abs(obj[:, xc, :]), origin='lower', aspect='auto'); ax_xz.set_title("|obj| x = 0 (z, y)")
        self._lp_ph = ax_ph.imshow(np.angle(obj[zc]), origin='lower', vmin=-np.pi, vmax=np.pi); ax_ph.set_title("phase z = 0")
        self._lp_pup = ax_pup.imshow(np.angle(self.pupil_func), origin='lower', vmin=-np.pi, vmax=np.pi); ax_pup.set_title("Pupil phase")
        self._lp_gap, = ax_loss.plot([], [], label='gap')
        self._lp_mis, = ax_loss.plot([], [], label='data misfit')
        ax_loss.set_xlabel("Iteration"); ax_loss.grid(True, alpha=0.3); ax_loss.legend()
        self.fig, self._lp_ax_loss = fig, ax_loss
        plt.ion(); plt.show()

    def _update_live_plot(self):
        obj = self.inverse_fft_3d(self.rec_fourier_images)
        zc, xc = obj.shape[0] // 2, obj.shape[1] // 2
        for im, data in ((self._lp_xy, np.abs(obj[zc])), (self._lp_xz, np.abs(obj[:, xc, :]))):
            im.set_data(data); im.set_clim(0, data.max())
        self._lp_ph.set_data(np.angle(obj[zc]))
        self._lp_pup.set_data(np.angle(self.pupil_func))
        its = range(len(self.gaps))
        self._lp_gap.set_data(its, self.gaps)
        self._lp_mis.set_data(its, self.data_misfits)
        ax = self._lp_ax_loss
        ax.relim(); ax.autoscale_view()
        if len(self.gaps) > 1:
            ax.set_yscale('log')
        ax.set_title(f"Iteration {self.iters_passed}, gap {self.gaps[-1]:.3e}")
        clear_output(wait=True)
        display(self.fig)
        self.fig.canvas.flush_events()


class Plot3D:
    def plot_rec_obj(self, vmin1=None, vmax1=None, vmin2=-np.pi, vmax2=np.pi,
                     cmap1="viridis", cmap2="viridis", axis=0):
        """Slider through the 3D object along `axis` (0 = z, 1 = x, 2 = y)."""
        obj = np.moveaxis(self.obj_3D, axis, 0)
        two_lists_one_slider(np.abs(obj), np.angle(obj), vmin1=vmin1, vmax1=vmax1,
                             vmin2=vmin2, vmax2=vmax2, cmap1=cmap1, cmap2=cmap2)

    def plot_rec_fourier(self, vmin1=None, vmax1=None, vmin2=-np.pi, vmax2=np.pi,
                         cmap1="viridis", cmap2="viridis", axis=0):
        S = np.moveaxis(self.rec_fourier_images, axis, 0)
        two_lists_one_slider(np.abs(S), np.angle(S), vmin1=vmin1, vmax1=vmax1,
                             vmin2=vmin2, vmax2=vmax2, cmap1=cmap1, cmap2=cmap2)

    def plot_pupil_func(self, vmin1=None, vmax1=None, vmin2=-np.pi, vmax2=np.pi,
                        cmap1="viridis", cmap2="viridis"):
        plot_images_side_by_side(np.abs(self.pupil_func), np.angle(self.pupil_func),
                                 vmin1=vmin1, vmax1=vmax1, vmin2=vmin2, vmax2=vmax2,
                                 title1="Pupil Amplitude", title2="Pupil Phase",
                                 cmap1=cmap1, cmap2=cmap2, figsize=(10, 5), show=True)

    def plot_losses(self):
        plt.figure()
        plt.semilogy(self.gaps, label='gap')
        plt.semilogy(self.data_misfits, label='data misfit')
        plt.xlabel("Iteration"); plt.legend()
