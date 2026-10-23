"""
Per-streak Fourier-ptychography base with curved Ewald slices and a 3D support.

Same unknowns as `PhaseRetrievalBase`: one 2D spectrum per streak (rec_fourier_images,
shape (Ns, n0, n1)) and one shared pupil. The difference is how the slices make up the
3D spectrum. Streak s at image frequency K sits on the curved cap (no tilt)

    q_z(s, K) = q_s + c(K),   q_s = (s - Ns//2) dqz,   c(K) = k - sqrt(k^2 - |K|^2)

so the 3D object is built exactly (no interpolation) as

    obj[z, x, y] = IFFT_K[ exp(+i c(K) z) * IFFT_s( F[s, K] ) ]

(see algorithms_ewald.slices_to_object). Streaks must be ordered by q_z with q_z = 0 at
index Ns//2, uniformly spaced by dqz - the same convention as `ifft_images_z` in runner.py.
With curvature=False the mapping is exactly the flat stacking used there.

Units: rad/nm (k = 2*pi/lambda). Layout: object [z, x, y], spectrum [qz, K0, K1].
"""
from dataclasses import dataclass
import numpy as np
import h5py

from .phase_abstract import PhaseRetrievalBase
from .utils_pr import time_it, split_psize


@dataclass
class EwaldSliceGeometry:
    """Iteration-invariant geometry of the curved-slice stack (see algorithms_ewald)."""
    obj_shape: tuple            # (Nz, n0, n1), Nz = number of streaks
    qz_axis: np.ndarray         # (Nz,) rad/nm, q_z = 0 at index Nz//2
    z_axis: np.ndarray          # (Nz,) nm
    dqz: float
    k: float                    # rad/nm
    curv: np.ndarray            # (n0, n1) c(K) in rad/nm
    ramp: np.ndarray = None     # (Nz, n0, n1) exp(+i c(K) z), None -> flat slices
    support: np.ndarray = None  # (Nz, n0, n1) bool real-space support [z, x, y], or None


class PhaseRetrievalEwaldBase(PhaseRetrievalBase):
    """
    Args (on top of PhaseRetrievalBase):
        wavelength : nm.
        dqz        : q_z spacing between neighbouring streaks (rad/nm), e.g. roi.delta_qz.
        support    : optional (Ns, n0, n1) bool real-space support [z, x, y]; can also be
                     given (or replaced) in prepare().
    """

    def __init__(self, wavelength=None, dqz=None, support=None, **kwargs):
        super().__init__(**kwargs)
        if wavelength is None or dqz is None:
            raise ValueError("wavelength (nm) and dqz (rad/nm) are required")
        self.wavelength = wavelength
        self.dqz = float(dqz)
        self.support = support
        self.obj_3D = None

    @time_it
    def prepare(self, extend=None, curvature=True, support=None, **kwargs):
        """
        extend    : pupil-grid padding, as in the 2D engine.
        curvature : place the slices on the curved Ewald caps (True) or on flat planes.
        support   : (Ns, n0, n1) bool real-space support [z, x, y]; None keeps the one
                    given at construction (None -> no support).
        """
        super().prepare(extend=extend, **kwargs)
        if support is not None:
            self.support = support
        self._build_ewald_geometry(curvature)

    def _build_ewald_geometry(self, curvature):
        Nz = self.num_streaks
        n0, n1 = self.img_shape
        k = 2 * np.pi / self.wavelength

        qz_axis = (np.arange(Nz) - Nz // 2) * self.dqz
        self.dz = 2 * np.pi / (Nz * self.dqz)
        z_axis = (np.arange(Nz) - Nz // 2) * self.dz

        K0 = (np.arange(n0) - n0 // 2) * self.dkx
        K1 = (np.arange(n1) - n1 // 2) * self.dky
        curv = k - np.sqrt(k**2 - K0[:, None]**2 - K1[None, :]**2)

        ramp = np.exp(1j * curv[None] * z_axis[:, None, None]) if curvature else None

        support = None
        if self.support is not None:
            support = np.asarray(self.support, dtype=bool)
            assert support.shape == (Nz, n0, n1), \
                f"support must be (num_streaks, n0, n1) = {(Nz, n0, n1)}, got {support.shape}"

        self.geom = EwaldSliceGeometry(obj_shape=(Nz, n0, n1), qz_axis=qz_axis, z_axis=z_axis,
                                       dqz=self.dqz, k=k, curv=curv, ramp=ramp, support=support)
        self.curvature = curvature

        psize_x, psize_y = split_psize(self.lr_psize)
        self.x_axis = (np.arange(n0) - n0 // 2) * psize_x
        self.y_axis = (np.arange(n1) - n1 // 2) * psize_y

        print(f"Ewald slices: Nz={Nz}, dqz={self.dqz:.3e} rad/nm, dz={self.dz:.2f} nm, "
              f"z FOV={Nz*self.dz:.0f} nm | curvature {'on' if curvature else 'off'}, "
              f"max c(K) = {curv.max():.3e} rad/nm = {curv.max()/self.dqz:.3f} dqz")

    def save_reconsturction(self, file_path, metadata={}):
        from .algorithms_ewald import slices_to_object

        super().save_reconsturction(file_path, metadata)
        obj = slices_to_object(self.geom, self.rec_fourier_images)
        with h5py.File(file_path, "a") as h5f:
            rec = h5f["Reconstructed_data"]
            rec.create_dataset("Object_3D_amplitude", data=np.abs(obj), compression="gzip")
            rec.create_dataset("Object_3D_phase", data=np.angle(obj), compression="gzip")
            if self.geom.support is not None:
                rec.create_dataset("Support", data=self.geom.support, compression="gzip")
            ax = h5f.create_group("Axes")
            ax.create_dataset("qz_rad_nm_inv", data=self.geom.qz_axis)
            ax.create_dataset("z_nm", data=self.geom.z_axis)
            ax.create_dataset("x_nm", data=self.x_axis)
            ax.create_dataset("y_nm", data=self.y_axis)
            ax.attrs["curvature"] = self.curvature
            ax.attrs["dqz_rad_nm_inv"] = self.dqz
            ax.attrs["wavelength_nm"] = self.wavelength
            ax.attrs["layout"] = "object [z, x, y]; spectrum [qz, Kx, Ky]"
