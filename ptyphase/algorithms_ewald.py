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

from .algorithms import DM, RAAR, RAAR_standard, ER, fft_images, ifft_images

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
        O = super().update_object_fts(PSI, pupil, pupil_slices, object_slices,
                                      objectFTs=objectFTs, eps=eps)
        if self.geom is None:
            return O
        return project_support_3d(self.geom, O)


class DMEwald(EwaldSupportMixin, DM):
    pass


class RAAREwald(EwaldSupportMixin, RAAR):
    pass


class RAAR_standardEwald(EwaldSupportMixin, RAAR_standard):
    pass


class EREwald(EwaldSupportMixin, ER):
    pass
