
import numpy as np 
from numpy.fft import fftshift, ifftshift, fft2, ifft2


from ..utils_pr import *
from ..data_fs import upsample_images

class EPRyReconstructor:
    def __init__(self, images, pupil_func, kout_vec, bounds_x, bounds_y, dks, omegas, 
                 num_iter=50, alpha=0.1, beta=0.1, hr_obj_image=None, hr_fourier_image=None):
        self.images = images
        self.pupil_func = pupil_func
        self.kout_vec = kout_vec
        self.kx_min_n, self.kx_max_n = bounds_x
        self.ky_min_n, self.ky_max_n = bounds_y
        self.dkx, self.dky = dks
        self.omega_obj_x, self.omega_obj_y = omegas
        self.num_iter = num_iter
        self.alpha = alpha
        self.beta = beta

        if hr_obj_image is None or hr_fourier_image is None:
            self.hr_obj_image, self.hr_fourier_image = init_hr_image(bounds_x, bounds_y, dks)
        else:
            self.hr_obj_image, self.hr_fourier_image = hr_obj_image, hr_fourier_image

    def iterate(self):
        nx_lr, ny_lr = self.images[0].shape
        nx_hr, ny_hr = self.hr_obj_image.shape

        for it in range(self.num_iter):
            print(f"Iteration {it+1}/{self.num_iter}")

            for i, (image, kx_iter, ky_iter) in enumerate(zip(self.images, self.kout_vec[:, 0], self.kout_vec[:, 1])):
                self._update_spectrum(image, kx_iter, ky_iter, nx_lr, ny_lr)

        self.hr_obj_image = fftshift(ifft2(ifftshift(self.hr_fourier_image)))
        return self.hr_obj_image, self.hr_fourier_image, self.pupil_func

    def _update_spectrum(self, image, kx_iter, ky_iter, nx_lr, ny_lr):
        """Handles the Fourier domain update."""
        kx_cidx = round((kx_iter - self.kx_min_n) / self.dkx)
        kx_lidx = round(max(kx_cidx - self.omega_obj_x / (2 * self.dkx), 0))
        kx_hidx = round(kx_cidx + self.omega_obj_x / (2 * self.dkx)) + (1 if nx_lr % 2 != 0 else 0)

        ky_cidx = round((ky_iter - self.ky_min_n) / self.dky)
        ky_lidx = round(max(ky_cidx - self.omega_obj_y / (2 * self.dky), 0))
        ky_hidx = round(ky_cidx + self.omega_obj_y / (2 * self.dky)) + (1 if ny_lr % 2 != 0 else 0)

        pupil_func_patch = self.pupil_func[kx_lidx:kx_hidx, ky_lidx:ky_hidx]
        image_FT = self.hr_fourier_image[kx_lidx:kx_hidx, ky_lidx:ky_hidx] * pupil_func_patch

        image_lr = fftshift(ifft2(ifftshift(image_FT)))
        image_lr_update = np.sqrt(image) * np.exp(1j * np.angle(image_lr))
        image_FT_update = fftshift(fft2(ifftshift(image_lr_update)))

        mod_pupil = np.abs(pupil_func_patch) ** 2
        weight_fac_pupil = np.conjugate(pupil_func_patch) / (mod_pupil.max() + 1e-23)

        delta_lowres_ft = image_FT_update - image_FT
        self.hr_fourier_image[kx_lidx:kx_hidx, ky_lidx:ky_hidx] += delta_lowres_ft * 0.9 * weight_fac_pupil
