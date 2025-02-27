
import numpy as np 
from numpy.fft import fftshift, ifftshift, fft2, ifft2


from ..utils_pr import *

class EPRy:
    
    def __init__(self, images, pupil_func, kout_vec, lr_psize, 
                 num_iter=50, alpha=0.1, beta=0.1, hr_obj_image=None, hr_fourier_image=None):
        self.images = images
        self.pupil_func = pupil_func
        self.kout_vec = kout_vec
        self.lr_psize = lr_psize
        
        self.num_iter = num_iter
        self.alpha = alpha
        self.beta = beta

        self.hr_obj_image = hr_obj_image
        self.hr_fourier_image = hr_fourier_image 
        


    def prepare(self):
        
        self.kout_vec = np.array(self.kout_vec)
        self.images = np.array(self.images)
        
        bounds_x, bounds_y, dks = prepare_dims(self.images, self.kout_vec, self.lr_psize)
        self.kx_min_n, self.kx_max_n = bounds_x
        self.ky_min_n, self.ky_max_n = bounds_y
        self.dkx, self.dky = dks
        
        omegas = calc_obj_freq_bandwidth(self.lr_psize)
        self.omega_obj_x, self.omega_obj_y = omegas
        
        
        if self.hr_obj_image is None or self.hr_fourier_image is None:
            self.hr_obj_image, self.hr_fourier_image = init_hr_image(bounds_x, bounds_y, dks)

        
    def iterate(self):
        nx_lr, ny_lr = self.images[0].shape
        nx_hr, ny_hr = self.hr_obj_image.shape

        for it in range(self.num_iter):
            print(f"Iteration {it+1}/{self.num_iter}")

            for i, (image, kx_iter, ky_iter) in enumerate(zip(self.images, self.kout_vec[:, 0], self.kout_vec[:, 1])):
                self._update_spectrum(image, kx_iter, ky_iter, nx_lr, ny_lr)

        self.hr_obj_image = fftshift(ifft2(ifftshift(self.hr_fourier_image)))
        return self.hr_obj_image, self.hr_fourier_image, self.pupil_func

    def compute_weight_fac(func):
        """Compute weighting factor for phase retrieval update."""
        
        mod = np.abs(func) ** 2
        return np.conjugate(func) / (mod.max() + 1e-23)

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

        weight_fac_pupil = self.beta * self.compute_weight_fac(pupil_func_patch)

        delta_lowres_ft = image_FT_update - image_FT
        self.hr_fourier_image[kx_lidx:kx_hidx, ky_lidx:ky_hidx] += delta_lowres_ft *  weight_fac_pupil
