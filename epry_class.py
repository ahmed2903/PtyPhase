
import numpy as np 
from numpy.fft import fftshift, ifftshift, fft2, ifft2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import display, clear_output
from tqdm.notebook import tqdm
from ..utils_pr import *
from ..plotting_fs import plot_images_side_by_side, update_live_plot, initialize_live_plot
from ..data_fs import downsample_array

class EPRy:
    
    def __init__(self, images, pupil_func: str, kout_vec, lr_psize, 
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
        
        self.bounds_x, self.bounds_y, self.dks = prepare_dims(self.images, self.kout_vec, self.lr_psize)
        self.kx_min_n, self.kx_max_n = self.bounds_x
        self.ky_min_n, self.ky_max_n = self.bounds_y
        self.dkx, self.dky = self.dks
        
        omegas = calc_obj_freq_bandwidth(self.lr_psize)
        self.omega_obj_x, self.omega_obj_y = omegas

        phase = np.load('phase_aberration_run215.npy')

        dims = round((self.kx_max_n - self.kx_min_n)/self.dkx), round((self.ky_max_n - self.ky_min_n)/self.dky)
        phase = downsample_array(phase, dims)
        self.pupil_func = np.exp(1j*phase)
        
        self._initiate_recons_images()
        
    def _initiate_recons_images(self):
        
        if self.hr_obj_image is None or self.hr_fourier_image is None:
            self.hr_obj_image, self.hr_fourier_image = init_hr_image(self.bounds_x, self.bounds_y, self.dks)

        self.nx_lr, self.ny_lr = self.images[0].shape
        self.nx_hr, self.ny_hr = self.hr_obj_image.shape
        
    def iterate(self, live_plot=False):

        if live_plot:
            fig, ax, img_amp, img_phase = initialize_live_plot(self.hr_obj_image)
        
        for it in range(self.num_iter):
            print(f"Iteration {it+1}/{self.num_iter}")
            for i, (image, kx_iter, ky_iter) in enumerate(tqdm(zip(self.images, self.kout_vec[:, 0], self.kout_vec[:, 1]), 
                                                               desc="Processing", total=len(self.images), unit="images")):
                
                self._update_spectrum(image, kx_iter, ky_iter)

            if live_plot:
                # Update the HR object image after all spectrum updates in this iteration
                self.hr_obj_image = fftshift(ifft2(ifftshift(self.hr_fourier_image)))
                update_live_plot(img_amp, img_phase, self.hr_obj_image, fig)
            
        
        self.hr_obj_image = fftshift(ifft2(ifftshift(self.hr_fourier_image)))
        
    def compute_weight_fac(self, func):
        """Compute weighting factor for phase retrieval update."""
        
        mod = np.abs(func) ** 2
        return np.conjugate(func) / (mod.max() + 1e-23)

    def _update_spectrum(self, image, kx_iter, ky_iter):
        """Handles the Fourier domain update."""
        kx_cidx = round((kx_iter - self.kx_min_n) / self.dkx)
        kx_lidx = round(max(kx_cidx - self.omega_obj_x / (2 * self.dkx), 0))
        kx_hidx = round(kx_cidx + self.omega_obj_x / (2 * self.dkx)) + (1 if self.nx_lr % 2 != 0 else 0)

        ky_cidx = round((ky_iter - self.ky_min_n) / self.dky)
        ky_lidx = round(max(ky_cidx - self.omega_obj_y / (2 * self.dky), 0))
        ky_hidx = round(ky_cidx + self.omega_obj_y / (2 * self.dky)) + (1 if self.ny_lr % 2 != 0 else 0)
        
        pupil_func_patch = self.pupil_func[kx_lidx:kx_hidx, ky_lidx:ky_hidx]
        image_FT = self.hr_fourier_image[kx_lidx:kx_hidx, ky_lidx:ky_hidx] * pupil_func_patch
        image_FT *= self.nx_lr/self.nx_hr
        
        image_lr = fftshift(ifft2(ifftshift(image_FT)))
        image_lr_update = np.sqrt(image) * np.exp(1j * np.angle(image_lr))
        image_FT_update = fftshift(fft2(ifftshift(image_lr_update)))
        image_lr_update *= self.nx_hr/self.nx_lr
        weight_fac_pupil = self.alpha * self.compute_weight_fac(pupil_func_patch)

        delta_lowres_ft = image_FT_update - image_FT
        self.hr_fourier_image[kx_lidx:kx_hidx, ky_lidx:ky_hidx] += delta_lowres_ft *  weight_fac_pupil

    def plot_rec_obj(self, 
                     vmin1= None, vmax1=None, 
                     vmin2= -np.pi, vmax2=np.pi, 
                     title1 = "Object Amplitude", title2 = "Object Phase", cmap1 = "viridis", cmap2 = "viridis"):
        
        image1 = np.abs(self.hr_obj_image)
        image2 = np.angle(self.hr_obj_image)
    
        plot_images_side_by_side(image1, image2, 
                                 vmin1= vmin1, vmax1=vmax1, 
                                 vmin2= vmin2, vmax2=vmax2, 
                                 title1=title1, title2=title2, cmap1=cmap1, cmap2=cmap2, figsize=(10, 5), show = True)
    
    def plot_rec_fourier(self, 
                         vmin1= None, vmax1=None, 
                     vmin2= -np.pi, vmax2=np.pi, 
                         title1 = "Fourier Amplitude", title2 = "Fourier Phase", cmap1 = "viridis", cmap2 = "viridis"):
        
        image1 = np.abs(self.hr_fourier_image)
        image2 = np.angle(self.hr_fourier_image)
    
        plot_images_side_by_side(image1, image2, 
                                 vmin1= vmin1, vmax1=vmax1, 
                                 vmin2= vmin2, vmax2=vmax2, 
                                 title1=title1, title2=title2, cmap1=cmap1, cmap2=cmap2, figsize=(10, 5), show = True)
    
    
    def plot_pupil_func(self, 
                        vmin1= None, vmax1=None, 
                        vmin2= -np.pi, vmax2=np.pi, 
                        title1 = "Pupil Amplitude", title2 = "Pupil Phase", cmap1 = "viridis", cmap2 = "viridis"):
        
        image1 = np.abs(self.pupil_func)
        image2 = np.angle(self.pupil_func)
    
        plot_images_side_by_side(image1, image2, 
                                 vmin1= vmin1, vmax1=vmax1, 
                                 vmin2= vmin2, vmax2=vmax2, 
                                 title1=title1, title2=title2, cmap1=cmap1, cmap2=cmap2, figsize=(10, 5), show = True)


class EPRy_lr(EPRy):
    
    
    def _initiate_recons_images(self):
        
        if self.hr_obj_image is None or self.hr_fourier_image is None: 
            self.hr_obj_image = np.ones_like(self.images[0]).astype(complex)
            self.hr_fourier_image = np.ones_like(self.images[0]).astype(complex)
        
        self.nx_lr, self.ny_lr = self.images[0].shape
        self.nx_hr, self.ny_hr = self.hr_obj_image.shape

    def _update_spectrum(self, image, kx_iter, ky_iter):
        """Handles the Fourier domain update."""
        kx_cidx = round((kx_iter - self.kx_min_n) / self.dkx)
        kx_lidx = round(max(kx_cidx - self.omega_obj_x / (2 * self.dkx), 0))
        kx_hidx = round(kx_cidx + self.omega_obj_x / (2 * self.dkx)) + (1 if self.nx_lr % 2 != 0 else 0)
        
        ky_cidx = round((ky_iter - self.ky_min_n) / self.dky)
        ky_lidx = round(max(ky_cidx - self.omega_obj_y / (2 * self.dky), 0))
        ky_hidx = round(ky_cidx + self.omega_obj_y / (2 * self.dky)) + (1 if self.ny_lr % 2 != 0 else 0)
        
        pupil_func_patch = self.pupil_func[kx_lidx:kx_hidx, ky_lidx:ky_hidx]
        image_FT = self.hr_fourier_image * pupil_func_patch
        image_FT *= self.nx_lr/self.nx_hr 
        
        image_lr = fftshift(ifft2(ifftshift(image_FT)))
        image_lr_update = np.sqrt(image) * np.exp(1j * np.angle(image_lr))
        image_FT_update = fftshift(fft2(ifftshift(image_lr_update)))
        image_lr_update *= self.nx_hr/self.nx_lr

        weight_fac_pupil = self.alpha * self.compute_weight_fac(pupil_func_patch)
        
        # Update fourier spectrum
        delta_lowres_ft = image_FT_update - image_FT
        self.hr_fourier_image += delta_lowres_ft *  weight_fac_pupil

        if np.any(np.isnan(self.hr_fourier_image)):
            raise ValueError("There is a Nan value, check the configurations ")
            
        # Update Pupil Function 
        weight_factor_obj = self.beta * self.compute_weight_fac(self.hr_fourier_image)
        self.pupil_func[kx_lidx:kx_hidx, ky_lidx:ky_hidx] += weight_factor_obj * delta_lowres_ft



