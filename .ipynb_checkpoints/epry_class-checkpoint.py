
import numpy as np 
from numpy.fft import fftshift, ifftshift, fft2, ifft2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import display, clear_output
from tqdm.notebook import tqdm


from .utils_pr import *
from .plotting import plot_images_side_by_side, update_live_plot, initialize_live_plot

#from ..data_fs import * #downsample_array, upsample_images, pad_to_double

class EPRy:
    
    def __init__(self, images, pupil_func: str, kout_vec, ks_pupil, lr_psize, alpha=0.1, beta=0.1, hr_obj_image=None, hr_fourier_image=None):
        
        self.images = images
        self.pupil_func = pupil_func
        self.kout_vec = kout_vec
        self.ks_pupil = ks_pupil
        self.lr_psize = lr_psize
        
        self.alpha = alpha
        self.beta = beta
        self.iters_passed = 0
        self.hr_obj_image = hr_obj_image
        self.hr_fourier_image = hr_fourier_image  
        

    ############################# Prepare ################################

    def prepare(self, **kwargs):
        print("Preparing")
        if 'zoom_factor' in kwargs:
            self.zoom_factor = kwargs['zoom_factor']

        if 'extend' in kwargs:
            extend = kwargs['extend']
        else:
            extend = None
            
        self._prep_images()
        
        self.kout_vec = np.array(self.kout_vec)
        self.bounds_x, self.bounds_y, self.dks = prepare_dims(self.images, self.ks_pupil, self.lr_psize, extend = extend)
        self.kx_min_n, self.kx_max_n = self.bounds_x
        self.ky_min_n, self.ky_max_n = self.bounds_y
        self.dkx, self.dky = self.dks
        
        omegas = calc_obj_freq_bandwidth(self.lr_psize)
        self.omega_obj_x, self.omega_obj_y = omegas

        self._load_pupil()
        self._initiate_recons_images()

    def _prep_images(self):

        self.images = np.abs(np.array(self.images))
        
    def _load_pupil(self):
        
        dims = round((self.kx_max_n - self.kx_min_n)/self.dkx), round((self.ky_max_n - self.ky_min_n)/self.dky)
        dims = make_dims_even(dims)
        full_array = np.zeros(dims)
        
        if isinstance(self.pupil_func, str):
            phase = np.load(self.pupil_func)
        elif isinstance(self.pupil_func, np.ndarray):
            phase = self.pupil_func
        else:
            phase = np.zeros(dims)
        
        # Get the scaling factors for each dimension
        scale_x = dims[0] / phase.shape[0] / 2
        scale_y = dims[1] / phase.shape[1] / 2 

        # Scale the pupil phase array to match the required pupil dimensions
        scaled_pupil_phase = zoom(phase, (scale_x, scale_y))

        # Calculate center indices
        N, M = dims[0]//2, dims[1]//2
        
        start_x, start_y = (dims[0] - N) // 2, (dims[1] - M) // 2
        end_x, end_y = start_x + N, start_y + M
    
        # Set central region to ones
        full_array[start_x:end_x, start_y:end_y] = scaled_pupil_phase
        
        self.pupil_func = np.exp(1j*full_array)
    
    def _initiate_recons_images(self):

        
        if self.hr_obj_image is None or self.hr_fourier_image is None:
            self.hr_obj_image, self.hr_fourier_image = init_hr_image(self.bounds_x, self.bounds_y, self.dks)
        
        self.nx_lr, self.ny_lr = self.images[0].shape
        self.nx_hr, self.ny_hr = self.hr_obj_image.shape

    ################################## Main Loop ##################################
    def iterate(self, iterations:int, live_plot=False):

        if live_plot:
            fig, ax, img_amp, img_phase = self._initialize_live_plot()
        
        for it in range(iterations):
            
            for i, (image, kx_iter, ky_iter) in enumerate(tqdm(zip(self.images, self.kout_vec[:, 0], self.kout_vec[:, 1]), 
                                                               desc="Processing", total=len(self.images), unit="images")):

                
                self._update_spectrum(image, kx_iter, ky_iter)
                
                if np.any(np.isnan(self.hr_fourier_image)):
                    raise ValueError(f"There is a Nan in the Fourier image for {it}-th iteration,{i}-th image")
                if np.any(np.isnan(self.hr_obj_image)):
                    raise ValueError(f"There is a Nan in the Object image for {it}-th iteration,{i}-th image")
                
            if live_plot:
                # Update the HR object image after all spectrum updates in this iteration
                self.hr_obj_image = ifft2(ifftshift(self.hr_fourier_image))
                self._update_live_plot(img_amp, img_phase, fig, it)
    
        self.iters_passed += 1
        self.hr_obj_image = ifft2(ifftshift(self.hr_fourier_image))
        
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
        image_FT *= (self.nx_lr/self.nx_hr)**2
        
        image_lr = ifft2(ifftshift(image_FT))
        image_lr_update = np.sqrt(image) * np.exp(1j * np.angle(image_lr))
        inv_pupil_func = 1/ (pupil_func_patch +1e-8)
        image_FT_update = fftshift(fft2(image_lr_update)) * inv_pupil_func
        image_lr_update *= (self.nx_hr/self.nx_lr)**2
        
        # Compute update weight factor
        weight_fac_pupil = self.alpha * self.compute_weight_fac(pupil_func_patch)
        weight_factor_obj = self.beta * self.compute_weight_fac(self.hr_fourier_image[kx_lidx:kx_hidx, ky_lidx:ky_hidx])

        if np.any(np.isnan(np.sqrt(image))):
            raise ValueError(f"There is a Nan in the image")
        if np.any(np.isnan(np.angle(image_lr))):
            raise ValueError(f"There is a Nan in the image_lr")
        if np.any(np.isnan(image_lr_update)):
            raise ValueError(f"There is a Nan in the image_lr_update")

        # Difference of exit waves 
        delta_lowres_ft = image_FT_update - image_FT

        # Update Fourier Spectrum
        self.hr_fourier_image[kx_lidx:kx_hidx, ky_lidx:ky_hidx] += delta_lowres_ft *  weight_fac_pupil
        
        # Update Pupil Function 
        self.pupil_func[kx_lidx:kx_hidx, ky_lidx:ky_hidx] += weight_factor_obj * delta_lowres_ft

    ################################# Plotting ###############################

    def _initialize_live_plot(self):
        """
        Initializes the live plot with two subplots: one for amplitude and one for phase.
        
        Returns:
            fig, ax: Matplotlib figure and axes.
            img_amp, img_phase: Image objects for real-time updates.
        """
    
        # Initialize empty images
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                
        # Initialize the plots with the initial image
        img_amp = axes[0].imshow(np.abs(self.hr_obj_image), vmin =.2, vmax = 1, cmap='viridis')
        axes[0].set_title("Object Object")
        cbar_amp = plt.colorbar(img_amp, ax=axes[0])
    
        img_phase = axes[1].imshow(np.abs(self.hr_fourier_image), cmap='viridis')
        axes[1].set_title("Fourier Amplitude")
        cbar_phase = plt.colorbar(img_phase, ax=axes[1])
    
        plt.tight_layout()
        plt.ion()  # Enable interactive mode
        plt.show()
    
        return fig, axes, img_amp, img_phase
    
    def _update_live_plot(self, img_amp, img_phase, fig, it):
        """
        Updates the live plot with new amplitude and phase images.
    
        Args:
            img_amp: Matplotlib image object for amplitude.
            img_phase: Matplotlib image object for phase.
            hr_obj_image: The complex object image to be plotted.
        """
        amplitude_obj = np.abs(self.hr_obj_image)
        amplitude_ft = np.abs(self.hr_fourier_image)
        
        img_amp.set_data(amplitude_obj)  # Normalize for visibility
        img_phase.set_data(amplitude_ft)
    
        amp_mean = np.mean(amplitude_obj)
        vmin = max(amp_mean - 0.1 * amp_mean, 0)
        vmax = amp_mean + 2 * amp_mean
        img_amp.set_clim(vmin, vmax)
    
        ft_mean = np.mean(amplitude_ft)
        vmin = ft_mean - 0.1 * ft_mean
        vmax = ft_mean + 2 * ft_mean
        img_phase.set_clim(vmin, vmax)
        
        fig.suptitle(f"Iteration: {it}", fontsize=12)
    
        clear_output(wait=True)
        display(fig)
        fig.canvas.flush_events()
    
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
            print("Ones")
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
        
        #image_lr = fftshift(ifft2(ifftshift(image_FT)))
        image_lr = ifft2(ifftshift(image_FT))
        
        image_lr_update = np.sqrt(image) * np.exp(1j * np.angle(image_lr))
        image_FT_update = fftshift(fft2(image_lr_update)) #* ( 1/ (pupil_func_patch +1e-23))

        weight_fac_pupil = self.alpha * self.compute_weight_fac(pupil_func_patch)
        
        # Update fourier spectrum
        delta_lowres_ft = image_FT_update - image_FT
        self.hr_fourier_image += delta_lowres_ft *  weight_fac_pupil

        if np.any(np.isnan(self.hr_fourier_image)):
            raise ValueError("There is a Nan value, check the configurations ")
            
        # Update Pupil Function 
        weight_factor_obj = self.beta * self.compute_weight_fac(self.hr_fourier_image)
        self.pupil_func[kx_lidx:kx_hidx, ky_lidx:ky_hidx] += weight_factor_obj * delta_lowres_ft


class EPRy_ones(EPRy_lr):
    

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
        
        image_lr = ifft2(ifftshift(image_FT))
        image_lr_update = np.sqrt(image) * np.exp(1j * np.angle(image_lr))
        image_FT_update = fftshift(fft2(image_lr_update)) #* ( 1/ (pupil_func_patch +1e-23))


        weight_fac_pupil = self.alpha * self.compute_weight_fac(pupil_func_patch)
        
        # Update fourier spectrum
        delta_lowres_ft = image_FT_update - image_FT
        self.hr_fourier_image += delta_lowres_ft *  weight_fac_pupil

        g_inter = ifft2(ifftshift(self.hr_fourier_image))
        g_ones = 1.0 * np.exp(1j*np.angle(g_inter))
        self.hr_fourier_image = fftshift(fft2(g_ones))
        
        if np.any(np.isnan(self.hr_fourier_image)):
            raise ValueError("There is a Nan value, check the configurations ")
            
        # Update Pupil Function 
        weight_factor_obj = self.beta * self.compute_weight_fac(self.hr_fourier_image)
        self.pupil_func[kx_lidx:kx_hidx, ky_lidx:ky_hidx] += weight_factor_obj * delta_lowres_ft


class EPRy_upsample(EPRy):
    
    def _prep_images(self):
        
        self.lr_psize = self.lr_psize /2 
        try:
            self.images = upsample_images(self.images, self.zoom_factor, n_jobs = 32)

        except: 
            print("Zoom factor not passed, default is 2.")
            self.images = upsample_images(self.images, zoom_factor=2, n_jobs = 32)
        self.images = np.array(self.images)

        
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
        
        image_lr = ifft2(ifftshift(image_FT))
        image_lr_update = np.sqrt(image) * np.exp(1j * np.angle(image_lr))
        image_FT_update = fftshift(fft2(image_lr_update)) * ( 1/ (pupil_func_patch +1e-23))
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
        

class EPRy_pad(EPRy):        

    def _prep_images(self):

        self.images = pad_to_double_parallel(self.images)
        self.images = np.array(self.images)
        
    def _initiate_recons_images(self):
        
        if self.hr_obj_image is None or self.hr_fourier_image is None: 
            self.hr_obj_image = np.ones_like(self.images[0]).astype(complex)
            self.hr_fourier_image = np.ones_like(self.images[0]).astype(complex)
        
        self.nx_lr, self.ny_lr = self.images[0].shape
        self.nx_hr, self.ny_hr = self.hr_obj_image.shape
        
        

    def _update_spectrum(self, image, kx_iter, ky_iter):
        """Handles the Fourier domain update."""
        kx_cidx = round((kx_iter - self.kx_min_n) / self.dkx)
        kx_lidx = round(max(kx_cidx - self.omega_obj_x / (2*self.dkx), 0))
        kx_hidx = round(kx_cidx + self.omega_obj_x / (2*self.dkx)) + (1 if self.nx_lr % 2 != 0 else 0)
        
        ky_cidx = round((ky_iter - self.ky_min_n) / self.dky)
        ky_lidx = round(max(ky_cidx - self.omega_obj_y / (2*self.dky), 0))
        ky_hidx = round(ky_cidx + self.omega_obj_y / (2*self.dky)) + (1 if self.ny_lr % 2 != 0 else 0)
        
        pupil_func_patch = self.pupil_func[kx_lidx:kx_hidx, ky_lidx:ky_hidx]
        image_FT = self.hr_fourier_image * pupil_func_patch 
        image_FT *= self.nx_lr/self.nx_hr 
        
        image_lr = ifft2(ifftshift(image_FT))
        image_lr_update = np.sqrt(image) * np.exp(1j * np.angle(image_lr))
        image_FT_update = fftshift(fft2(image_lr_update)) * ( 1/ (pupil_func_patch + 1e-23))
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