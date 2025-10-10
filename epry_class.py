
import numpy as np 
from numpy.fft import fftshift, ifftshift, fft2, ifft2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import display, clear_output
from tqdm.notebook import tqdm
import h5py
import os
from .utils_pr import *
from .plotting import plot_images_side_by_side, update_live_plot, initialize_live_plot
from scipy.ndimage import fourier_shift
from PIL import Image
#from ..data_fs import * #downsample_array, upsample_images, pad_to_double


def pad_to_double(img):
    h, w = img.shape
    
    # Calculate padding for each side
    pad_h = h // 2
    pad_w = w // 2
    
    # Pad the image with zeros
    padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    return padded_img

def crop_to_half(img):
    
    h, w = img.shape
    
    # Target dimensions (half of the current size)
    new_h, new_w = h // 2, w // 2
    
    # Calculate start indices for cropping
    start_h = (h - new_h) // 2
    start_w = (w - new_w) // 2
    
    # Crop symmetrically around the center
    cropped_img = img[start_h:start_h+new_h, start_w:start_w+new_w]
    
    return cropped_img
    
def forward_fft(arr):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(arr)))

def inverse_fft(arr):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(arr)))

class EPRy:
    
    def __init__(self, images, pupil_func: str, kout_vec, ks_pupil, 
                 lr_psize, alpha=0.1, beta=0.1, 
                 hr_obj_image=None, hr_fourier_image=None,
                 num_jobs=4):
        
        self.images = images # Coherent images
        self.num_images = self.images.shape[0]
        
        self.pupil_func = pupil_func
        self.kout_vec = kout_vec
        self.ks_pupil = ks_pupil
        self.lr_psize = lr_psize
        
        self.alpha = alpha
        self.beta = beta
        self.hr_obj_image = hr_obj_image
        self.hr_fourier_image = hr_fourier_image  
        
        self.losses = []
        self.iters_passed = 0

        self.zoom_factor = 1
        self.num_jobs = num_jobs

    ############################# Prepare ################################
    @time_it
    def prepare(self, **kwargs):
        
        print("Preparing")
        
        if 'zoom_factor' in kwargs:
            self.zoom_factor = kwargs['zoom_factor']

        if 'extend' in kwargs:
            self.extend = kwargs['extend']
        else:
            self.extend = None
            
        self._prep_images()
        
        self.kout_vec = np.array(self.kout_vec)
        self.bounds_x, self.bounds_y, self.dks = prepare_dims(self.images, self.ks_pupil, self.lr_psize, extend = self.extend)

        self.kx_min_n, self.kx_max_n = self.bounds_x
        self.ky_min_n, self.ky_max_n = self.bounds_y
        self.dkx, self.dky = self.dks
        
        omegas = calc_obj_freq_bandwidth(self.lr_psize)
        self.omega_obj_x, self.omega_obj_y = omegas

        self._load_pupil()

        self._initiate_recons_images()

    def _prep_images(self):

        self.images = np.abs(np.array(self.images))
        
    def _load_init_fs_image(self):
        
        pass
    
    def _load_pupil(self):
        
        dims = round((self.kx_max_n - self.kx_min_n)/self.dkx), round((self.ky_max_n - self.ky_min_n)/self.dky)
        #dims = round(self.omega_obj_x/self.dkx), round(self.omega_obj_y/self.dky)
        
        dims = make_dims_even(dims)

        full_array = np.zeros(dims)
        amp_array = np.ones(dims)
        
        self.ctf = np.zeros(dims).astype(complex)
        
        if isinstance(self.pupil_func, str):
            phase = np.load(self.pupil_func)
        elif isinstance(self.pupil_func, np.ndarray):
            phase = self.pupil_func
        else:
            phase = np.zeros(dims)
        
        # Get the scaling factors for each dimension
        if self.extend == 'double':
            factor = 2
        elif self.extend == 'triple':
            factor = 3
        elif self.extend == 'quadruple':
            factor = 4
        elif self.extend == 'quintiple':
            factor = 5
        elif self.extend == None:
            factor = 1
        
        scale_x = dims[0] / phase.shape[0] / factor
        scale_y = dims[1] / phase.shape[1] / factor

        # Scale the pupil phase array to match the required pupil dimensions
        scaled_pupil_phase = zoom(phase, (scale_x, scale_y))
        
        # Calculate center indices
        N, M = dims[0]//factor , dims[1]//factor
        
        start_x, start_y = (dims[0] - N) // 2, (dims[1] - M) // 2
        end_x, end_y = start_x + N, start_y + M
    
        # Set central region to ones
        full_array[start_x:end_x, start_y:end_y] = scaled_pupil_phase

        self.ctf[start_x:end_x, start_y:end_y] = 1.0
        
        self.pupil_func = amp_array* np.exp(1j*full_array)
    
    def _initiate_recons_images(self):

        
        if self.hr_obj_image is None and self.hr_fourier_image is None:
            self.hr_obj_image, self.hr_fourier_image = init_hr_image(self.bounds_x, self.bounds_y, self.dks)
        
        elif self.hr_obj_image is not None:
            self.hr_fourier_image = forward_fft(self.hr_obj_image) #fftshift(fft2(self.hr_obj_image))
            
        elif self.hr_fourier_image is not None:
            self.hr_obj_image = inverse_fft(self.hr_fourier_image) #(ifftshift(self.hr_fourier_image))

        # FIX HR OBJECT 
        
        self.nx_lr, self.ny_lr = self.images[0].shape
        self.nx_hr, self.ny_hr = self.hr_obj_image.shape

    ################################## Main Loop ##################################
    @time_it
    def iterate(self, iterations:int, live_plot=False, save_gif=False):

        if live_plot:
            fig, ax, img_amp, img_phase, fourier_amp, loss_im, axes = self._initialize_live_plot()
        if save_gif:
  
            frame_files = []
        for it in range(iterations):
            
            self.iter_loss = 0
            #for i, (image, kx_iter, ky_iter) in enumerate(tqdm(zip(self.images, self.kout_vec[:, 0], self.kout_vec[:, 1]), 
            #                                                   desc="Processing", total=len(self.images), unit="images")):
            for i, (image, kx_iter, ky_iter) in tqdm(enumerate(zip(self.images, self.kout_vec[:, 0], self.kout_vec[:, 1]))):
                
                #self._update_spectrum(image, kx_iter, ky_iter)
                self._update_spectrum(image, kx_iter, ky_iter)
                
                if np.any(np.isnan(self.hr_fourier_image)):
                    raise ValueError(f"There is a Nan in the Fourier image for {it}-th iteration,{i}-th image")
                if np.any(np.isnan(self.hr_obj_image)):
                    raise ValueError(f"There is a Nan in the Object image for {it}-th iteration,{i}-th image")
                
            if live_plot:
                # Update the HR object image after all spectrum updates in this iteration
                self.hr_obj_image = inverse_fft(self.hr_fourier_image) #ifft2(ifftshift(self.hr_fourier_image))
                self._update_live_plot(img_amp, img_phase, fourier_amp, loss_im, fig, it, axes)
            if save_gif:
                # Save the frame
                frame_file = f"tmp/frame_{it}.png"
                plt.savefig(frame_file)
                frame_files.append(frame_file)

            self.losses.append(self.iter_loss/self.num_images)
            self.iters_passed += 1

        if save_gif:
            # Create the GIF using Pillow
            frames = [Image.open(file) for file in frame_files]
            frames[0].save(
                "recon_gif.gif",
                save_all=True,
                append_images=frames[4:],
                duration=400,  # in milliseconds
                loop=0
            )
            
            # Cleanup temporary frame files
            for file in frame_files:
                os.remove(file)
                
        self.hr_obj_image = inverse_fft(self.hr_fourier_image) #(ifftshift(self.hr_fourier_image))
    
    def compute_weight_fac(self, func):
        """Compute weighting factor for phase retrieval update."""
        
        mod = np.abs(func) ** 2
        return np.conjugate(func) / (mod.max() + 1e-23)

    def _compute_loss(self, pred, target):
        
        return np.sqrt(np.sum(np.abs(pred - target) ** 2)) 
    
    def _update_spectrum(self, image, kx_iter, ky_iter):
        
        """Handles the Fourier domain update."""
        kx_cidx = round((kx_iter - self.kx_min_n) / self.dkx)
        kx_lidx = round(max(kx_cidx - self.omega_obj_x / (2 * self.dkx), 0))
        kx_hidx = round(kx_cidx + self.omega_obj_x / (2 * self.dkx)) + (1 if self.nx_lr % 2 != 0 else 0)
        
        ky_cidx = round((ky_iter - self.ky_min_n) / self.dky)
        ky_lidx = round(max(ky_cidx - self.omega_obj_y / (2 * self.dky), 0))
        ky_hidx = round(ky_cidx + self.omega_obj_y / (2 * self.dky)) + (1 if self.ny_lr % 2 != 0 else 0)

        
        image_FT = self.hr_fourier_image[kx_lidx:kx_hidx, ky_lidx:ky_hidx] * self.pupil_func
        #image_FT *= (self.nx_lr/self.nx_hr)**2
        
        image_lr = inverse_fft(image_FT) #(ifftshift(image_FT))
        image_lr_update = np.sqrt(image) * np.exp(1j * np.angle(image_lr))
        #image_lr_update *= (self.nx_hr/self.nx_lr)**2
        #inv_pupil_func = 1/ (pupil_func_patch +1e-8)
        image_FT_update = forward_fft(image_lr_update) #(fft2(image_lr_update)) #* inv_pupil_func
        
        
        # Compute update weight factor
        weight_fac_pupil = self.alpha * self.compute_weight_fac(self.hr_fourier_image[kx_lidx:kx_hidx, ky_lidx:ky_hidx])
        weight_factor_obj = self.beta * self.compute_weight_fac(self.pupil_func)

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
        self.pupil_func += weight_factor_obj * delta_lowres_ft

        image_lr_new = np.abs(inverse_fft(self.hr_fourier_image[kx_lidx:kx_hidx, ky_lidx:ky_hidx]*self.pupil_func))**2
        
        self.iter_loss+=self._compute_loss(image_lr_new, image)
        
    ################################# Plotting ###############################

    def _initialize_live_plot(self):
        """
        Initializes the live plot with two subplots: one for amplitude and one for phase.
        
        Returns:
            fig, ax: Matplotlib figure and axes.
            img_amp, img_phase: Image objects for real-time updates.
        """
    
        # Initialize empty images
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
                
        # Initialize the plots with the initial image
        img_amp = axes[0,0].imshow(np.abs(self.hr_obj_image), cmap='viridis') #, vmin =.2, vmax = 1, cmap='viridis')
        axes[0,0].set_title("Object Amplitude")
        cbar_amp = plt.colorbar(img_amp, ax=axes[0,0])

        img_phase = axes[0,1].imshow(np.angle(self.hr_obj_image), cmap='viridis')
        axes[0,1].set_title("Object Phase")
        img_phase.set_clim(-np.pi, np.pi)
        cbar_phase = plt.colorbar(img_phase, ax=axes[0,1])
        
        
        fourier_amp = axes[1,0].imshow(np.abs(self.hr_fourier_image), cmap='viridis')
        axes[1,0].set_title("Fourier Amplitude")
        cbar_fourier = plt.colorbar(fourier_amp, ax=axes[1,0])
        
        loss_im, = axes[1,1].plot([],[])
        axes[1,1].set_xlabel("iteration")
        axes[1,1].set_ylabel("loss")

        plt.tight_layout()
        plt.ion()  # Enable interactive mode
        plt.show()
    
        return fig, axes, img_amp, img_phase, fourier_amp, loss_im, axes
    
    def _update_live_plot(self, img_amp, img_phase, fourier_amp, loss_im, fig, it, axes):
        """
        Updates the live plot with new amplitude and phase images.
    
        Args:
            img_amp: Matplotlib image object for amplitude.
            img_phase: Matplotlib image object for phase.
            hr_obj_image: The complex object image to be plotted.
        """
        amplitude_obj = np.abs(self.hr_obj_image)
        phase_obj = np.angle(self.hr_obj_image)
        amplitude_ft = np.abs(self.hr_fourier_image)
        
        img_amp.set_data(amplitude_obj)  # Normalize for visibility
        img_phase.set_data(phase_obj)
        fourier_amp.set_data(amplitude_ft)
        
        loss_im.set_xdata(range(self.iters_passed))
        loss_im.set_ydata(self.losses)

        axes[1,1].set_title(f"Iteration: {it}, loss: {self.iter_loss/self.num_images:.2f}", fontsize=12)
        axes[1,1].relim()
        axes[1,1].autoscale_view()
        if it>1:
            axes[1,1].set_yscale('log')

        amp_mean = np.mean(amplitude_obj)
        vmin = max(amp_mean + 2 * amp_mean, 0)
        vmax = amp_mean + 5 * amp_mean
        img_amp.set_clim(vmin, vmax)
    
        ft_mean = np.mean(amplitude_ft)
        vmin = ft_mean + 2 * ft_mean
        vmax = ft_mean + 5 * ft_mean
        fourier_amp.set_clim(vmin, vmax)
            
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
                                 title1=title1, title2=title2, 
                                 cmap1=cmap1, cmap2=cmap2, 
                                 figsize=(10, 5), show = True)

    def plot_losses(self):

        plt.figure()
        plt.plot(self.losses)
        plt.xlabel("Iteration")
        plt.ylabel("Error")
        plt.title("Error Graph")


    def save_reconsturction(self, file_path):
        """
        Save the reconstruction data and metadata to an HDF5 file.
    
        Args:
            file_path (str): The path where the HDF5 file will be saved.
        """
        # Prepare metadata
        metadata = {
            
        "Num_iters": self.iters_passed,
        "upsample_factor": self.zoom_factor,
        "pupil_dims" : self.pupil_func.shape,
        "coherent_image_dims": self.images[0].shape,
        }

        kbounds = {
        "bounds_kx": str(self.bounds_x),
        "bounds_ky": str(self.bounds_y),
        "dks": str(self.dks),
        "obj_bandwidth_x": str(self.omega_obj_x),
        "obj_bandwidth_y": str(self.omega_obj_y)
        }


        with h5py.File(file_path, "w") as h5f:
            # Save metadata as attributes in the root group
            recon_params = h5f.create_group("Recon_Params")

            kdata = h5f.create_group("K_space_Params")
            
            for key, value in metadata.items():
                recon_params.attrs[key] = value 
                
            for key, value in kbounds.items():
                kdata.attrs[key] = value


            recon_group = h5f.create_group("Reconstructed_data")
            # Save reconstructed images 
            amp = np.abs(self.hr_obj_image)
            pha = np.angle(self.hr_obj_image)
            recon_group.create_dataset("Object_amplitude", data=amp, compression="gzip")
            recon_group.create_dataset("Object_phase", data=pha, compression="gzip")
            
            # Save spectrum 
            amp = np.abs(self.hr_fourier_image)
            pha = np.angle(self.hr_fourier_image)
            recon_group.create_dataset("Fourier_amplitude", data=amp, compression="gzip")
            recon_group.create_dataset("Fourier_phase", data=pha, compression="gzip")
            
            # Save reconstructed pupil function
            amp = np.abs(self.pupil_func)
            pha = np.angle(self.pupil_func)
            recon_group.create_dataset("Pupil_amplitude", data=amp, compression="gzip")
            recon_group.create_dataset("Pupil_phase", data=pha, compression="gzip")
    
            # Save loss values
            losses = h5f.create_group("Losses")
            losses.create_dataset("main_loss_values", data=np.array(self.losses), compression="gzip")
        
class EPRy_lr(EPRy):
    def _prep_images(self):
        
        self.images = np.abs(np.array(self.images))

        if self.zoom_factor > 1:
            self.images = upsample_images(self.images, self.zoom_factor, n_jobs = self.num_jobs)
            self.lr_psize = self.lr_psize / self.zoom_factor 
            
        self.images = np.array(self.images)
        
    def _initiate_recons_images(self):
        if self.hr_obj_image is None and self.hr_fourier_image is None: 
            self.hr_obj_image = np.ones_like(self.images[0]).astype(complex)
            self.hr_fourier_image = np.ones_like(self.images[0]).astype(complex)
        
        elif self.hr_obj_image is not None:
            self.hr_fourier_image = forward_fft(self.hr_obj_image) #(fft2(self.hr_obj_image))
            
        elif self.hr_fourier_image is not None:
            self.hr_obj_image = inverse_fft(self.hr_fourier_image) #(ifftshift(self.hr_fourier_image))
            
        self.nx_lr, self.ny_lr = self.images[0].shape
        self.nx_hr, self.ny_hr = self.hr_obj_image.shape

    def _update_spectrum(self, image,  ky_iter, kx_iter):
        """Handles the Fourier domain update."""
        kx_cidx = round((kx_iter - self.kx_min_n) / self.dkx)
        kx_lidx = round(max(kx_cidx - self.omega_obj_x / (2 * self.dkx), 0))
        kx_hidx = round(kx_cidx + self.omega_obj_x / (2 * self.dkx)) + (1 if self.nx_lr % 2 != 0 else 0)
        
        ky_cidx = round((ky_iter - self.ky_min_n) / self.dky)
        ky_lidx = round(max(ky_cidx - self.omega_obj_y / (2 * self.dky), 0))
        ky_hidx = round(ky_cidx + self.omega_obj_y / (2 * self.dky)) + (1 if self.ny_lr % 2 != 0 else 0)

        
        ctf_patch = self.ctf[kx_lidx:kx_hidx, ky_lidx:ky_hidx]
        pupil_func_patch = self.pupil_func[kx_lidx:kx_hidx, ky_lidx:ky_hidx]

        
        image_FT = self.hr_fourier_image * pupil_func_patch #* ctf_patch
        
        #image_lr = fftshift(ifft2(ifftshift(image_FT)))
        image_lr = inverse_fft(image_FT) #ifft2(ifftshift(image_FT))

        
        image_lr_update = np.sqrt(image) * np.exp(1j *np.angle(image_lr))
        
        image_FT_update = forward_fft(image_lr_update) #* ctf_patch #(fft2(image_lr_update)) 

        
        weight_fac_pupil = self.alpha * self.compute_weight_fac(pupil_func_patch)
        weight_factor_obj = self.beta * self.compute_weight_fac(self.hr_fourier_image)
        
        # Update fourier spectrum
        delta_lowres_ft = image_FT_update - image_FT
        self.hr_fourier_image += delta_lowres_ft * weight_fac_pupil 

        if np.any(np.isnan(self.hr_fourier_image)):
            raise ValueError("There is a Nan value, check the configurations ")
            
        # Update Pupil Function 
        self.pupil_func[kx_lidx:kx_hidx, ky_lidx:ky_hidx] += weight_factor_obj * delta_lowres_ft * ctf_patch

        new_exit = self.hr_fourier_image*self.pupil_func[kx_lidx:kx_hidx, ky_lidx:ky_hidx]

        image_lr_new = np.abs(inverse_fft(new_exit))**2
                
        self.iter_loss += self._compute_loss(image_lr_new, image)


class EPRy_high(EPRy):
    @time_it
    def prepare(self, **kwargs):
        
        print("Preparing")
        
        if 'zoom_factor' in kwargs:
            self.zoom_factor = kwargs['zoom_factor']

        if 'extend' in kwargs:
            self.extend = kwargs['extend']
        else:
            self.extend = None
            
        
        
        self.kout_vec = np.array(self.kout_vec)
        self.bounds_x, self.bounds_y, self.dks = prepare_dims(self.images, self.ks_pupil, self.lr_psize, extend = self.extend)

        self.kx_min_n, self.kx_max_n = self.bounds_x
        self.ky_min_n, self.ky_max_n = self.bounds_y
        self.dkx, self.dky = self.dks
        
        omegas = calc_obj_freq_bandwidth(self.lr_psize)
        self.omega_obj_x, self.omega_obj_y = omegas

        self._load_pupil()
        self._prep_images()
        self._initiate_recons_images()

        # self.ctf = np.ones_like(self.pupil_func)
        # self.ctf = crop_to_half(self.ctf)
        # self.ctf = pad_to_shape(self.ctf, self.pupil_func.shape).astype(complex)     
    
    def _prep_images(self):
        
        self.images = np.abs(np.array(self.images))
        
          
        
        if self.zoom_factor > 1:
            self.images = upsample_images(self.images, self.zoom_factor, n_jobs = self.num_jobs)
            self.lr_psize = self.lr_psize / self.zoom_factor 

        print(self.pupil_func.shape)
        self.images = pad_to_shape_parallel(self.images, self.pupil_func.shape, n_jobs=64)
        self.images = np.array(self.images)
        
    def _initiate_recons_images(self):

        if self.hr_obj_image is None and self.hr_fourier_image is None:
            self.hr_obj_image, self.hr_fourier_image = init_hr_image(self.bounds_x, self.bounds_y, self.dks)
        
        elif self.hr_obj_image is not None:
            self.hr_fourier_image = forward_fft(self.hr_obj_image) #fftshift(fft2(self.hr_obj_image))
            
        elif self.hr_fourier_image is not None:
            self.hr_obj_image = inverse_fft(self.hr_fourier_image) #(ifftshift(self.hr_fourier_image))

        
        self.nx_lr, self.ny_lr = self.images[0].shape
        self.nx_hr, self.ny_hr = self.hr_obj_image.shape

        


    def shift_array(self, kx_iter, ky_iter):
        
        kx_cidx = round((kx_iter - self.kx_min_n) / self.dkx)
        ky_cidx = round((ky_iter - self.ky_min_n) / self.dky)

        kx_center = self.nx_hr // 2
        ky_center = self.ny_hr // 2

        kx_offset = kx_cidx - kx_center
        ky_offset = ky_cidx - ky_center
        
        pupil_shifted = fourier_shift(self.pupil_func, shift=(ky_offset, kx_offset))

        return pupil_shifted
        
    def _update_spectrum(self, image, kx_iter, ky_iter):
        """Handles the Fourier domain update."""


        shifted_hr_image = self.shift_array(kx_iter, ky_iter)
        
        image_FT = self.hr_fourier_image * self.pupil_func * self.ctf

        
        image_lr = inverse_fft(image_FT) #ifft2(ifftshift(image_FT))
        #image = pad_to_pupil(image, self.pupil_func.shape)
        
        image_lr_update = np.sqrt(image) * np.exp(1j * np.angle(image_lr))
        
        image_FT_update = forward_fft(image_lr_update) #(fft2(image_lr_update)) 

        
        weight_fac_pupil = self.alpha * self.compute_weight_fac(self.pupil_func)
        weight_factor_obj = self.beta * self.compute_weight_fac(self.hr_fourier_image)
        
        # Update fourier spectrum
        delta_lowres_ft = image_FT_update - image_FT
        self.hr_fourier_image += delta_lowres_ft * weight_fac_pupil 

        if np.any(np.isnan(self.hr_fourier_image)):
            raise ValueError("There is a Nan value, check the configurations ")
            
        # Update Pupil Function 
        self.pupil_func += weight_factor_obj * delta_lowres_ft 

        new_exit = self.hr_fourier_image*self.pupil_func * self.ctf #[kx_lidx:kx_hidx, ky_lidx:ky_hidx]

        # Test 
        image_lr_new = np.abs(inverse_fft(new_exit))
        
        #image_lr_new = np.abs(ifft2(ifftshift(self.hr_fourier_image*self.pupil_func[kx_lidx:kx_hidx, ky_lidx:ky_hidx])))**2
        
        self.iter_loss+=self._compute_loss(image_lr_new, image)


class EPRy_fsc_lr(EPRy_lr):
    @time_it
    def iterate(self, iterations:int, start_fsc:int, live_plot=False):

        if live_plot:
            fig, ax, img_amp, img_phase, fourier_amp, loss_im, axes = self._initialize_live_plot()
        
        for it in range(iterations):
            
            self.iter_loss = 0
            for i, (image, kx_iter, ky_iter) in enumerate(tqdm(zip(self.images, self.kout_vec[:, 0], self.kout_vec[:, 1]), 
                                                               desc="Processing", total=len(self.images), unit="images")):

                
                self._update_spectrum(image, kx_iter, ky_iter)
                
                if np.any(np.isnan(self.hr_fourier_image)):
                    raise ValueError(f"There is a Nan in the Fourier image for {it}-th iteration,{i}-th image")
                if np.any(np.isnan(self.hr_obj_image)):
                    raise ValueError(f"There is a Nan in the Object image for {it}-th iteration,{i}-th image")
            
            if self.iters_passed % start_fsc == 0 and self.iters_passed != 0:
                print("fourier space constraint")
                self._center_fourier_spectrum()
                
            if live_plot:
                # Update the HR object image after all spectrum updates in this iteration
                self.hr_obj_image = inverse_fft(self.hr_fourier_image) #ifft2(ifftshift(self.hr_fourier_image))
                self._update_live_plot(img_amp, img_phase, fourier_amp, loss_im, fig, it, axes)

            self.losses.append(self.iter_loss/self.num_images)
            self.iters_passed += 1
            
        self.hr_obj_image = inverse_fft(self.hr_fourier_image) #(ifftshift(self.hr_fourier_image))
        

    def _center_fourier_spectrum(self):
        print("here")
        F = self.hr_fourier_image
        power = np.abs(F) ** 2

        # Create coordinate grids
        ny, nx = F.shape
        x = np.arange(nx)
        y = np.arange(ny)
        X, Y = np.meshgrid(x, y)

        # Compute weighted centroid
        total_power = np.sum(power) + 1e-8
        cx = np.sum(X * power) / total_power
        cy = np.sum(Y * power) / total_power

        # Desired center
        cx_target = nx // 2
        cy_target = ny // 2

        shift_x = -round(cx_target - cx)
        shift_y = -round(cy_target - cy)

        print(shift_y, shift_x)
        # Apply shift to recenter spectrum (fractional pixel shift allowed)
        self.hr_fourier_image = forward_fft(fourier_shift(inverse_fft(F), shift=(shift_y, shift_x))) #fft2(fourier_shift(ifft2(F), shift=(shift_y, shift_x)))

class EPRy_upsample(EPRy_lr):
    
    def _prep_images(self):
        
        try:
            self.images = upsample_images(self.images, self.zoom_factor, n_jobs = self.num_jobs)
            self.lr_psize = self.lr_psize / self.zoom_factor 

        except: 
            print("Zoom factor not passed, default is 2.")
            self.lr_psize = self.lr_psize / 2 
            self.images = upsample_images(self.images, zoom_factor=2, n_jobs = self.num_jobs)
        self.images = np.array(self.images)
        
class EPRy_pad(EPRy_lr):        

    def _prep_images(self):

        self.images = pad_to_double_parallel(self.images, n_jobs=self.num_jobs)
        self.images = np.array(self.images)
        