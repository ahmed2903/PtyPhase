from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output
from .plotting import plot_images_side_by_side
import h5py
from .utils_pr import time_it, prepare_dims, calc_obj_freq_bandwidth, pad_to_shape, make_dims_even
from scipy.ndimage import zoom 
from joblib import Parallel, delayed

class PhaseRetrievalBase(ABC):
    def __init__(self, images, 
                 pupil_func, 
                 kout_vec, 
                 ks_pupil, 
                 lr_psize,
                 alpha=0.1, 
                 beta=0.1, 
                 hr_fourier_image = None, 
                 hr_obj_image = None,
                 num_jobs=4, 
                 backend="threading"):
        
        self.images = images
        self.num_images = self.images.shape[0]
        self.img_shape = self.images.shape[1:]
        
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
        self.iter_loss = 0

        self.num_jobs = num_jobs
        self.backend = backend

    @time_it
    def prepare(self, **kwargs):
        
        print("Preparing")

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
        
        self.pupil_shape = self.pupil_func.shape
        
        self.upsample_coherent_images()
        
        self._initiate_recons_images()

        self.compute_bounds()
        
        self.PSI = self.get_psi()
    
    def _extract_patch_to_center(self, bounds, arr):
        
        (lx, hx, ly, hy), (rl, rh, cl, ch) = bounds
        out = np.zeros_like(self.pupil_func, dtype=complex)
        
        out[rl:rh, cl:ch] = arr[lx:hx, ly:hy]
        
        return out
    
    def _compute_single_exit(self, bounds, pupil, objectFT):
        """Compute exit wave for a single k-vector"""
        
        this_pupil = self._extract_patch_to_center(bounds, pupil)
        psi = this_pupil * objectFT
        
        return psi

    @time_it
    def get_psi(self):
        '''
        exit initialization where the pupil function and the object spectrum
        are at the centre
        '''
        
        exit_FT_centred = Parallel(n_jobs=self.num_jobs, backend = self.backend)(
            delayed(self._compute_single_exit)(bound, self.pupil_func, self.hr_fourier_image)
            for bound in self.patch_bounds
        )
        
        return np.array(exit_FT_centred)
    
    def _compute_patch_bounds(self, kout):
        
        kx_iter, ky_iter = kout[:2]
        
        kx_cidx = round((kx_iter - self.kx_min_n) / self.dkx)
        ky_cidx = round((ky_iter - self.ky_min_n) / self.dky)

        lx = round(max(kx_cidx - self.omega_obj_x / (2 * self.dkx), 0))
        hx = round(kx_cidx + self.omega_obj_x / (2 * self.dkx)) + (1 if self.nx_lr % 2 != 0 else 0)

        ly = round(max(ky_cidx - self.omega_obj_y / (2 * self.dky), 0))
        hy = round(ky_cidx + self.omega_obj_y / (2 * self.dky)) + (1 if self.ny_lr % 2 != 0 else 0)
        
        rl = self.pupil_shape [0]//2 - self.img_shape[0]//2
        rh = self.pupil_shape [0]//2 + self.img_shape[0]//2
        cl = self.pupil_shape [1]//2 - self.img_shape[1]//2
        ch =  self.pupil_shape [1]//2 + self.img_shape[1]//2

        return (lx, hx, ly, hy), (rl, rh, cl, ch)
    
    def compute_bounds(self):

        self.patch_bounds = []
        
        for kout in self.kout_vec:
            (lx, hx, ly, hy), (rl, rh, cl, ch) = self._compute_patch_bounds(
                kout)
            
            self.patch_bounds.append(((lx, hx, ly, hy), (rl, rh, cl, ch)))
        
        
    def _upsample_coh_img(self, image, shape):
        
        image_FT = self.forward_fft(image)
        
        new_image_FT = pad_to_shape(image_FT, shape)
        
        upsamp_img = np.abs(self.inverse_fft(new_image_FT))/ (image.shape[0]**2/new_image_FT.shape[0]**2) #Attention!!

        return upsamp_img

    @time_it
    def upsample_coherent_images(self):
        
        padded_images = Parallel(n_jobs=self.num_jobs, backend = self.backend)(
        delayed(self._upsample_coh_img)(image, self.pupil_func.shape) for image in self.images
        )
        
        self.coherent_imgs_upsampled = np.array(padded_images)
        
    def _prep_images(self):
        
        self.images = np.array(self.images)
    
    def _initiate_recons_images(self):
        
        if self.hr_obj_image is None and self.hr_fourier_image is None: 
            
            self.hr_obj_image = np.ones_like(self.images[0]).astype(complex)
            self.hr_fourier_image = np.ones_like(self.images[0]).astype(complex)
            self.hr_fourier_image = pad_to_shape(self.hr_fourier_image, self.pupil_func.shape)
            self.hr_obj_image = pad_to_shape(self.hr_obj_image, self.pupil_func.shape)
        
        elif self.hr_fourier_image is None:
            self.hr_fourier_image = self.forward_fft(self.hr_obj_image)
            self.hr_fourier_image = pad_to_shape(self.hr_fourier_image, self.pupil_func.shape)
            self.hr_obj_image = self.inverse_fft(self.hr_fourier_image)
            
        elif self.hr_obj_image is None:
            self.hr_fourier_image = pad_to_shape(self.hr_fourier_image, self.pupil_func.shape)
            self.hr_obj_image = self.inverse_fft(self.hr_fourier_image)
            
            
        self.nx_lr, self.ny_lr = self.images[0].shape
        self.nx_hr, self.ny_hr = self.hr_fourier_image.shape
        
    def _load_pupil(self):
        
        dims = round((self.kx_max_n - self.kx_min_n)/self.dkx), round((self.ky_max_n - self.ky_min_n)/self.dky)
        
        dims = make_dims_even(dims)

        full_array = np.zeros(dims, dtype = complex)
        amp_array = np.ones(dims)
        
        self.ctf = np.zeros(dims, dtype = complex)
        
        if isinstance(self.pupil_func, str):
            phase = np.load(self.pupil_func)
        elif isinstance(self.pupil_func, np.ndarray):
            phase = self.pupil_func
        else:
            phase = np.ones(dims, dtype = complex)
        
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

        self.pupil_coords = (start_x, end_x, start_y, end_y)
        
        self.ctf[start_x:end_x, start_y:end_y] = 1.0

        self.pupil_func = full_array
        # self.pupil_func = amp_array * np.exp(1j*full_array)
    
    def save_reconsturction(self, file_path):
        """
        Save the reconstruction data and metadata to an HDF5 file.
    
        Args:
            file_path (str): The path where the HDF5 file will be saved.
        """
        # Prepare metadata
        metadata = {
        "Num_iters": self.iters_passed,
        "pupil_dims" : self.pupil_func.shape,
        }

        with h5py.File(file_path, "w") as h5f:
            # Save metadata as attributes in the root group
            recon_params = h5f.create_group("Recon_Params")
            
            for key, value in metadata.items():
                recon_params.attrs[key] = value 


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
            
    # ---------- Fourier Utilities ----------
    @staticmethod
    def forward_fft(x):
        """Centered 2D FFT"""
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x))) 
    @staticmethod
    def inverse_fft(x):
        """Centered 2D iFFT"""
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))


class LivePlot:
    def _initialize_live_plot(self):
        """
        Initializes the live plot with 3 rows:
        - Row 1: Object Amplitude and Phase
        - Row 2: Fourier Amplitude and placeholder
        - Row 3: Loss plot (full width)
        
        Returns:
            fig, axes: Matplotlib figure and axes.
            img_amp, img_phase, fourier_amp: Image objects for real-time updates.
            loss_im: Line object for loss plot.
        """
        # Create figure with GridSpec for custom layout
        fig = plt.figure(figsize=(8, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.05, wspace=0.2)
        
        # First row - 2 columns
        ax_amp = fig.add_subplot(gs[0, 0])
        ax_phase = fig.add_subplot(gs[0, 1])
        
        # Second row - 2 columns
        ax_fourier = fig.add_subplot(gs[1, 0])
        ax_pupil= fig.add_subplot(gs[1, 1])  
        
        # Third row - single column spanning both columns
        ax_loss = fig.add_subplot(gs[2, :])
        
        # Store axes in a list for compatibility
        axes = [[ax_amp, ax_phase], [ax_fourier, ax_pupil], ax_loss]
        
        # Initialize the plots with the initial image
        img_amp = ax_amp.imshow(np.abs(self.hr_obj_image), cmap='viridis')
        ax_amp.set_title("Object Amplitude")
        cbar_amp = plt.colorbar(img_amp, ax=ax_amp)
        
        img_phase = ax_phase.imshow(np.angle(self.hr_obj_image), cmap='viridis')
        ax_phase.set_title("Object Phase")
        img_phase.set_clim(-np.pi, np.pi)
        cbar_phase = plt.colorbar(img_phase, ax=ax_phase)
        
        fourier_amp = ax_fourier.imshow(np.log1p(np.abs(self.hr_fourier_image)), cmap='viridis')
        ax_fourier.set_title("Fourier Amplitude")
        cbar_fourier = plt.colorbar(fourier_amp, ax=ax_fourier)
        
        # Hide the extra subplot or use it for something else
        pupil_phase = ax_pupil.imshow(np.angle(self.pupil_func), cmap='viridis')
        ax_pupil.set_title("Pupil Phase")
        pupil_phase.set_clim(-np.pi, np.pi)
        cbar_pupil = plt.colorbar(pupil_phase, ax=ax_pupil)
        
        
        # Loss plot on the full-width bottom row
        loss_im, = ax_loss.plot(range(self.iters_passed), self.losses)
        ax_loss.set_xlabel("Iteration")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(True, alpha=0.3)
        
        plt.ion()  # Enable interactive mode
        plt.show()

        self.fig, self.axes, self.img_amp, self.img_phase, self.fourier_amp, self.pupil_phase,self.loss_im = fig, axes, img_amp, img_phase, fourier_amp, pupil_phase,loss_im
        
    #return fig, axes, img_amp, img_phase, fourier_amp, pupil_phase,loss_im


    def _update_live_plot(self):
        """
        Updates the live plot with new amplitude and phase images.
        
        Args:
            img_amp: Matplotlib image object for amplitude.
            img_phase: Matplotlib image object for phase.
            fourier_amp: Matplotlib image object for Fourier amplitude.
            loss_im: Line object for loss plot.
            fig: Matplotlib figure.
            it: Current iteration number.
            axes: List of axes objects.
        """
        img_amp, img_phase, fourier_amp, pupil_phase, loss_im, fig, it, axes = self.img_amp, self.img_phase, self.fourier_amp, self.pupil_phase, self.loss_im, self.fig, self.iters_passed, self.axes
        
        amplitude_obj = np.abs(self.hr_obj_image)
        phase_obj = np.angle(self.hr_obj_image)
        amplitude_ft =np.log1p(np.abs(self.hr_fourier_image))
        pupil_pha = np.angle(self.pupil_func)
        
        img_amp.set_data(amplitude_obj)
        img_phase.set_data(phase_obj)
        fourier_amp.set_data(amplitude_ft)
        pupil_phase.set_data(pupil_pha)
        
        # Update loss plot (now on the full-width bottom row)
        ax_loss = axes[2]  # Third row
        loss_im.set_xdata(range(self.iters_passed))
        loss_im.set_ydata(self.losses)
        ax_loss.set_title(f"Iteration: {it}, loss: {self.iter_loss:.2f}", fontsize=12)
        ax_loss.relim()
        ax_loss.autoscale_view()
        
        if it > 1:
            ax_loss.set_yscale('log')
        
        # # Update amplitude colormap limits
        # amp_mean = np.mean(amplitude_obj)
        # vmin = max(amp_mean + 2 * amp_mean, 0)
        
        vmax = amplitude_obj.max()
        img_amp.set_clim(vmin=None, vmax = vmax)
        
        # # Update Fourier amplitude colormap limits
        # ft_mean = np.mean(amplitude_ft)
        # vmin = ft_mean + 2 * ft_mean
        vmax = amplitude_ft.max()
        fourier_amp.set_clim(vmin=None, vmax = vmax)
        
        clear_output(wait=True)
        display(fig)
        fig.canvas.flush_events()


class Plot:
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
    