from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from .plotting import plot_images_side_by_side
import h5py

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
                 backend="numpy"):
        
        self.images = images
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
        self.iter_loss = 0
        self.zoom_factor = 1
        self.num_jobs = num_jobs
            
    @abstractmethod
    def prepare(self, **kwargs):
        """Prepare reconstruction (dims, pupil, initial HR images, etc.)."""
        pass

    @abstractmethod
    def iterate(self, iterations:int, live_plot=False, save_gif=False):
        """Run main reconstruction loop."""
        pass

    @abstractmethod
    def _update_spectrum(self, image, kx_iter, ky_iter):
        """Algorithm-specific Fourier update step."""
        pass
    
    # @abstractmethod
    # def post_process(self, *args, **kwargs):
    #     """Finalize results, cleanup, plotting."""
    #     pass
    
    @staticmethod
    def _compute_loss(pred, target):
        
        return np.sqrt(np.sum(np.abs(pred - target) ** 2)) 
    
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
        ax_pupil= fig.add_subplot(gs[1, 1])  # Extra plot if needed
        
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
        cbar_pupil = plt.colorbar(pupil_phase, ax=ax_pupil)
        
        
        # Loss plot on the full-width bottom row
        loss_im, = ax_loss.plot([], [])
        ax_loss.set_xlabel("Iteration")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(True, alpha=0.3)
        
        plt.ion()  # Enable interactive mode
        plt.show()
        
        return fig, axes, img_amp, img_phase, fourier_amp, pupil_phase,loss_im


    def _update_live_plot(self, img_amp, img_phase, fourier_amp, pupil_phase, loss_im, fig, it, axes):
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
        ax_loss.set_title(f"Iteration: {it}, loss: {self.iter_loss/self.num_images:.2f}", fontsize=12)
        ax_loss.relim()
        ax_loss.autoscale_view()
        
        if it > 1:
            ax_loss.set_yscale('log')
        
        # # Update amplitude colormap limits
        # amp_mean = np.mean(amplitude_obj)
        # vmin = max(amp_mean + 2 * amp_mean, 0)
        # vmax = amp_mean + 4 * amp_mean
        # img_amp.set_clim(vmin, vmax)
        
        # # Update Fourier amplitude colormap limits
        # ft_mean = np.mean(amplitude_ft)
        # vmin = ft_mean + 2 * ft_mean
        # vmax = ft_mean + 4 * ft_mean
        # fourier_amp.set_clim(vmin, vmax)
        
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
    