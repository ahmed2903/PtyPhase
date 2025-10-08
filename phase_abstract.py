from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output

class PhaseRetrievalBase(ABC):
    def __init__(self, images, 
                 pupil_func, 
                 kout_vec, 
                 ks_pupil, 
                 lr_psize,
                 alpha=0.1, 
                 beta=0.1, 
                 rec_obj_image = None, 
                 rec_fourier_image = None,
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
        self.rec_obj_image = rec_obj_image
        self.rec_fourier_image = rec_fourier_image
        self.losses = []
        self.iters_passed = 0
        self.zoom_factor = 1
        self.num_jobs = num_jobs
        self.backend = backend

        # Backend selection
        if backend == "torch":
            import torch
            self.xp = torch
            self.fft2 = torch.fft.fft2
            self.ifft2 = torch.fft.ifft2
            self.fftshift = torch.fft.fftshift
            self.ifftshift = torch.fft.ifftshift
        else:
            import numpy as np
            from numpy.fft import fft2, ifft2, fftshift, ifftshift
            self.xp = np
            self.fft2 = fft2
            self.ifft2 = ifft2
            self.fftshift = fftshift
            self.ifftshift = ifftshift
            
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
    
    @abstractmethod
    def post_process(self, *args, **kwargs):
        """Finalize results, cleanup, plotting."""
        pass
    
    # ---------- Fourier Utilities ----------
    def fft2c(self, x):
        """Centered 2D FFT"""
        return self.fft.fftshift(self.fft.fft2(self.fft.ifftshift(x))) 

    def ifft2c(self, x):
        """Centered 2D iFFT"""
        return self.fft.fftshift(self.fft.ifft2(self.fft.ifftshift(x)))


class LivePlot:
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
        img_amp = axes[0,0].imshow(np.abs(self.rec_obj_image), vmin =.2, vmax = 1, cmap='viridis')
        axes[0,0].set_title("Object Amplitude")
        cbar_amp = plt.colorbar(img_amp, ax=axes[0,0])

        img_phase = axes[0,1].imshow(np.angle(self.rec_obj_image), cmap='viridis')
        axes[0,1].set_title("Object Phase")
        img_phase.set_clim(-np.pi, np.pi)
        cbar_phase = plt.colorbar(img_phase, ax=axes[0,1])
        
        
        fourier_amp = axes[1,0].imshow(np.abs(self.rec_fourier_image), cmap='viridis')
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
        amplitude_obj = np.abs(self.rec_obj_image)
        phase_obj = np.angle(self.rec_obj_image)
        amplitude_ft = np.abs(self.rec_fourier_image)
        
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
        vmax = amp_mean + 10 * amp_mean
        img_amp.set_clim(vmin, vmax)
    
        ft_mean = np.mean(amplitude_ft)
        vmin = ft_mean + 2 * ft_mean
        vmax = ft_mean + 10 * ft_mean
        fourier_amp.set_clim(vmin, vmax)
            
        clear_output(wait=True)
        display(fig)
        fig.canvas.flush_events()
        ...