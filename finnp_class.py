import numpy as np 
import matplotlib.pyplot as plt
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.fft as fft
import torch.nn.functional as F 
from torch.nn.utils import clip_grad_norm_
import inspect
from torch.autograd import Variable
from IPython.display import display, clear_output
import multiprocessing as mp
from functools import partial

from ..utils_pr import *
from ..plotting_fs import plot_images_side_by_side, update_live_plot, initialize_live_plot
from ..data_fs import * #downsample_array, upsample_images, pad_to_double

class ForwardModel(nn.Module):
    def __init__(self, spectrum_size, pupil_size, band_multiplier, device):
        super(ForwardModel, self).__init__()
        
        self.spectrum_size = spectrum_size
        self.pupil_size = pupil_size
        self.band_multiplier = band_multiplier

        # Initial guess: upsampled low-resolution image spectrum
        self.spectrum_amp = nn.Parameter(torch.normal(1,1,(spectrum_size[0]*self.band_multiplier, spectrum_size[1]*self.band_multiplier), dtype=torch.float64))
        self.spectrum_pha = nn.Parameter(torch.zeros(spectrum_size[0]*self.band_multiplier, spectrum_size[1]*self.band_multiplier, dtype=torch.float64))
        

        self.pupil_amp = nn.Parameter(torch.ones(pupil_size[0], pupil_size[1], dtype=torch.float64))
        self.pupil_pha = nn.Parameter(torch.zeros(pupil_size[0], pupil_size[1], dtype=torch.float64))
        
        self.ctf = mask_torch_ctf(pupil_size, device = device) 
        
        self.down_sample = nn.AvgPool2d(kernel_size=self.band_multiplier) #, divisor_override=1)

        # Learnable convolutional layer (replacing AvgPool2d)
        # self.down_sample = nn.Conv2d(1, 1, kernel_size=self.band_multiplier, stride=self.band_multiplier, padding=0, bias=False)
        
        # Initialize conv weights similar to average pooling (for stability)
        # nn.init.constant_(self.down_sample.weight, 1.0 / (self.band_multiplier ** 2))
        # nn.init.constant_(self.down_sample.bias, 0.0)
        

    def forward(self, bounds):
        """ Forward propagation: reconstruct low-resolution complex field """
        # Create complex spectrum and pupil
        spectrum = self.spectrum_amp * torch.exp(1j * self.spectrum_pha)
        
        pupil = (self.pupil_amp * self.ctf) * torch.exp(1j * self.pupil_pha * self.ctf)       
        
        sx,ex,sy,ey = bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]

        pupil_patch = pupil.index_select(0, torch.arange(sx, ex, device=pupil.device))
        pupil_patch = pupil_patch.index_select(1, torch.arange(sy, ey, device=pupil.device))

        # Apply pupil function 
        forward_spectrum = spectrum * pupil_patch

        # Inverse Fourier Transform to simulate low-resolution image
        low_res_image = fft.fftshift(fft.ifft2(fft.ifftshift(forward_spectrum)))

        low_res_amp = torch.abs(low_res_image)
        low_res_pha = torch.angle(low_res_image)

        low_res_amp = self.down_sample(low_res_amp.unsqueeze(0)).squeeze(0)
        low_res_pha = self.down_sample(low_res_pha.unsqueeze(0)).squeeze(0)
        
        low_res_image = low_res_amp * torch.exp(1j*low_res_pha)
        
        return low_res_image
    
    
class FINN:
    
    def __init__(self, images, 
                 kout_vec, 
                 lr_psize, 
                 band_multiplier=1):
        
        self.images = images
        self.kout_vec = kout_vec
        self.lr_psize = lr_psize
        self.band_multiplier = band_multiplier
        self.losses = []
        self.sec_loss_fn = None
        self.epochs_passed = 0

    def prepare(self, model,  double_pupil = False, device = "cpu"):

        
        self.set_device(device=device)
        self._prep_images()
        self.kout_vec = np.array(self.kout_vec)
        self.bounds_x, self.bounds_y, self.dks = prepare_dims(self.images, self.kout_vec, lr_psize = self.lr_psize, extend_to_double = double_pupil)
        self.kx_min_n, self.kx_max_n = self.bounds_x
        self.ky_min_n, self.ky_max_n = self.bounds_y
        self.dkx, self.dky = self.dks
        
        self.pupil_dims = round((self.kx_max_n - self.kx_min_n)/self.dkx), round((self.ky_max_n - self.ky_min_n)/self.dky)
        self.pupil_dims = make_dims_even(self.pupil_dims)
        self.omega_obj_x, self.omega_obj_y  = calc_obj_freq_bandwidth(self.lr_psize)
        
        
        self.model = model(spectrum_size = self.image_dims, pupil_size = self.pupil_dims, band_multiplier = self.band_multiplier, device = self.device).to(self.device)
        
        if self.device.type == "cuda":
            cuda_device_count = torch.cuda.device_count()
            if cuda_device_count > 1:
                print(f"Available cuda devices: {cuda_device_count}")
                self.cuda_device_count = cuda_device_count
                #self.model = nn.DataParallel(self.model)
                self.model.to(self.device)
            print('Using %s'%torch.cuda.get_device_name(0))
        else:
            print('Using %s'%self.device.type)

        self.model = self.model.double()


        #self.set_model(model, spectrum_size = self.image_dims, pupil_size = self.pupil_dims)
        
        #self._load_pupil()


    def _prep_images(self):

        self.images = np.array(self.images)
        self.image_dims = self.images[0].shape
        self.nx_lr, self.ny_lr = self.image_dims
        
        # Ensure target_image is a tensor and has the same size as reconstructed_image
        if not isinstance(self.images, torch.Tensor):
            self.images = torch.tensor(self.images, dtype=torch.float64, device = self.device)
        
        # # If target_image is not complex, convert it to complex
        # if not self.images.is_complex():
        #     self.images = self.images.to(dtype=torch.complex64)
            
            
        
    def _load_pupil(self):
        pass
        # self.pupil_dims = round((self.kx_max_n - self.kx_min_n)/self.dkx), round((self.ky_max_n - self.ky_min_n)/self.dky)

        # if isinstance(self.pupil_func, str):
        #     phase = np.load(self.pupil_func)
        # elif isinstance(self.pupil_func, np.ndarray):
        #     phase = self.pupil_func
        # else:
        #     phase = np.zeros(self.pupil_dims)

        # phase = downsample_array(phase, self.pupil_dims)
        # self.pupil_func = np.exp(1j*phase)
        

    def set_device(self, device='cpu'):
        """
        Sets the device to either CPU ('cpu') or GPU ('cuda'), if available.
        """
        if torch.cuda.is_available() and device == 'cuda':
            torch.cuda.empty_cache()
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
    def GetKwArgs(self, obj, kwargs):
        obj_sigs = []
        obj_args = {}
        for arg in inspect.signature(obj).parameters.values():
            if not arg.default is inspect._empty:
                obj_sigs.append(arg.name)
        for key, value in kwargs.items():
            if key in obj_sigs:
                obj_args[key] = value
        return obj_args

    def set_spectrum_optimiser(self, optimiser, **kwargs):
        """
        Add an optimiser. Must occur before 
        corresponding AddScheduler. 
        Specify key words for optimiser.
        """
        optimiser_args = self.GetKwArgs(optimiser, kwargs)
        if not "lr" in optimiser_args:
            raise ValueError("Learning rate must be passed")
        
        self.spectrum_optimiser = optimiser([self.model.spectrum_amp, self.model.spectrum_pha], **optimiser_args)

    def set_pupil_optimiser(self, optimiser, freeze_pupil_amp = False, **kwargs):
        """
        Add an optimiser. Must occur before 
        corresponding AddScheduler. 
        Specify key words for optimiser.
        """
        optimiser_args = self.GetKwArgs(optimiser, kwargs)
        if not "lr" in optimiser_args:
            raise ValueError("Learning rate must be passed")

        if freeze_pupil_amp:
            self.pupil_optimiser = optimiser([self.model.pupil_pha], **optimiser_args)
        else:
            self.pupil_optimiser = optimiser([self.model.pupil_amp, self.model.pupil_pha], **optimiser_args)
         
    def set_loss_func(self, loss_func, beta= 1, **kwargs):
        
        func_args = self.GetKwArgs(loss_func, kwargs)
        self.loss_fn = loss_func(**func_args)
        self.beta = beta

    def set_secondary_loss_func(self, loss_func, gamma, **kwargs):

        func_args = self.GetKwArgs(loss_func, kwargs)
        self.sec_loss_fn = loss_func(**func_args)
        self.gamma = gamma
        self.beta = 1.0-gamma

    def tv_regularization(self, image):
        """
        Compute the Total Variation (TV) regularization term
    
        Args:
        images (torch.Tensor): A 4D tensor of shape (batch_size, channels, height, width).
    
        Returns:
        torch.Tensor: The TV regularization term for each image in the batch (shape: (batch_size,)).
        """
        # Compute gradients in x and y directions
        gradient_x = torch.abs(torch.roll(image, shifts=-1, dims=1) - image)
        #gradient_x = gradient_x[:, :-1]  # Remove the last column (invalid due to roll)
        gradient_y = torch.abs(torch.roll(image, shifts=-1, dims=0) - image)
        #gradient_y = gradient_y[:-1, :]  # Remove the last row (invalid due to roll)
    
        # Compute the TV regularization term for each image
        tv = torch.sum(torch.sqrt(gradient_x**2 + gradient_y**2))
        
        return tv


    def set_alpha_scheduler(self, alpha_flag, alpha_init, alpha_steps, gamma):
        """
        Initialize the alpha scheduler.
    
        Args:
            n_epochs (int): Number of epochs to keep alpha = 0.
            alpha_init (float): Initial value of alpha after n_epochs.
            m_epochs (int): Frequency of updating alpha (every m epochs).
            gamma (float): Multiplicative factor for alpha.
    
        Returns:
            dict: A dictionary containing the scheduler parameters and state.
        """
        self.alpha = 0.0
        self.alpha_init = alpha_init
        self.alpha_flag = alpha_flag
        self.alpha_steps = alpha_steps
        self.gamma = gamma
    
    def update_alpha(self, epoch):
        """
        Update alpha based on the current epoch.
    
        Args:
            scheduler (dict): The scheduler dictionary returned by `set_alpha_scheduler`.
            epoch (int): Current epoch number.
    
        Returns:
            float: Updated value of alpha.
        """
        if epoch < self.alpha_flag:
            # First n epochs: alpha = 0
            self.alpha = 0.0
        elif epoch == self.alpha_flag:
            # After n epochs: set alpha to alpha_init
            self.alpha = self.alpha_init
            self.last_alpha_update = epoch
        elif (epoch > self.alpha_flag) and (
            (epoch - self.last_alpha_update) >= self.alpha_steps
        ):
            # Every m epochs after n_epochs: multiply alpha by gamma
            self.alpha *= self.gamma
            self.last_alpha_update = epoch
    
        
    def iterate(self, epochs, optim_flag = 5, live_flag = None, n_jobs = -1):
            
        self.num_epochs = epochs
        self.optim_flag = optim_flag
        self.last_alpha_update = self.num_epochs

        # Create separate optimizers for spectrum and pupil
        current_optimizer = self.spectrum_optimiser
        epochs_since_switch = 0  # Counter to track epochs since last switch

        if live_flag is not None:
            # plotting live loss
            plt.ion()
            fig, ax = plt.subplots()
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Live Loss Plot")
            line, = ax.plot([],[])
            plt.tight_layout()
            plt.ion()
            plt.show()
        
        
        for epoch in tqdm(range(self.num_epochs), desc="Processing", total=self.num_epochs, unit="Epochs"):
            
            current_optimizer.zero_grad()
            self.epoch_loss = 0

            #data = zip(self.images, self.kout_vec[:, 0], self.kout_vec[:, 1])
            #losses = Parallel(n_jobs=n_jobs)(delayed(self._update_spectrum)(image, kx, ky) for image, kx, ky in data)
            #self.epoch_loss += sum(losses)
            
            for i, (image, kx_iter, ky_iter) in enumerate(zip(self.images, self.kout_vec[:, 0], self.kout_vec[:, 1])):
                self._update_spectrum(image, kx_iter, ky_iter)
                
            # Backpropagation
            self.epoch_loss.backward()
            clip_grad_norm_(self.model.parameters(), max_norm = 100, norm_type=2)
            current_optimizer.step()
            
            # update switch counter
            epochs_since_switch += 1
            if epochs_since_switch >= optim_flag:
                if current_optimizer == self.spectrum_optimiser:
                    current_optimizer = self.pupil_optimiser
                else:
                    current_optimizer = self.spectrum_optimiser
                
                epochs_since_switch = 0  # Reset the counter

            # Update alpha value
            self.update_alpha(epoch)
                
            # Update loss list
            self.losses.append(self.epoch_loss.detach().cpu().numpy())

            self.epochs_passed +=1
            # Updating live plot
            if live_flag is not None:
                if epoch % live_flag == 0:
                    self._update_live_loss(fig,ax,line,epoch)

            # Logging
            if epoch % optim_flag == 0:
                print(f"Epoch [{epoch}/{self.num_epochs}], Loss: {self.epoch_loss.item():.6f} , Alpha: {self.alpha:.6f} ")
                
        self.post_process()
        
    def _update_live_loss(self, fig, ax, line, epoch):

        line.set_xdata(range(self.epochs_passed))
        line.set_ydata(self.losses)
        ax.relim()
        ax.autoscale_view()

        clear_output(wait=True)
        display(fig)
    
        # Refresh the plot
        
        #fig.canvas.draw()
        fig.canvas.flush_events()
        
        
    def _update_spectrum(self, image, kx_iter, ky_iter):
        
        #image = Variable(image).to(self.device)
        
        """Handles the Fourier domain update."""
        kx_cidx = round((kx_iter - self.kx_min_n) / self.dkx)
        kx_lidx = round(max(kx_cidx - self.omega_obj_x / (2 * self.dkx) * self.band_multiplier, 0))
        kx_hidx = round(kx_cidx + self.omega_obj_x / (2 * self.dkx) *self.band_multiplier) + (1 if self.nx_lr % 2 != 0 else 0)

        ky_cidx = round((ky_iter - self.ky_min_n) / self.dky)
        ky_lidx = round(max(ky_cidx - self.omega_obj_y / (2 * self.dky) *self.band_multiplier, 0))
        ky_hidx = round(ky_cidx + self.omega_obj_y / (2 * self.dky) *self.band_multiplier) + (1 if self.ny_lr % 2 != 0 else 0)
        
        bounds = [[kx_lidx, kx_hidx], [ky_lidx, ky_hidx]]
        reconstructed_image = self.model(bounds)

        #scale = (torch.sum(torch.sqrt(torch.abs(image)))/ torch.sum(torch.abs(reconstructed_image) ))
        #scaled_image = reconstructed_image* scale
        
        #image = torch.sqrt(image)
        #image *= (1/torch.sum(torch.abs(image)))

        tv_reg = self.tv_regularization(reconstructed_image)
        
        loss = self.loss_fn(torch.abs(reconstructed_image), torch.sqrt(torch.abs(image)))
        
        if torch.isnan(loss):
            raise ValueError("There is a Nan value, check the configurations ")
            
        self.epoch_loss += self.beta * loss  + self.alpha * tv_reg # Accumulate loss
    
        if self.sec_loss_fn is not None:
            loss2 = self.sec_loss_fn(torch.abs(reconstructed_image), torch.sqrt(torch.abs(image)))
            self.epoch_loss += self.gamma * loss2
            
    def post_process(self):
        
        spectrum_amp = self.model.spectrum_amp.detach()  # Detach from computation graph
        spectrum_pha = self.model.spectrum_pha.detach()
        
        self.recon_spectrum = spectrum_amp * torch.exp(1j * spectrum_pha)
        self.recon_spectrum = self.recon_spectrum.cpu().numpy()
        
        self.recon_obj = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(self.recon_spectrum)))
        
        pupil_amp = self.model.pupil_amp.detach()  # Detach from computation graph
        pupil_pha = self.model.pupil_pha.detach()

        self.pupil_func = pupil_amp * torch.exp(1j * pupil_pha)
        self.pupil_func = self.pupil_func.cpu().numpy()

        self.ctf = self.model.ctf.detach().cpu().numpy  # Detach from computation graph

    def plot_rec_obj(self, 
                     vmin1= None, vmax1=None, 
                     vmin2= -np.pi, vmax2=np.pi, 
                     title1 = "Object Amplitude", title2 = "Object Phase", cmap1 = "viridis", cmap2 = "viridis"):
        
        image1 = np.abs(self.recon_obj)
        image2 = np.angle(self.recon_obj)
    
        plot_images_side_by_side(image1, image2, 
                                 vmin1= vmin1, vmax1=vmax1, 
                                 vmin2= vmin2, vmax2=vmax2, 
                                 title1=title1, title2=title2, cmap1=cmap1, cmap2=cmap2, figsize=(10, 5), show = True)
    
    def plot_rec_fourier(self, 
                         vmin1= None, vmax1=None, 
                     vmin2= -np.pi, vmax2=np.pi, 
                         title1 = "Fourier Amplitude", title2 = "Fourier Phase", cmap1 = "viridis", cmap2 = "viridis"):
        
        image1 = np.abs(self.recon_spectrum)
        image2 = np.angle(self.recon_spectrum)

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

    def plot_ctf(self, 
                        vmin1= None, vmax1=None, 
                        title1 = "CTF", cmap1 = "viridis"):
    
        plot_roi_from_numpy(self.ctf, name=title, vmin1=vmin1, vmax1=vmax1)


    def plot_loss(self):

        plt.figure()
        plt.plot(self.losses)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss Metric per Epoch")
        plt.show()
    
    def save_reconsturction(self, file_path):

        # Prepare metadata
        metadata = {
            
        "Num_epochs": self.num_epochs,
        "Optim_flag": self.optim_flag,
        "upsample_factor": self.band_multiplier,
        "pupil_dims" : self.pupil_dims,
        "coherent_image_dims": self.image_dims,
        }
        optimiser_spectrum = {**self.spectrum_optimiser.param_groups[0]}
        optimiser_pupil = {**self.pupil_optimiser.param_groups[0]}

        kbounds = {
        "bounds_kx": str(self.bounds_x),
        "bouns_ky": str(self.bounds_y),
        "dks": str(self.dks),
        "obj_bandwidth_x": str(self.omega_obj_x),
        "obj_bandwidth_y": str(self.omega_obj_y)
        }

        alpha_meta = {
        "alpha_begin" : self.alpha,
        "alpha_init": self.alpha_init,
        "alpha_flag":self.alpha_flag,
        "alpha_steps": self.alpha_steps ,
        "gamma": self.gamma,
        }
        

        with h5py.File(file_path, "w") as h5f:
            # Save metadata as attributes in the root group
            recon_params = h5f.create_group("Recon Params")
            optimser1_params= h5f.create_group("Spectrum Optimiser Data")
            optimser2_params= h5f.create_group("Pupil Optimiser Data")

            kdata = h5f.create_group("K-space Params")
            tv_params = h5f.create_group("TV update")
            
            for key, value in metadata.items():
                recon_params.attrs[key] = value  # Store each parameter as an attribute
            
            for key, value in optimiser_spectrum.items():
                if key == 'params':
                    continue
                optimser1_params.attrs[key] = str(value)
                
            for key, value in optimiser_pupil.items():
                if key == 'params':
                    continue
                optimser2_params.attrs[key] = str(value)
                
            for key, value in kbounds.items():
                kdata.attrs[key] = value
            for key, value in alpha_meta.items():
                tv_params.attrs[key] = value
                        
            # Save reconstructed images 
            amp = np.abs(self.recon_obj)
            pha = np.angle(self.recon_obj)
            h5f.create_dataset("object_amplitude", data=amp, compression="gzip")
            h5f.create_dataset("object_phase", data=pha, compression="gzip")
            
            # Save spectrum 
            amp = np.abs(self.recon_spectrum)
            pha = np.angle(self.recon_spectrum)
            h5f.create_dataset("Fourier_amplitude", data=amp, compression="gzip")
            h5f.create_dataset("Fourier_phase", data=pha, compression="gzip")
            
            # Save reconstructed pupil function
            amp = np.abs(self.pupil_func)
            pha = np.angle(self.pupil_func)
            h5f.create_dataset("Pupil_amplitude", data=amp, compression="gzip")
            h5f.create_dataset("Pupil_phase", data=pha, compression="gzip")
    
            # Save loss values
            h5f.create_dataset("loss_values", data=np.array(self.losses), compression="gzip")

        
        

def train_fourier_ptychography(model, target_images, num_epochs=500, lr=0.01):
    """
    Train the Fourier Ptychography network over multiple measurements.
    
    Parameters:
    - model: Instance of FourierPtychographyNet
    - target_images: List of measured low-resolution images
    - pupil_amps: List of corresponding pupil function amplitudes
    - pupil_phas: List of corresponding pupil function phases
    - num_epochs: Number of epochs to train
    - lr: Learning rate
    """
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    loss_fn = nn.MSELoss()

    target_images = np.array(target_images)
    
    # Ensure target_image is a tensor and has the same size as reconstructed_image
    if not isinstance(target_images, torch.Tensor):
        target_images = torch.tensor(target_images, dtype=torch.float32)
    
    # If target_image is not complex, convert it to complex
    if not target_images.is_complex():
        target_images = target_images.to(dtype=torch.complex64)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        total_loss = 0  # Accumulate loss over all measurements

        # Iterate over all measurements (different illuminations)
        for i in range(len(target_images)):
            
            #CTF = CTFs[i]
            target_image = target_images[i,:,:]

            # Forward pass for the current measurement
            reconstructed_image = model()

            # Compute loss for this measurement
            loss = loss_fn(torch.abs(reconstructed_image), torch.abs(target_image))
            total_loss += loss  # Accumulate loss
        
        # Backpropagation
        total_loss.backward()
        optimizer.step()

        # Logging
        if epoch % 5 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {total_loss.item():.6f}")

    return model
