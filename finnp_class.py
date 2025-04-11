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

from .shrink_wrap import ShrinkWrap
from .utils_pr import *
from .plotting import plot_images_side_by_side, plot_roi_from_numpy
from tqdm.notebook import tqdm, trange
from scipy.ndimage import zoom 

class ForwardModel(nn.Module):
    """
    A neural network-based forward model for simulating the low-resolution images 
    from a high-resolution spectrum using a pupil function. The object spectrum and 
    the pupil function are treated as layers with learnable parameters.

    Attributes:
        spectrum_amp (torch.nn.Parameter): Amplitude of the high-resolution spectrum.
        spectrum_pha (torch.nn.Parameter): Phase of the high-resolution spectrum.
        pupil_amp (torch.nn.Parameter): Amplitude of the pupil function.
        pupil_pha (torch.nn.Parameter): Phase of the pupil function.
        ctf (torch.Tensor): Contrast transfer function (CTF) mask.
        down_sample (torch.nn.AvgPool2d): Downsampling operation for generating low-resolution images.
    """
    def __init__(self, spectrum_size, pupil_size, band_multiplier, device):
        """
        Initializes the ForwardModel with given parameters.

        Args:
            spectrum_size (tuple): The size of the high-resolution spectrum (height, width).
            pupil_size (tuple): The size of the pupil function (height, width).
            band_multiplier (int): The factor by which the spectrum is upsampled.
            device (torch.device): The computing device (CPU or GPU).
        """
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
        
        self.down_sample = nn.AvgPool2d(kernel_size=self.band_multiplier) 
        

    def forward(self, bounds):
        """
        Simulates the forward propagation to generate a low-resolution image.

        Args:
            bounds (list of tuples): The spatial region (sx, ex, sy, ey) for extracting a pupil function patch.

        Returns:
            torch.Tensor: The simulated low-resolution image as a complex tensor.
        """
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
    """
    A neural network-based framework for image reconstruction using Fourier-based methods.

    This class handles loading images, preparing dimensions, performing reconstruction using 
    a neural network, visualizing images and loss functions, setting loss functions, optimizers, 
    and all related components for the reconstruction process.
    """
    def __init__(self, images, 
                 kin_vec, 
                 pupil_kins, 
                 lr_psize, 
                 band_multiplier=1,
                 debug=False,
                    verbose=True):
        """
        Initializes the FINN reconstruction framework.

        Args:
            images (numpy.ndarray or torch.Tensor): The input images for reconstruction.
            kin_vec (numpy.ndarray or torch.Tensor): The k-space output vectors.
            lr_psize (float): The pixel size of the low-resolution images.
            band_multiplier (int, optional): Factor for upsampling the spectrum. Default is 1.
        """
        self.images = images
        self.kin_vec = kin_vec
        self.pupil_kins = pupil_kins
        self.lr_psize = lr_psize
        self.band_multiplier = band_multiplier
        self.tv_losses = []
        self.supp_losses = []
        self.main_losses = []
        self.sec_loss_fn = None
        self.pupil_optimiser = None
        self.epochs_passed = 0
        self.num_epochs = 0 
        self.grad_norms = {}
        self.debug = debug
        self.verbose = verbose
        
    def prepare(self, model,  extend_pupil = None, device = "cpu"):

        """
        Prepares the reconstruction pipeline by setting the device, processing image dimensions, 
        and initializing the neural network model.

        This method:
        - Sets the computation device (CPU or GPU).
        - Prepares images and converts kout vectors to NumPy arrays.
        - Computes spatial frequency bounds and step sizes.
        - Determines the pupil dimensions, ensuring they are even.
        - Calculates the object frequency bandwidth.
        - Initializes the neural network model with the computed dimensions.

        Args:
            model (torch.nn.Module): The neural network model for reconstruction.
            double_pupil (str, optional): Mode of pupil extension. "by_bandwidth", "double", None. Default is None
            device (str, optional): The computation device ('cpu' or 'cuda'). Default is 'cpu'.
        """
        self.set_device(device=device)
        self._prep_images()
        self.kin_vec = np.array(self.kin_vec)
        self.bounds_x, self.bounds_y, self.dks = prepare_dims(self.images, self.pupil_kins, lr_psize = self.lr_psize, extend = extend_pupil)
        
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

        self.grad_norms = {name: [] for name, param in self.model.named_parameters()}  # Initialize empty lists



    def _prep_images(self):
        """
        Converts the input images to a NumPy array and ensures they are stored as a PyTorch tensor.

        This method:
        - Converts `self.images` into a NumPy array if it is not already one.
        - Extracts image dimensions and assigns them to `self.image_dims`.
        - Ensures `self.images` is a PyTorch tensor of type float64 and moves it to the specified device.

        """
        self.images = np.array(self.images)
        self.image_dims = self.images[0].shape
        self.nx_lr, self.ny_lr = self.image_dims
        
        # Ensure target_image is a tensor and has the same size as reconstructed_image
        if not isinstance(self.images, torch.Tensor):
            self.images = torch.tensor(self.images, dtype=torch.float64, device = self.device)
        
    def save_model(self, file_path):
        """
        Save the model's state to a file.
    
        Args:
            file_path (str): The path where the model's state will be saved.

        """
        # Save the model's state_dict (parameters)
        torch.save(self.model.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        """
        Load the model's state from a file.
    
        Args:
            file_path (str): The path from which the model's state will be loaded.
        """
        # Load the model's state_dict (parameters)
        self.model.load_state_dict(torch.load(file_path))
        self.model.eval()  # Set model to evaluation mode (if needed)
        print(f"Model loaded from {file_path}")
        
    def load_pupil(self, pupil_phase_array):
        """
        Load and scale the initial pupil phase guess to fit within self.pupil_dims, 
        while ensuring that it can be updated with gradients during optimization.
    
        Args:
            pupil_phase_array (numpy.ndarray): The initial pupil phase guess.

        """
        if isinstance(pupil_phase_array, str):
            pupil_phase_array = np.load(pupil_phase_array)
        
        # Get the scaling factors for each dimension
        scale_x = self.pupil_dims[0] / pupil_phase_array.shape[0] / 2
        scale_y = self.pupil_dims[1] / pupil_phase_array.shape[1] / 2 
        
        # Scale the pupil phase array to match the required pupil dimensions
        scaled_pupil_phase = zoom(pupil_phase_array, (scale_x, scale_y))
        
        # Convert the scaled pupil phase to a tensor that requires gradients
        pupil_tensor = torch.tensor(scaled_pupil_phase, 
                                        dtype=torch.float64, 
                                        device=self.device)

        full_tensor = torch.zeros(self.pupil_dims, dtype=torch.float64, device=self.device)
    
        # Calculate center indices
        N, M = self.pupil_dims[0]//2, self.pupil_dims[1]//2
        
        start_x, start_y = (self.pupil_dims[0] - N) // 2, (self.pupil_dims[1] - M) // 2
        end_x, end_y = start_x + N, start_y + M
    
        # Set central region to ones
        full_tensor[start_x:end_x, start_y:end_y] = pupil_tensor.detach().clone().requires_grad_()

        self.model.pupil_pha = nn.Parameter(full_tensor)

    def set_device(self, device='cpu'):
        """
        Configures the computation device for the model.

        This method:
        - Checks if a GPU is available when 'cuda' is specified.
        - Clears CUDA memory cache if switching to GPU.
        - Sets `self.device` to either CPU or GPU accordingly.

        Args:
            device (str): The desired computation device ('cpu' or 'cuda').
        """
        if torch.cuda.is_available() and device == 'cuda':
            torch.cuda.empty_cache()
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
    def GetKwArgs(self, obj, kwargs):
        """
        Extracts valid keyword arguments for a given object's function signature.

        This method:
        - Retrieves the parameter names of `obj` that have default values.
        - Filters `kwargs` to retain only those keys that match these parameters.
        - Returns a dictionary containing only valid keyword arguments.

        Args:
            obj (callable): The function or method whose signature is analyzed.
            kwargs (dict): The dictionary of keyword arguments to filter.

        Returns:
            dict: A dictionary containing valid keyword arguments for `obj`.
        """
        obj_sigs = []
        obj_args = {}
        for arg in inspect.signature(obj).parameters.values():
            if not arg.default is inspect._empty:
                obj_sigs.append(arg.name)
        for key, value in kwargs.items():
            if key in obj_sigs:
                obj_args[key] = value
        return obj_args
    ############################## Set Optimisers ##############################
    def set_pupil_schedular(self, scheduler, **kwargs):
        """
        Add scheduler function to schedule LR update.
        StepLR just multiplies LR by gamma by n epochs.
        """
        scheduler_args = self.GetKwArgs(scheduler, kwargs)
        if scheduler is optim.lr_scheduler.StepLR and not "step_size" in scheduler_args:
            scheduler_args['step_size'] = kwargs['step_size']
        print(f"schedular argss: {scheduler_args}")
        self.pupil_schedular = scheduler(self.pupil_optimiser, **scheduler_args)

    def set_spectrum_schedular(self, scheduler, **kwargs):
        """
        Add scheduler function to schedule LR update.
        StepLR just multiplies LR by gamma by n epochs.
        """
        scheduler_args = self.GetKwArgs(scheduler, kwargs)
        
        if scheduler is optim.lr_scheduler.StepLR and not "step_size" in scheduler_args:
            scheduler_args['step_size'] = kwargs['step_size']
        print(f"schedular argss: {scheduler_args}")
        self.spectrum_schedular = scheduler(self.spectrum_optimiser, **scheduler_args)
        
    def set_spectrum_optimiser(self, optimiser, **kwargs):
        """
        Sets the optimizer for the spectrum parameters.

        This method:
        - Filters valid keyword arguments for the optimizer using `GetKwArgs`.
        - Ensures that a learning rate ("lr") is provided.
        - Initializes the optimizer with the model's spectrum amplitude and phase parameters.

        Args:
            optimiser (torch.optim.Optimizer): The optimizer class to be used.
            **kwargs: Additional keyword arguments for the optimizer.

        Raises:
            ValueError: If the learning rate ("lr") is not provided in `kwargs`.
        """
        optimiser_args = self.GetKwArgs(optimiser, kwargs)
        if not "lr" in optimiser_args:
            raise ValueError("Learning rate must be passed")
        
        self.spectrum_optimiser = optimiser([self.model.spectrum_amp, self.model.spectrum_pha], 
                                             #self.model.pupil_amp, self.model.pupil_pha], 
                                            **optimiser_args)
        
    def set_pupil_optimiser(self, optimiser, freeze_pupil_amp = False, **kwargs):
        """
        Sets the optimizer for the pupil parameters.

        This method:
        - Filters valid keyword arguments for the optimizer using `GetKwArgs`.
        - Ensures that a learning rate ("lr") is provided.
        - Initializes the optimizer with the model's spectrum amplitude and phase parameters.

        Args:
            optimiser (torch.optim.Optimizer): The optimizer class to be used.
            **kwargs: Additional keyword arguments for the optimizer.

        Raises:
            ValueError: If the learning rate ("lr") is not provided in `kwargs`.
        """
        optimiser_args = self.GetKwArgs(optimiser, kwargs)
        if not "lr" in optimiser_args:
            raise ValueError("Learning rate must be passed")

        if freeze_pupil_amp:
            self.pupil_optimiser = optimiser([self.model.pupil_pha], **optimiser_args)
        else:
            self.pupil_optimiser = optimiser([self.model.pupil_amp, self.model.pupil_pha], **optimiser_args)

        self.spectrum_optimiser.param_groups[0]['params'] = [self.model.spectrum_amp, self.model.spectrum_pha]
    ######################################################################################################
    ################################## Losses and Regs ###################################################
    ######################################################################################################
    
    ################################## Loss Functions ##################################
    def set_loss_func(self, loss_func, beta= 1, **kwargs):
        """
        Sets the primary loss function for the reconstruction process.

        This method:
        - Extracts valid keyword arguments for the loss function using `GetKwArgs`.
        - Initializes the loss function with the provided parameters.
        - Sets a weighting factor `beta` for potential loss scaling.

        Args:
            loss_func (callable): The loss function to be used.
            beta (float, optional): A scaling factor for the loss function. Default is 1.
            **kwargs: Additional keyword arguments for the loss function.
        """
        func_args = self.GetKwArgs(loss_func, kwargs)
        if isinstance(loss_func, type):  
            self.loss_fn = loss_func(**func_args)  # Instantiating the loss function
        else:
            self.loss_fn = loss_func
        self.beta = beta

    def set_secondary_loss_func(self, loss_func, delta, **kwargs):
        """
        Sets the secondary loss function and adjusts the weighting factors.

        This method:
        - Extracts valid keyword arguments for the loss function using `GetKwArgs`.
        - Initializes the secondary loss function with the provided parameters.
        - Sets the weighting factors `gamma` for the secondary loss and updates `beta` accordingly.

        Args:
            loss_func (callable): The secondary loss function to be used.
            delta (float): Weighting factor for the secondary loss function.
            **kwargs: Additional keyword arguments for the loss function.
        """
        func_args = self.GetKwArgs(loss_func, kwargs)
        if isinstance(loss_func, type):  
            self.loss_fn = loss_func(**func_args)  # Instantiating the loss function
        else:
            self.loss_fn = loss_func
        self.delta = delta
        self.beta = 1.0-delta
    ################################## TV Regularisaton ##################################
    def tv_regularization(self, image):
        """
        Compute the Total Variation (TV) regularization term.

        This method computes the Total Variation regularization for a given image. The TV regularization term 
        encourages smoothness by penalizing large gradients between adjacent pixels. It is often used in image 
        reconstruction to avoid overfitting and promote image smoothness.

        Args:
            image (torch.Tensor): A tensor representing the image or batch of images, with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The TV regularization term for each image in the batch (shape: (batch_size,)).
        """
        # Compute gradients in x and y directions
        gradient_x = torch.abs(torch.roll(image, shifts=-1, dims=1) - image)
        #gradient_x = gradient_x[:, :-1]  # Remove the last column 
        gradient_y = torch.abs(torch.roll(image, shifts=-1, dims=0) - image)
        #gradient_y = gradient_y[:-1, :]  # Remove the last row 
    
        # Compute the TV regularization term for each image
        tv = torch.sum(torch.sqrt(gradient_x**2 + gradient_y**2))
        
        return tv

    def set_tv_param(self, alpha_min, alpha_max):
        """
        Initialize the alpha tv scheduler.

        This method sets up the alpha scheduler, which controls the alpha parameter during training. The 
        alpha parameter can evolve based on the specified frequency and the multiplicative factor gamma.

        Args:
            alpha_flag (bool): Flag indicating whether to enable alpha scheduling.
            alpha_init (float): Initial value of alpha after the specified number of epochs.
            alpha_steps (int): Number of epochs after which alpha is updated. (Default = 1)
            gamma (float): Multiplicative factor for updating alpha. (Default = 1)

        """
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        
    def update_alpha(self, epoch):
        """
        Update alpha based on the current epoch.
    
        This method updates the alpha parameter based on the current epoch. Initially, alpha is set to zero 
        for the first n epochs. After that, it is set to the initial value (`alpha_init`), and every `alpha_steps` 
        epochs, alpha is multiplied by a factor (`gamma`).

        Args:
            epoch (int): Current epoch number.

        Returns:
            float: Updated value of alpha.
        """

        if epoch < self.alpha_flag:
            # First n epochs: alpha = 0
            self.last_alpha_update = self.num_epochs
            self.alpha = 0.0
        elif epoch == self.alpha_flag:
            # After n epochs: set alpha to alpha_init
            self.alpha = self.alpha_init
            self.last_alpha_update = epoch
        
        
        elif (epoch > self.alpha_flag) and (
            (epoch - self.last_alpha_update) >= self.alpha_steps
        ):
            
            self.alpha *= self.gamma
            self.last_alpha_update = epoch
            
    ################################## Support Penalty ##################################
    def support_penalty(self, image):
        """Computes a penalty term based on the support constraint.

        This method penalizes intensity outside the defined support region by calculating 
        the fraction of the image that lies outside the support mask.
    
        Args:
            image (torch.Tensor): The input image (typically a reconstructed amplitude array).
    
        Returns:
            torch.Tensor: A scalar penalty value that quantifies how much of the image 
                          exists outside the support region.
        """
        amp = torch.abs(image)
        unmasked_amp = torch.ones_like(amp)
        unmasked_amp[self.support > 0.5 ] = 0
        unmasked_amp *= amp
        total_penalty = torch.abs(torch.sum(unmasked_amp)/torch.sum(amp)) + 1e-6

        return total_penalty
        
    
    def set_sw_support(self, flag, steps, sigma, threshold, phase_pen = 0, init_support = None, zeta = 1):
        """Sets the support mask and parameters for the shrinkwrap algorithm.

        This method initializes or updates the support mask used in the phase retrieval 
        process. If `init_support` is provided, it is used as the initial support; 
        otherwise, a default support mask of ones is created. It also configures the 
        shrinkwrap parameters for adaptive support updates.
    
        Args:
            flag (bool): Whether shrinkwrap is enabled (`True`) or disabled (`False`).
            steps (int): Number of shrinkwrap update steps during the reconstruction.
            sigma (float): Standard deviation for the Gaussian smoothing in shrinkwrap.
            threshold (float): Threshold value for updating the support mask.
            init_support (np.ndarray, torch.Tensor, or None, optional): 
                The initial support mask. If `None`, a mask of ones is created. 
                Defaults to `None`.
            phase_pen (boolean, optional): Whether to penalise phases outside the support region. Default to 'False'
            zeta (float, optional): A scaling factor for the support mask. Defaults to `1`.
        """
        self.phase_pen = phase_pen
        if init_support == None:
            shp = self.image_dims[0]*self.band_multiplier,self.image_dims[1]*self.band_multiplier 
            self.support = torch.ones(shp, requires_grad=False, device = self.device)
        elif isinstance(init_support, np.ndarray):
            self.support = torch.tensor(init_support, dtype=torch.float64, device = self.device)
        elif isinstance(init_support, torch.Tensor):
            self.support = init_support
        else:
            raise ValueError("Passed in support must be a np.ndarray or torch.Tensor. Else None, and it will be initialised to array of ones.")
            
        self.sw_flag = flag
        self.sw_steps = steps
        self.sw_sigma = sigma
        self.sw_threshold = threshold
        self.zeta_init = zeta
        self.shrinkwrap = lambda data: ShrinkWrap( data=data,
                                                    sigma=sigma, 
                                                    threshold=threshold, 
                                                    kernel_size=3, 
                                                  device=self.device)
    def _update_support(self):
        """
        Private method for updating the shrink_wrap support. 

        The function first extract the current spectrum amplitude and phase, then computes the object by performing the fft. 
        The support is then update using the ShrinkWrap class. 
        This method gets called in update_sw_support()
        """
        support = self.shrinkwrap(torch.abs(self.recon_obj_tensor))
        self.support = support.get()
        
        
    def update_sw_support(self, epoch):
        """
        Update shrink wrap support based on the current epoch.
    
        This method updates the shrink wrap parameter based on the current epoch. Initially, alpha is set to an array of ones 
        for the first n epochs. After that, and every `sw_steps` epochs, it is updated based on the last object estiamte.

        Args:
            epoch (int): Current epoch number.
        """            
        if self.sw_steps is None:
            return
            
        if epoch == self.sw_flag:
            self._update_support()
            self.last_sw_update = epoch
            
        elif (epoch > self.sw_flag) and (
            (epoch - self.last_sw_update) >= self.sw_steps
        ):
            self._update_support()
            self.last_sw_update = epoch
        
    #########################################################################################
    ################################## Learning Parameters ##################################
    #########################################################################################
   
    ################################# Main Loop #################################
    def iterate(self, epochs, optim_flag = 5, live_flag = None, n_jobs = -1):
        """
        Iterate through the optimization process for a specified number of epochs.

        This method runs the training loop, optimizing the spectrum and pupil with alternating optimizers, 
        and performing backpropagation to minimize the loss. If the `live_flag` is specified, it updates the 
        loss plot in real-time.

        Args:
            epochs (int): The number of epochs for training.
            optim_flag (int, optional): The frequency of switching between optimizers. Default is 5.
            live_flag (int, optional): The frequency of updating the live loss plot. If None, the plot is not updated. Default is None.
            n_jobs (int, optional): The number of parallel jobs for processing. Default is -1 (all available cores).

        """
        self.num_epochs += epochs
        self.optim_flag = optim_flag

        # Create separate optimizers for spectrum and pupil
        current_optimizer = self.spectrum_optimiser
        current_schedular = self.spectrum_schedular
        
        epochs_since_switch = 0  # Counter to track epochs since last switch

        if live_flag is not None:
            #fig, ax, line_mse, line_tv, line_supp = self._init_live_plot()
            fig, ax, line_mse, line_tv = self._init_live_plot()
        if self.debug:
            epochs = 2
            
        for epoch in tqdm(range(epochs), desc="Processing", total=epochs, unit="Epochs"):
            
            current_optimizer.zero_grad()
            #self.epoch_loss = 0
            self.mse_loss = 0
            self.tv_loss = 0
            self.supp_loss = 0

            # Update Support 
            self.update_sw_support(self.epochs_passed)

            for i, (image, kx_iter, ky_iter) in enumerate(zip(self.images, self.kin_vec[:, 0], self.kin_vec[:, 1])):    
                # Forward Pass
                self._process_image(image, kx_iter, ky_iter)

        
            # Compute gradients
            #self._compute_gradients()
            #self._compute_adaptive_gradients()
            self.mse_loss.backward()
            
            # Update Parameters
            current_optimizer.step()    
            current_schedular.step()
            
            if self.debug:
                self._print_debugs()
            if self.verbose:
                self._print_verbose()
            
            # update switch counter
            if self.pupil_optimiser is not None:
                epochs_since_switch += 1
                if epochs_since_switch >= optim_flag:
                    if current_optimizer == self.spectrum_optimiser:
                        current_optimizer = self.pupil_optimiser
                        current_schedular = self.pupil_schedular
                    else:
                        current_optimizer = self.spectrum_optimiser
                        current_schedular = self.spectrum_schedular
                    epochs_since_switch = 0  # Reset the counter
            
            
            # Update loss list
            self.tv_losses.append(self.tv_loss.cpu().detach().numpy())
            self.supp_losses.append(self.supp_loss.cpu().detach().numpy())
            self.main_losses.append(self.mse_loss.cpu().detach().numpy())
        
            self.epochs_passed +=1
            
            # Updating live plot
            if live_flag is not None and epoch % live_flag == 0:
                #self._update_live_loss(fig,ax,line_mse,line_tv,line_supp,epoch)
                self._update_live_loss(fig,ax,line_mse,line_tv,epoch)
                    
        self.post_process()

    ################################# Helper function #################################
    def _compute_grad_norm(self,grad_list):
        """Compute the total L2 norm of a list of gradients (ignoring None)."""
        non_none_grads = [g for g in grad_list if g is not None]
        if not non_none_grads:
            return 0.0
        return torch.norm(torch.stack([torch.norm(g) for g in non_none_grads]))
        
    def _normalize_gradients(self,gradients):
        """Normalize gradients to have unit L2 norm (to balance their contributions)."""
        
        non_none_grads = [g for g in gradients if g is not None]
        if not non_none_grads:
            return gradients

        
        # Compute the total norm of all gradients (treat as one vector)
        total_norm = torch.sqrt(sum(torch.norm(g, p=2) ** 2 for g in non_none_grads))
        
        # Normalize each gradient (avoid division by zero)
        eps = 1e-8
        normalized_grads = [g / (total_norm + eps) if g is not None else None 
                            for g in gradients]

        if self.debug:
            print(f"Gradient list length before norm: {len(gradients)}, after: {len(normalized_grads)}")
            
        return normalized_grads

    def _compute_gradients(self):
        # Compute independent gradients 
        self.tv_grad = torch.autograd.grad(self.tv_loss, self.model.parameters(), retain_graph=True, allow_unused=True)
        self.mse_grad = torch.autograd.grad(self.mse_loss, self.model.parameters(), retain_graph = True) 
        self.supp_grad = torch.autograd.grad(self.supp_loss, self.model.parameters(), retain_graph = True, allow_unused=True) 
        
        self.tv_grad = self._normalize_gradients(self.tv_grad)
        self.mse_grad = self._normalize_gradients(self.mse_grad)
        self.supp_grad = self._normalize_gradients(self.supp_grad)
          
        # Compute the mean of gradients
        combined_grad = []
        
        # Loop over gradients and compute combined gradient
        for g_mse, g_tv in zip(self.mse_grad, self.tv_grad):
            if g_mse is None and g_tv is None:
                combined_grad.append(None)
                continue
            if g_mse is None:
                g = g_tv
            elif g_tv is None:
                g = g_mse 
            else:
                g = (g_mse + g_tv)/2
                
            combined_grad.append(g)

        self.combined_grads = combined_grad
        
        for param, grad in zip(self.model.parameters(), self.combined_grads):
            param.grad = grad
            
    def _compute_adaptive_gradients(self):
        
        eps = 1e-8
        
        # Adaptive TV Weight
        alpha_tv = self.mse_loss.item() / (self.tv_loss.item() + eps)
        alpha_tv = max(self.alpha_min, min(alpha_tv, self.alpha_max))  # Clamp alpha_tv

        # Compute independent gradients 
        self.tv_grad = torch.autograd.grad(self.tv_loss, self.model.parameters(), retain_graph=True, allow_unused=True)
        self.mse_grad = torch.autograd.grad(self.mse_loss, self.model.parameters(), retain_graph = True) 
        
        # Normalise gradients 
        #self.tv_grad = self._normalize_gradients(self.tv_grad)
        #self.mse_grad = self._normalize_gradients(self.mse_grad)
        
        combined_grad = []
        
        # Loop over gradients and compute combined gradient
        for g_mse, g_tv in zip(self.mse_grad, self.tv_grad):

            if g_mse is None and g_tv is None:
                combined_grad.append(None)
                continue

            if g_mse is None:
                g = alpha_tv * g_tv
            elif g_tv is None:
                g = g_mse 
            else:
                g = g_mse + alpha_tv * g_tv

            combined_grad.append(g)

        self.combined_grads = combined_grad
        
        for param, grad in zip(self.model.parameters(), self.combined_grads):
            param.grad = grad
        
    def _get_current_object_image(self):
        """
        Private method for updating the reconstructed object. 
        """
        spectrum_amp = self.model.spectrum_amp  
        spectrum_pha = self.model.spectrum_pha
        
        self.recon_spectrum = spectrum_amp * torch.exp(1j * spectrum_pha)
        self.recon_obj_tensor = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(self.recon_spectrum)))
    
    def _process_image(self, image, kx_iter, ky_iter):        
        """
        Performs a Fourier domain update on the spectrum using the given image and 
        the current kx, ky values.

        This method calculates the Fourier domain bounds for the given image, 
        performs the reconstruction using the model, computes the loss, and applies 
        regularization to the reconstruction.

        Args:
            image (torch.Tensor): The input image to be used for updating the spectrum.
            kx_iter (float): The kx value used for determining the bounds in the Fourier domain.
            ky_iter (float): The ky value used for determining the bounds in the Fourier domain.

        """
        kx_cidx = round((kx_iter - self.kx_min_n) / self.dkx)
        kx_lidx = round(max(kx_cidx - self.omega_obj_x / (2 * self.dkx) * self.band_multiplier, 0))
        kx_hidx = round(kx_cidx + self.omega_obj_x / (2 * self.dkx) *self.band_multiplier) + (1 if self.nx_lr % 2 != 0 else 0)

        ky_cidx = round((ky_iter - self.ky_min_n) / self.dky)
        ky_lidx = round(max(ky_cidx - self.omega_obj_y / (2 * self.dky) *self.band_multiplier, 0))
        ky_hidx = round(ky_cidx + self.omega_obj_y / (2 * self.dky) *self.band_multiplier) + (1 if self.ny_lr % 2 != 0 else 0)
        
        bounds = [[kx_lidx, kx_hidx], [ky_lidx, ky_hidx]]
        low_resolution_image = self.model(bounds)
        self._get_current_object_image()
        
        tv_reg = self.tv_regularization(self.recon_obj_tensor)
        supp_loss = self.support_penalty(self.recon_obj_tensor)

        # Amplitude Based
        # loss = self.loss_fn(torch.abs(low_resolution_image), torch.sqrt(torch.abs(image)))

        # Intensity Based
        loss = self.loss_fn(torch.abs(low_resolution_image)**2, torch.abs(image))
        
        self.mse_loss += self.beta * loss
        self.tv_loss += tv_reg
        self.supp_loss += supp_loss            

        if self.sec_loss_fn is not None:
            sec_loss = self.sec_loss_fn(torch.abs(low_resolution_image), torch.sqrt(torch.abs(image)))
            self.mse_loss += self.delta * sec_loss

        if torch.isnan(loss):
            raise ValueError("There is a Nan value, check the configurations ")
            
        #self.epoch_loss += (self.beta * loss + self.alpha * tv_reg + self.zeta * supp_loss) /(self.beta + self.alpha + self.zeta) # Accumulate loss
        
            
    def post_process(self):
        """
        Performs post-processing to extract and compute the final results after training.

        This method detaches the computed spectra and pupil functions from the computation graph, 
        reconstructs the spectrum, object, and pupil functions, and calculates the corresponding 
        contrast transfer function (CTF).

        """
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

        self.final_support = np.abs(self.support.cpu().numpy())
        
    ######################################################################################################
    ################################## Plotting and Saving ###############################################
    ######################################################################################################
    def _print_debugs(self):
        are_equal = all( torch.allclose(g1, g2, rtol=1e-5, atol=1e-8) 
                        if (g1 is not None and g2 is not None) 
                        else (g1 is None and g2 is None)
                        for g1, g2 in zip(self.combined_grads, self.mse_grad))
                    
        print(f"Are combined_grads == mse_grad? {are_equal}")
        print(f"mse_loss requires grad = {self.mse_loss.requires_grad}") 
        print(f"mse_loss computational graph = {self.mse_loss.grad_fn}")
        for (name, param), g1, g2, g3 in zip(self.model.named_parameters(), self.mse_grad, self.tv_grad, self.supp_grad):
            print(f"Param {name} = {param.requires_grad, param.grad.norm()}")
            print(f"Param {name}: MSE Grad - {g1 is not None}, TV Grad - {g2 is not None}, Supp Grad - {g3 is not None}")   
            if g1 is not None:
                
                print(f"Param {name}: MSE Grad - {g1.norm()}")
            if g2 is not None:
                
                print(f"Param {name}: TV Grad - {g2.norm()}")
            if g3 is not None:
                
                print(f"Param {name}: Supp Grad - {g3.norm()}")
            
        for name, param in self.model.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}")
            
            if not param.requires_grad:
                print(f"{name} is FROZEN (requires_grad=False)!")
            if param.grad is None:
                self.grad_norms[name].append(param.grad.norm().item().cpu().detach().numpy())
                print(f"{name} has NO gradient!")
            elif torch.all(param.grad == 0):
                print(f"{name} has ZERO gradient!")
    
            if param.grad is not None:
                print(f"Parameter: {name} | Shape: {param.shape} | Grad Norm: {param.grad.norm()}")
    
    def _print_verbose(self):
        # Compute norms
        self.mse_norm = self._compute_grad_norm(self.mse_grad)
        self.tv_norm = self._compute_grad_norm(self.tv_grad)
        self.supp_norm = self._compute_grad_norm(self.supp_grad)
        self.combined_norm = self._compute_grad_norm(self.combined_grads)

    def _init_live_plot(self):
        # plotting live loss
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Live Loss Plot")
        
        line_mse, = ax.plot([],[])
        #line_supp, = ax.plot([],[])
        line_tv, = ax.plot([],[])
        
        line_mse.set_label('MSE')
        #line_supp.set_label('Support')
        line_tv.set_label('TV')
        
        ax.legend()
        plt.tight_layout()
        plt.ion()
        plt.show()

        return fig, ax, line_mse, line_tv#, line_supp
        
    def _update_live_loss(self, fig, ax,line_mse,line_tv, epoch):
        """
        Update the live loss plot during training.

        This method updates the live loss plot by setting the x and y data for the line plot, 
        and then refreshing the plot to reflect the new loss values.

        Args:
            fig (matplotlib.figure.Figure): The figure object for the live plot.
            ax (matplotlib.axes.Axes): The axes object for the plot.
            line (matplotlib.lines.Line2D): The line object representing the loss curve.
            epoch (int): The current epoch number (used to update the plot).

        """
        
        line_mse.set_xdata(range(self.epochs_passed))
        line_mse.set_ydata(self.main_losses)

        line_tv.set_xdata(range(self.epochs_passed))
        line_tv.set_ydata(self.tv_losses)
        #line_tv.set_ydata(self.main_losses)
        
        #line_supp.set_xdata(range(self.epochs_passed))
        #line_supp.set_ydata(self.supp_losses)
        
        ax.set_title(f"Epoch = {self.epochs_passed-1}, MSE Loss = {self.mse_loss:.2f}, TV Loss = {self.tv_loss:.2f}")
        ax.set_yscale('log')
        ax.relim()
        ax.autoscale_view()
        
        if self.verbose:
            total = self.mse_norm + self.tv_norm + self.supp_norm
            text = (f'Gradient Contributions:\n'
            f'{"MSE:":<15} {(self.mse_norm/total).item()*100:>5.1f}%\n'
            f'{"TV:":<15} {(self.tv_norm/total).item()*100:>5.1f}%\n'
            f'{"Support:":<15} {(self.supp_norm/total).item()*100:>5.1f}%')
        
            props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
            
            ax.text(0.95, 0.8, text, 
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=props)
            
        clear_output(wait=True)
        display(fig)
            
        #fig.canvas.draw()
        fig.canvas.flush_events()
    
    def plot_final_support(self, title = "Support", cmap = "viridis"):
        """
        Plots the final shrink wrapped object support.
    
        Args:
            title (str): Title for the object support.
            cmap (str): Colormap for the object support.
        """
        image1 = self.final_support
    
        plot_roi_from_numpy(image1, title=title, cmap=cmap)
        
    def plot_rec_obj(self, 
                     vmin1= None, vmax1=None, 
                     vmin2= -np.pi, vmax2=np.pi, 
                     title1 = "Object Amplitude", title2 = "Object Phase", cmap1 = "viridis", cmap2 = "viridis"):
        """
        Plots the reconstructed object amplitude and phase side by side.
    
        Args:
            vmin1, vmax1 (float): Minimum and maximum values for the object amplitude plot.
            vmin2, vmax2 (float): Minimum and maximum values for the object phase plot.
            title1, title2 (str): Titles for the object amplitude and phase plots.
            cmap1, cmap2 (str): Colormap for the object amplitude and phase plots.
        """
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
        """
        Plots the reconstructed Fourier amplitude and phase side by side.
    
        Args:
            vmin1, vmax1 (float): Minimum and maximum values for the Fourier amplitude plot.
            vmin2, vmax2 (float): Minimum and maximum values for the Fourier phase plot.
            title1, title2 (str): Titles for the Fourier amplitude and phase plots.
            cmap1, cmap2 (str): Colormap for the Fourier amplitude and phase plots.

        """
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
        """
        Plots the pupil function amplitude and phase side by side.
    
        Args:
            vmin1, vmax1 (float): Minimum and maximum values for the pupil amplitude plot.
            vmin2, vmax2 (float): Minimum and maximum values for the pupil phase plot.
            title1, title2 (str): Titles for the pupil amplitude and phase plots.
            cmap1, cmap2 (str): Colormap for the pupil amplitude and phase plots.

        """
        image1 = np.abs(self.pupil_func)
        image2 = np.angle(self.pupil_func)
    
        plot_images_side_by_side(image1, image2, 
                                 vmin1= vmin1, vmax1=vmax1, 
                                 vmin2= vmin2, vmax2=vmax2, 
                                 title1=title1, title2=title2, cmap1=cmap1, cmap2=cmap2, figsize=(10, 5), show = True)

    def plot_ctf(self, 
                        vmin1= None, vmax1=None, 
                        title1 = "CTF", cmap1 = "viridis"):
        """
        Plots the contrast transfer function (CTF).
    
        Args:
            vmin1, vmax1 (float): Minimum and maximum values for the CTF plot.
            title1 (str): Title for the CTF plot.
            cmap1 (str): Colormap for the CTF plot.
        """
        plot_roi_from_numpy(self.ctf, name=title1, vmin1=vmin1, vmax1=vmax1)


    def plot_loss(self, log_scale = True):
        """
        Plots the loss over epochs.
        """
        plt.figure()
        plt.plot(self.tv_losses, label = 'TV')
        plt.plot(self.main_losses, label = 'MSE')
        #plt.plot(self.supp_losses, label = 'Support')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        if log_scale:
            plt.yscale("log")
        plt.legend()
        plt.title("Loss Metrics per Epoch")
        plt.show()
        
    def plot_grad_norms(self):
        """
        Plots the grad norms over epochs.
        """
        plt.figure(figsize=(10, 6))
        for name, norms in self.grad_norms.items():
            plt.plot(norms, label=name)
        plt.xlabel("Epoch")
        plt.ylabel("Gradient Norm")
        plt.title("Gradient Norms Over Training")
        plt.legend()
        plt.grid()
        plt.show()
    
    def save_reconsturction(self, file_path):
        """
        Save the reconstruction data and metadata to an HDF5 file.
    
        Args:
            file_path (str): The path where the HDF5 file will be saved.
        """
        # Prepare metadata
        metadata = {
            
        "Num_epochs": self.num_epochs,
        "Optim_flag": self.optim_flag,
        "upsample_factor": self.band_multiplier,
        "pupil_dims" : self.pupil_dims,
        "coherent_image_dims": self.image_dims,
        }
        optimiser_spectrum = {**self.spectrum_optimiser.param_groups[0]}
        if self.pupil_optimiser is not None:
            optimiser_pupil = {**self.pupil_optimiser.param_groups[0]}

        kbounds = {
        "bounds_kx": str(self.bounds_x),
        "bounds_ky": str(self.bounds_y),
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

            recon_group = h5f.create_group("Reconstructed Data")
            # Save reconstructed images 
            amp = np.abs(self.recon_obj)
            pha = np.angle(self.recon_obj)
            recon_group.create_dataset("object_amplitude", data=amp, compression="gzip")
            recon_group.create_dataset("object_phase", data=pha, compression="gzip")
            
            # Save spectrum 
            amp = np.abs(self.recon_spectrum)
            pha = np.angle(self.recon_spectrum)
            recon_group.create_dataset("Fourier_amplitude", data=amp, compression="gzip")
            recon_group.create_dataset("Fourier_phase", data=pha, compression="gzip")
            
            # Save reconstructed pupil function
            amp = np.abs(self.pupil_func)
            pha = np.angle(self.pupil_func)
            recon_group.create_dataset("Pupil_amplitude", data=amp, compression="gzip")
            recon_group.create_dataset("Pupil_phase", data=pha, compression="gzip")
    
            # Save loss values
            losses = h5f.create_group("Losses")
            losses.create_dataset("support_loss_values", data=np.array(self.supp_losses), compression="gzip")
            losses.create_dataset("tv_reg_values", data=np.array(self.tv_losses), compression="gzip")
            losses.create_dataset("main_loss_values", data=np.array(self.main_losses), compression="gzip")
        

def train_fourier_ptychography(model, target_images, num_epochs=500, lr=0.01):
    """
    Train the Fourier Ptychography network over multiple measurements.
    
    Args:
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
