import numpy as np 
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
from scipy.signal import convolve2d

#from ..data_fs import * #downsample_array, upsample_images, pad_to_double

from skimage.restoration import unwrap_phase
from .utils_zernike import *

from .phase_abstract import Plot, LivePlot, PhaseRetrievalBase

class difference_map(PhaseRetrievalBase, Plot, LivePlot):

    def __init__(self, backend = 'threading', **kwargs):

        super().__init__(**kwargs)
        
        # self.object_est = object_est   #has shape different from the pupil_function
        # self.object_estFT = object_estFT  
        

        self.Npupil_rows, self.Npupil_cols = self.pupil_func.shape
        self.Ncoherent_imgs, self.Nr1, self.Nc1 = self.images.shape

        self.backend = backend

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
        
        self.upsample_coherent_images()
        
        self._initiate_recons_images()
    
    def _upsample_coh_img(self, image, shape):
        
        image_FT = self.forward_fft(image)
        
        new_image_FT = pad_to_shape(image_FT, shape)
        
        upsamp_img = np.abs(self.inverse_fft(new_image_FT))/ (self.Nr1**2/self.Npupil_rows_up**2) #Attention!!

        return upsamp_img
    
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
        
        elif self.hr_obj_image is not None:
            self.hr_fourier_image = self.forward_fft(self.hr_obj_image)
            
        elif self.hr_fourier_image is not None:
            self.hr_obj_image = self.inverse_fft(self.hr_fourier_image)
            
        self.nx_lr, self.ny_lr = self.images[0].shape
        self.nx_hr, self.ny_hr = self.hr_fourier_image.shape
        
        self.hr_fourier_image = pad_to_shape(self.hr_fourier_image, self.pupil_func.shape)
        self.hr_obj_image = pad_to_shape(self.hr_obj_image, self.pupil_func.shape)
        
    def _load_pupil(self):
        
        dims = round((self.kx_max_n - self.kx_min_n)/self.dkx), round((self.ky_max_n - self.ky_min_n)/self.dky)
        
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
    
    
    def iterate(self, iterations, live_plot = True):

        
        # Initial Guess
        phi_0 = np.random.random(self.pupil_func.shape)
        #self.hr_obj_image = self.images[0]*np.exp(1j*0.02*phi_0)
            
        #self.hr_fourier_image = self.forward_fft(self.hr_obj_image)

        self._initialize_exit()

        if live_plot:
            fig, axes, img_amp, img_phase, fourier_amp, pupil_phase, loss_im = self._initialize_live_plot()
            self.hr_obj_image = self.inverse_fft(self.hr_fourier_image)
            self._update_live_plot(img_amp, img_phase, fourier_amp, pupil_phase, loss_im, fig, self.iters_passed, axes)
        
        for it in range(iterations):
            print(f"iteration {it+1}")
            
            prev_objectFT = self.hr_obj_image
            prev_PSI = self.PSI_FT_centred 
                        
            self.hr_fourier_image, pupilss = self._objectFT_update(self.pupil_func, self.PSI_FT_centred )
            
            self.PSI_FT_centred, this_R = self.difference_map_engine(self.PSI_FT_centred, self.hr_fourier_image, self.pupil_func, self.coherent_imgs_upsampled)
            
            self.hr_obj_image = self.inverse_fft(self.hr_fourier_image )
            # update pupil
            self.pupil_func = self._pupil_update()
            
            if live_plot:
                self.hr_obj_image = self.inverse_fft(self.hr_fourier_image)
                self._update_live_plot(img_amp, img_phase, fourier_amp, pupil_phase, loss_im, fig, self.iters_passed, axes)
            

    #### DIFFERENCE MAP Engine 
    def _process_single_image(self, i, image, kx_iter, ky_iter, PSI_i, objectFT, pupil):
        
        this_pupil = self._get_pupil_patch_centred(kx_iter, ky_iter, pupil)
        
        Ps_psi = self._get_exitFT(this_pupil, objectFT)
        
        Rs_psi = 2 * Ps_psi - PSI_i
        
        Pm_rpsi = self.project_data(image, Rs_psi)
        
        PSI_n_i = PSI_i + Pm_rpsi - Ps_psi
        
        err_im = self.compute_error(image, Ps_psi)
        
        return PSI_n_i, err_im


    def difference_map_engine(self, PSI, objectFT, pupil, images, n_jobs=-1):
        
        results = Parallel(n_jobs=n_jobs, backend = self.backend)(
            delayed(self._process_single_image)(
                i, image, kx_iter, ky_iter, PSI[i], objectFT, pupil
            )
            for i, (image, kx_iter, ky_iter) in enumerate(zip(images, self.ks[:, 0], self.ks[:, 1]))
        )
        
        PSI_n = np.array([r[0] for r in results])
        err_tot = sum(r[1] for r in results)
        
        return PSI_n, err_tot
        
    
    
    ######### INITIALISE EXIT ########## 
    
    def _get_exitFT(self, pupil, objectFT):
        return pupil * objectFT
    
    def _compute_single_exit(self, i, kx_iter, ky_iter, pupil, objectFT):
        """Compute exit wave for a single k-vector"""
        this_pupil = self._get_pupil_patch_centred(kx_iter, ky_iter, pupil)
        this_PSI = self._get_exitFT(this_pupil, objectFT)
        return this_PSI

    def _initialize_exit(self):
        '''
        exit initialization where the pupil function and the object spectrum
        are at the centre
        '''
        
        exit_FT_centred = Parallel(n_jobs=self.num_jobs, backend = self.backend)(
            delayed(self._compute_single_exit)(i, kx_iter, ky_iter, self.pupil_func, self.hr_fourier_image)
            for i, (kx_iter, ky_iter) in enumerate(self.ks)
        )
        
        return np.array(exit_FT_centred)

    ######## PUPIL UPDATE ##########
    def _process_single_pupil_update(self, i, kx_iter, ky_iter, objectFT, PSI_i):
        """Process a single k_s iteration for pupil update"""
        this_objectFT = self._shift_signalFT(kx_iter, ky_iter, objectFT)
        
        denom_contrib = np.abs(this_objectFT)**2
        
        this_PSI = self._shift_signalFT(kx_iter, ky_iter, PSI_i)
        
        numer_contrib = np.conjugate(this_objectFT) * this_PSI
        
        return this_PSI, denom_contrib, numer_contrib
    
    def _pupil_update(self, objectFT, PSI, pupil_func_abs, n_jobs=-1):
        '''
        Updating the pupil function
        '''        
        results = Parallel(n_jobs=self.num_jobs)(
            delayed(self._process_single_pupil_update, backend = self.backend)(
                i, kx_iter, ky_iter, objectFT, PSI[i]
            )
            for i, (kx_iter, ky_iter) in enumerate(self.ks)
        )
        
        # Unpack results
        PSI_FT_shifted = np.array([r[0] for r in results])
        denom_contribs = [r[1] for r in results]
        numer_contribs = [r[2] for r in results]
        
        # Sum contributions
        denom = np.sum(denom_contribs, axis=0)
        numer = np.sum(numer_contribs, axis=0)
        
        pupil_func_update = numer / (denom + 1e-15)
        pupil_func_update = pupil_func_update * pupil_func_abs
        
        return pupil_func_update, PSI_FT_shifted

    
    ############## OBJECT UPDATE ############
    
    def _process_single_ks(self, i, kx_iter, ky_iter, pupil, PSI_FT_centred):
        """Process a single k_s iteration"""
        this_pupil = self._get_pupil_patch_centred(kx_iter, ky_iter, pupil)
        
        denom_contrib = np.abs(this_pupil)**2
        
        this_PSI = PSI_FT_centred[i]
        numer_contrib = np.conjugate(this_pupil) * this_PSI
        
        return this_pupil, denom_contrib, numer_contrib


    def _objectFT_update(self, pupil, PSI_FT_centred, n_jobs=-1):
        
        print(f"Object FT updates")
        
        # Parallel computation
        results = Parallel(n_jobs=self.num_jobs, backend = self.backend)(
            delayed(self._process_single_ks)(i, kx_iter, ky_iter, pupil, PSI_FT_centred)
            for i, (kx_iter, ky_iter) in enumerate(self.ks)
        )
        
        # Unpack results
        pupil_patches = [r[0] for r in results]
        denom_contribs = [r[1] for r in results]
        numer_contribs = [r[2] for r in results]
        
        # Sum contributions
        denom = np.sum(denom_contribs, axis=0)
        numer = np.sum(numer_contribs, axis=0)
        
        objectFT_update = numer / (denom + 1e-15)
        
        return objectFT_update, pupil_patches
    
    
    ###### Helpers
        

    def _shift_objectFT(self, kx_iter, ky_iter):
        '''
        This method shifts the object's FT estimate to the (kx_cidx, ky_cidx) location
        '''
        object_patch = np.zeros_like(self.pupil_func)
        
        kx_cidx = round((kx_iter - self.kx_min_n) / self.dkx)
        kx_lidx = round(max(kx_cidx - self.omega_obj_x / (2 * self.dkx), 0))
        kx_hidx = round(kx_cidx + self.omega_obj_x / (2 * self.dkx)) + (1 if self.nx_lr % 2 != 0 else 0)
        
        ky_cidx = round((ky_iter - self.ky_min_n) / self.dky)
        ky_lidx = round(max(ky_cidx - self.omega_obj_y / (2 * self.dky), 0))
        ky_hidx = round(ky_cidx + self.omega_obj_y / (2 * self.dky)) + (1 if self.ny_lr % 2 != 0 else 0)

        rl = self.pupil_func.shape[0]//2 - self.Nr1//2
        rh = self.pupil_func.shape[0]//2 + self.Nr1//2
        cl = self.pupil_func.shape[1]//2 - self.Nc1//2
        ch = self.pupil_func.shape[1]//2 + self.Nc1//2

        object_patch[kx_lidx:kx_hidx, ky_lidx:ky_hidx] = self.hr_fourier_image[rl:rh, cl:ch]  
        return object_patch

    def _get_pupil_patch_centred(self, kx_iter, ky_iter, pupil):
        '''
        This method get Pupil patch to the centre
        '''
        pupil_func_patch = np.zeros_like(self.pupil_func).astype(complex) 
        
        kx_cidx = round((kx_iter - self.kx_min_n) / self.dkx)
        kx_lidx = round(max(kx_cidx - self.omega_obj_x / (2* self.dkx), 0))
        kx_hidx = round(kx_cidx + self.omega_obj_x / ( 2*self.dkx)) 
        
        ky_cidx = round((ky_iter - self.ky_min_n) / self.dky)
        ky_lidx = round(max(ky_cidx - self.omega_obj_y / (2* self.dky), 0))
        ky_hidx = round(ky_cidx + self.omega_obj_y / ( 2*self.dky))
        
        
        rl = self.pupil_func.shape[0]//2 - self.Nr1//2
        rh = self.pupil_func.shape[0]//2 + self.Nr1//2
        cl = self.pupil_func.shape[1]//2 - self.Nc1//2
        ch = self.pupil_func.shape[1]//2 + self.Nc1//2

        pupil_func_patch[rl : rh, cl:ch] = pupil[ kx_lidx : kx_hidx , ky_lidx : ky_hidx ]

        return pupil_func_patch

    def _shift_signalFT(self, kx_iter, ky_iter, objectFT):
        '''
        This method shifts the object's FT estimate to the (kx_cidx, ky_cidx) location
        '''
        object_patch = np.zeros_like(objectFT)
        
        kx_cidx = round((kx_iter - self.kx_min_n) / self.dkx)
        kx_lidx = round(max(kx_cidx - self.omega_obj_x / ( 2*self.dkx), 0))
        kx_hidx = round(kx_cidx + self.omega_obj_x / ( 2*self.dkx)) + (1 if self.nx_lr % 2 != 0 else 0)
        
        ky_cidx = round((ky_iter - self.ky_min_n) / self.dky)
        ky_lidx = round(max(ky_cidx - self.omega_obj_y / ( 2*self.dky), 0))
        ky_hidx = round(ky_cidx + self.omega_obj_y / (2 * self.dky)) + (1 if self.ny_lr % 2 != 0 else 0)

        rl = self.Npupil_rows_up//2 - self.Nr1//2
        rh = self.Npupil_rows_up//2 + self.Nr1//2 
        cl = self.Npupil_cols_up//2 - self.Nc1//2
        ch = self.Npupil_cols_up//2 + self.Nc1//2 

        object_patch[kx_lidx:kx_hidx, ky_lidx:ky_hidx] = objectFT[rl:rh, cl:ch]  
        
        return object_patch
    
    def compute_error(self, image, PSI):

        image_cal = np.abs( self.inverse_fft(PSI) )
        
        err_image = np.sum ( np.abs( image_cal - image ) )/ np.sum(image)

        return err_image

        

    def project_data( self, image, arr_FT):
        '''
        measurement projection
        '''
        psi =  self.inverse_fft(arr_FT) 
        
        psi_new = np.sqrt(image) * np.exp(1j*np.angle(psi))
        
        PSI_PM  =  self.forward_fft(psi_new)  
        
        return PSI_PM



    
            

        
        

        







    