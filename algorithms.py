import numpy as np
from joblib import Parallel, delayed
from .utils_pr import time_it
from .zernike import SquarePolynomials
import inspect

fft_images = lambda imgs: np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(imgs, axes=(-2, -1)), axes=(-2, -1)), axes = (-2,-1))
ifft_images = lambda imgs: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(imgs, axes=(-2, -1)), axes=(-2, -1)), axes = (-2,-1))

fft_2D = lambda img: np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))
ifft_2D = lambda img: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(img)))


class AlgorithmKernel:
    """Base kernel interface for reciprocal-space phase retrieval algorithms.

    Methods operate on arrays kept in the Runner.
    """
    
    def step_single(self, slices, objectFT, PSI, images, n_jobs, backend):    
        raise NotImplementedError
    
    @time_it
    def step(self, slices_list, objectFT_list, PSI_list, images_list, pupil_func, n_jobs, backend, **kwargs):    
        
        step_arguments = self.GetKwArgs(self.step_single, kwargs)
        
        n_streaks = len(slices_list)
        
        assert(len(objectFT_list) == len(PSI_list) == len(images_list) == n_streaks, 
               "slices_list, objectFT_list, PSI_list and images_list must have the same length (one per streak)")
        
        assert(type(objectFT_list) == type(PSI_list) == type(images_list) == type(slices_list) == list, 
               "slices_list, objectFT_list, PSI_list and images_list must have the same type (list)")

        if len(PSI_list) > 1:
            Psi_updated_list = Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(self.step_single)(slices, objFT, pupil_func, psi, img)
                for slices, objFT, psi, img in zip(slices_list, objectFT_list, PSI_list, images_list, **step_arguments)
            )
        else: 
            Psi_updated_list = []
            for slices, objFT, psi, img in zip(slices_list, objectFT_list, PSI_list, images_list):
                
                Psi_updated_list.append(self.step_single(slices, objFT, pupil_func, psi, img, **step_arguments))

        return Psi_updated_list

        
    # ~~~~___________ Objects Update ___________~~~~


    def update_object_ft_single(self, PSI, pupil, slices):
        "Update object ft for a single set of kins"
        patch_shape = PSI[0].shape
        denom = np.zeros(patch_shape, dtype=np.float64)
        numer = np.zeros(patch_shape, dtype=np.complex128)
    
        for sl, psi_i in zip(slices, PSI):
            pupil_patch = pupil[sl]
            denom += np.abs(pupil_patch)**2
            numer += np.conjugate(pupil_patch) * psi_i
        
        return numer / (denom + 1e-12)

    @time_it
    def update_object_fts(self, PSI_list, pupil, slices_list):
        "Update object ft for a multiple sets of kins"
        objectFT_updates = []
        
        for PSI, slices in zip(PSI_list, slices_list):

            objectFT_updates.append(self.update_object_ft_single(PSI, pupil, slices) )
                        
        return objectFT_updates
        
    # ~~~~___________ Pupil Update ___________~~~~ 
    @time_it
    def update_pupil(self, PSI_list, objectFT_list, slices_list, pupil_func, ctf):
        
        """Shared pupil update across all sets of kins."""
        
        pupil_shape = pupil_func.shape
        denom = np.zeros(pupil_shape, dtype=np.float64)
        numer = np.zeros(pupil_shape, dtype=np.complex128)

        for PSI, objectFT, slices in zip(PSI_list, objectFT_list, slices_list):

            # tmp_numer, tmp_denom = update_object_ft_single(PSI, objectFT, slices, pupil_func, ctf )
            for sl, psi_i in zip(slices, PSI):
                
                numer[sl] += np.conjugate(objectFT) * psi_i
                denom[sl] += np.abs(objectFT)**2
                
        
        pupil_func_update = numer / (denom + 1e-12)

        pha = np.angle(pupil_func_update) * np.abs(ctf)
        amp = np.abs(pupil_func_update) * np.abs(ctf)
        
        return  amp * np.exp(1j * pha) 

    def update_pupil_single(self, PSI, objectFT, slices, pupil_func, ctf):
        
        pupil_shape = pupil_func.shape
        denom = np.zeros(pupil_shape, dtype=np.float64)
        numer = np.zeros(pupil_shape, dtype=np.complex128)
    
        # Pre-allocate output buffers
        for sl, psi_i in zip(slices, PSI):
            # (lx, hx, ly, hy), (rl, rh, cl, ch) = bd
            this_objectFT = np.zeros(pupil_shape, dtype=np.complex128)
            this_PSI = np.zeros(pupil_shape, dtype=np.complex128)
            
            this_objectFT[sl] = objectFT
            this_PSI[sl] = psi_i
    
            denom[sl] += np.abs(this_objectFT[sl])**2
            numer[sl] += np.conjugate(this_objectFT[sl]) * this_PSI[sl] #* np.abs(ctf[lx:hx, ly:hy])
            
        pupil_func_update = numer / (denom + 1e-12)
        pha = np.angle(pupil_func_update) * np.abs(ctf)
        amp = np.abs(pupil_func_update) * np.abs(ctf)
        return amp * np.exp(1j * pha) #pupil_func_update #
        
    
    # ~~~~___________ Projections ___________~~~~
    
    
    def project_data(self, images, Psi):
        '''
        measurement projection
        '''
        psi =  ifft_images(Psi) 
        
        psi_new = np.sqrt(images) * np.exp(1j*np.angle(psi))
        
        Psi_  =  fft_images(psi_new)  
        
        return Psi_

   
    def _compute_single_exit(self, sl, pupil, objectFT):
        """Compute exit wave for a single k-vector"""
        
        psi = pupil[sl] * objectFT
        
        return psi

    def project_model(self, slices, pupil, objectFT):
        '''
        exit initialization where the pupil function and the object spectrum
        are at the centre
        this only works for a single set of kins.
        it is parallised for all sets in self.step. 
        '''

        Psi_model = [self._compute_single_exit(sl, pupil, objectFT) for sl in slices]
        
        return np.stack(Psi_model, axis=0)

    
    def project_Zernike(self, objectFT, pupil, slices, pupil_coords):
        
        amp = np.abs(pupil)
        pha = np.angle(pupil)
        
        zernike_pupil = amp * np.exp(
                        1j * SquarePolynomials.project_wavefront(pha, coords=pupil_coords)
                    )

        Psi_zernike = [self._compute_single_exit(sl, zernike_pupil, objectFT) for sl in slices]
        
        return np.stack(Psi_zernike, axis=0)
        
        
    def project_support(self, support, objectFT, pupil,slices):
        
        
        objectFTnew =  fft_2D(ifft_2D(objectFT) * support)

        Psi_supp = [self._compute_single_exit(sl, pupil, objectFTnew) for sl in slices]
        
        return Psi_supp
    
    # ~~~~___________ Helpers ___________~~~~

    def compute_error(self, old_psi_list, new_psi_list):

        errors = []
        
        for old_psi, new_psi in zip(old_psi_list, new_psi_list):
            err = np.linalg.norm(new_psi - old_psi) / np.linalg.norm(old_psi)
            errors.append(err)
            
        total_error = np.mean(errors)
        
        return total_error, errors
    
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


class DM(AlgorithmKernel):
    def __init__(self, beta = 1.0, beta_decay = None):

        self.beta = beta

        self.beta_decay = beta_decay
        
    def step_single(self, slices, objectFT, pupil_func, PSI, images):
        
        Psi_model = self.project_model(slices, pupil_func, objectFT)
        
        Psi_reflection = 2 * Psi_model - PSI
        
        Psi_data_reflection = self.project_data(images, Psi_reflection)

        Psi_n = PSI + self.beta * (Psi_data_reflection - Psi_model)
        if self.beta_decay is not None:
            self.beta *= self.beta_decay
        return Psi_n


        

class DNC(AlgorithmKernel):
    
    def __init__(self, beta = 0.9, weights: list = [1.0,1.0,1.0,1.0]):
        
        self.beta = beta 
        self.weights = weights 
        
    def step_single(self, slices, objectFT, PSI, pupil_func, images, support, pupil_coords):
        
        Reflection_model = 2 * self.project_model(slices, pupil_func, objectFT) - PSI
        Reflection_data = 2 * self.project_data(images, PSI) - PSI
        Reflection_zernike = 2 * self.project_Zernike(objectFT, pupil_func, slices, pupil_coords) - PSI
        Reflection_support = 2 * self.project_support(support, objectFT, pupil_func, slices) - PSI
        
        
        concur = self.weights[0] * Reflection_model + self.weights[1] * Reflection_data + self.weights[2] * Reflection_zernike + self.weights[3] * Reflection_support 
        
        Psi_n = PSI + self.beta( concur - PSI ) 
        
        return Psi_n

class RAAR(AlgorithmKernel):
    
    def __init__(self, beta = 0.9, beta_decay = None):

        self.beta = beta

        self.beta_decay = beta_decay
        
    
    def step_single(self, slices, objectFT, pupil_func, PSI, images):
         
        Psi_reflection_model = 2*self.project_model(slices, pupil_func, objectFT) - PSI
        
        Psi_project_model = self.project_data(images, Psi_reflection_model)

        Psi_n = self.beta * Psi_project_model + (1-self.beta) * PSI
            
        return Psi_n
    


class AAR(AlgorithmKernel):
    
    def step_single(self, slices, objectFT, pupil_func, PSI, images):
        
        Psi_model = self.project_model(slices, pupil_func, objectFT)

        Psi_data_reflection = self.project_data(images, 2*Psi_model - PSI)

        Psi_n = 0.5 * (PSI + Psi_data_reflection)

        return Psi_n




class HPR(AlgorithmKernel):
    def __init__(self, beta = 0.9):

        self.beta = beta
    def step_single(self, slices, objectFT, pupil_func, PSI, images):
        Psi_model = self.project_model(slices, pupil_func, objectFT)

        Psi_data_reflection = self.project_data(images, 2*Psi_model - PSI)

        Psi_n = PSI + self.beta * (Psi_data_reflection - PSI)

        return Psi_n


class DM_PIE(AlgorithmKernel):
    def __init__(self, alpha=0.9):

        self.alpha = alpha

    def step_single(self, slices, objectFT, pupil_func, PSI, images):
        Psi_model = self.project_model(slices, pupil_func, objectFT)

        Psi_reflection = (1 + self.alpha) * Psi_model - self.alpha * PSI

        Psi_data_reflection = self.project_data(images, Psi_reflection)

        Psi_n = PSI + Psi_data_reflection - Psi_model

        return Psi_n

class ePIE(AlgorithmKernel):
    def __init__(self, alpha_obj=0.9):

        self.alpha_obj = alpha_obj

    def step_single(self, slices, objectFT, pupil_func, PSI, images):
        
        Psi_model = self.project_model(slices, pupil_func, objectFT)

        Psi_data = self.project_data(images, Psi_model)

        Psi_n = PSI + self.alpha_obj * (Psi_data - Psi_model)

        return Psi_n

    def compute_weight_fac(self, func):
        """Compute weighting factor for phase retrieval update."""
        
        mod = np.abs(func) ** 2
        return np.conjugate(func) / (mod.max() + 1e-23)
  
        
        
