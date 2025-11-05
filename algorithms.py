import numpy as np
from joblib import Parallel, delayed


fft_images = lambda imgs: np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(imgs, axes=(-2, -1)), axes=(-2, -1)), axes = (-2,-1))
ifft_images = lambda imgs: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(imgs, axes=(-2, -1)), axes=(-2, -1)), axes = (-2,-1))

    

class AlgorithmKernel:
    """Base kernel interface for reciprocal-space phase retrieval algorithms.

    Methods operate on arrays kept in the Runner.
    Kernels should be stateless (store only hyperparameters).
    """
    
    def step(self, bounds, objectFT, pupil_func, PSI, images, n_jobs, backend):    
        raise NotImplementedError
    
    ####### Object Updates ###########
    
    def _update_object_patch(self, slices, pupil_func, psi):
                
        this_pupil = self._extract_patch_to_center(slices, pupil_func)
        
        denom_contrib = np.abs(this_pupil)**2
        
        numer_contrib = np.conjugate(this_pupil) * psi
        
        return denom_contrib, numer_contrib
    
    
    # def update_object_ft(self, PSI: np.ndarray, pupil: np.ndarray, bounds: np.ndarray, n_jobs, backend):

    #     results = Parallel(n_jobs=n_jobs, backend = backend)(
    #         delayed(self._update_object_patch)(bound, pupil, psi_i)
    #         for bound, psi_i in zip(bounds, PSI)
    #     )
        
    #     denom_contribs = [r[0] for r in results]
    #     numer_contribs = [r[1] for r in results]
        
    #     # Sum contributions
    #     denom = np.sum(denom_contribs, axis=0)
    #     numer = np.sum(numer_contribs, axis=0)
        
    #     objectFT_update = numer / (denom + 1e-8)
        
    #     return objectFT_update

    def update_object_ft(self, PSI, pupil, slices):
        """
        Vectorized object update with precomputed slices.
        """
        patch_shape = PSI[0].shape
        denom = np.zeros(patch_shape, dtype=np.float64)
        numer = np.zeros(patch_shape, dtype=np.complex128)
    
        for sl, psi_i in zip(slices, PSI):
            pupil_patch = pupil[sl]
            denom += np.abs(pupil_patch)**2
            numer += np.conjugate(pupil_patch) * psi_i
    
        return numer / (denom + 1e-8)
    
    ############# Pupil Update ###############
    
    def _process_single_pupil_update(self, bounds, objectFT, psi, pupil_shape):
        """Process a single k_s iteration for pupil update"""
    
        
        this_objectFT = self._insert_center_to_kvec_location(bounds, objectFT, pupil_shape)
        # this_objectFT = objectFT
        denom_contrib = np.abs(this_objectFT)**2
        
        this_PSI = self._insert_center_to_kvec_location(bounds, psi, pupil_shape)
        # this_PSI = psi
        
        numer_contrib = np.conjugate(this_objectFT) * this_PSI 
        
        return denom_contrib, numer_contrib
    
    # def update_pupil(self, PSI, objectFT, bounds, pupil_func, ctf, beta = 0.9, n_jobs=1, backend='threading'):
    #     '''
    #     Updating the pupil function
    #     '''        
    #     pupil_shape = pupil_func.shape
    #     results = Parallel(n_jobs=n_jobs, backend = backend)(
    #         delayed(self._process_single_pupil_update)(
    #             k_vector, objectFT, psi_i, pupil_shape
    #         )
    #         for k_vector, psi_i in zip(bounds, PSI)
    #     )
        
    #     denom_contribs = np.array([r[0] for r in results])
    #     numer_contribs = np.array([r[1] for r in results])
        
    #     # Sum contributions
    #     denom = np.sum(denom_contribs, axis=0)
    #     numer = np.sum(numer_contribs, axis=0)
        
    #     pupil_func_update = numer / (denom + 1e-8)

    #     # pupil_func_update = (1 - beta) * pupil_func + beta * pupil_func_update

    #     pha = np.angle(pupil_func_update) * np.abs(ctf)
    #     amp = np.abs(pupil_func_update) * np.abs(ctf)
        
    #     pupil_func_update = amp * np.exp(1j*pha) 
        
    #     return pupil_func_update

    def update_pupil(self, PSI, objectFT, slices, pupil_func, ctf):
        
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
            
        pupil_func_update = numer / (denom + 1e-8)
        # pha = np.angle(pupil_func_update) * np.abs(ctf)
        # amp = np.abs(pupil_func_update) * np.abs(ctf)
        return pupil_func_update #amp * np.exp(1j * pha)

    
    ######## project data #########
    
    def project_data(self, images, Psi):
        '''
        measurement projection
        '''
        psi =  ifft_images(Psi) 
        
        psi_new = np.sqrt(images) * np.exp(1j*np.angle(psi))
        
        Psi_  =  fft_images(psi_new)  
        
        return Psi_
    
    ########## Helpers ##############

    def _insert_center_to_kvec_location(self, bounds, arr, pupil_shape):
        
        (lx, hx, ly, hy), (rl, rh, cl, ch) = bounds
        out = np.zeros(pupil_shape, dtype = complex)
        out[lx:hx, ly:hy] = arr
        # out = arr[rl:rh, cl:ch]
        return out
    
    def _extract_patch_to_center(self, sl, arr):
        
        # (lx, hx, ly, hy), (rl, rh, cl, ch) = bounds
        #out = np.zeros_like(arr)
        #out[rl:rh, cl:ch] = arr[lx:hx, ly:hy]
        out = arr[sl]
        return out
   
    def _compute_single_exit(self, sl, pupil, objectFT):
        """Compute exit wave for a single k-vector"""
        
        this_pupil = self._extract_patch_to_center(sl, pupil)
        psi = this_pupil * objectFT
        
        return psi

    def project_model(self, slices, pupil, objectFT, n_jobs, backened):
        '''
        exit initialization where the pupil function and the object spectrum
        are at the centre
        '''
        
        exit_FT_centred = Parallel(n_jobs=n_jobs, backend = backened)(
            delayed(self._compute_single_exit)(sl, pupil, objectFT)
            for sl in slices
        )
        
        return np.array(exit_FT_centred)
    
    def compute_error(self, old_psi, new_psi):
        
        err = np.linalg.norm(new_psi - old_psi) 
        return err
    

class DM(AlgorithmKernel):
    
    def step(self, bounds, objectFT, pupil_func, PSI, images, n_jobs, backend):
        
        Psi_model = self.project_model(bounds, pupil_func, objectFT, n_jobs, backend)
        
        Psi_reflection = 2 * Psi_model - PSI
        
        Psi_data_reflection = self.project_data(images, Psi_reflection)

        Psi_n = PSI + Psi_data_reflection - Psi_model
        
        return Psi_n



class RAAR(AlgorithmKernel):
    
    def __init__(self, beta = 0.9, beta_decay = None):

        self.beta = beta

        self.beta_decay = beta_decay
        
    
    def step(self, slices, objectFT, pupil_func, PSI, images, n_jobs, backend):


         
        Psi_reflection_model = 2*self.project_model(slices, pupil_func, objectFT, n_jobs, backend) - PSI
        
        Psi_project_model = self.project_data(images, Psi_reflection_model)
        

        Psi_n = self.beta * Psi_project_model + (1-self.beta) * PSI

        if self.beta_decay is not None:
            self.beta *= self.beta_decay
            
        return Psi_n
    


class AAR(AlgorithmKernel):
    
    def step(self, slices, objectFT, pupil_func, PSI, images, n_jobs, backend):
        
        Psi_model = self.project_model(slices, pupil_func, objectFT, n_jobs, backend)

        Psi_data_reflection = self.project_data(images, 2*Psi_model - PSI)

        Psi_n = 0.5 * (PSI + Psi_data_reflection)

        return Psi_n




class HPR(AlgorithmKernel):
    def __init__(self, beta = 0.9):

        self.beta = beta
    def step(self, slices, objectFT, pupil_func, PSI, images, n_jobs, backend):
        Psi_model = self.project_model(slices, pupil_func, objectFT, n_jobs, backend)

        Psi_data_reflection = self.project_data(images, 2*Psi_model - PSI)

        Psi_n = PSI + self.beta * (Psi_data_reflection - PSI)

        return Psi_n


class DM_PIE(AlgorithmKernel):
    def __init__(self, alpha=0.9):

        self.alpha = alpha

    def step(self, slices, objectFT, pupil_func, PSI, images, n_jobs, backend):
        Psi_model = self.project_model(slices, pupil_func, objectFT, n_jobs, backend)

        Psi_reflection = (1 + self.alpha) * Psi_model - self.alpha * PSI

        Psi_data_reflection = self.project_data(images, Psi_reflection)

        Psi_n = PSI + Psi_data_reflection - Psi_model

        return Psi_n

class ePIE(AlgorithmKernel):
    def __init__(self, alpha_obj=0.9):

        self.alpha_obj = alpha_obj

    def step(self, slices, objectFT, pupil_func, PSI, images, n_jobs, backend):
        
        Psi_model = self.project_model(slices, pupil_func, objectFT, n_jobs, backend)

        Psi_data = self.project_data(images, Psi_model)

        Psi_n = PSI + self.alpha_obj * (Psi_data - Psi_model)

        return Psi_n

    def compute_weight_fac(self, func):
        """Compute weighting factor for phase retrieval update."""
        
        mod = np.abs(func) ** 2
        return np.conjugate(func) / (mod.max() + 1e-23)
  
        
        
