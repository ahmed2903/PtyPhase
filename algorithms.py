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
    
    # @time_it
    # def step(self, slices_list, objectFT_list, PSI_list, images_list, pupil_func, n_jobs, backend, **kwargs):    
        
    #     step_arguments = self.GetKwArgs(self.step_single, kwargs)
    #     n_streaks = len(slices_list)

    #     assert(len(objectFT_list) == len(PSI_list) == len(images_list) == n_streaks, 
    #            "slices_list, objectFT_list, PSI_list and images_list must have the same length (one per streak)")
        
    #     assert(type(objectFT_list) == type(PSI_list) == type(images_list) == type(slices_list) == list, 
    #            "slices_list, objectFT_list, PSI_list and images_list must have the same type (list)")

    #     if len(PSI_list) > 1:
    #         Psi_updated_list = Parallel(n_jobs=n_jobs, backend=backend)(
    #             delayed(self.step_single)(slices, objFT, pupil_func, psi, img, **step_arguments)
    #             for slices, objFT, psi, img in zip(slices_list, objectFT_list, PSI_list, images_list)
    #         )
    #     else: 
    #         Psi_updated_list = []
    #         for slices, objFT, psi, img in zip(slices_list, objectFT_list, PSI_list, images_list):
                
    #             Psi_updated_list.append(self.step_single(slices, objFT, pupil_func, psi, img, **step_arguments))

    #     return Psi_updated_list
    
        
    # ~~~~___________ Objects Update ___________~~~~


    def update_object_fts(self, PSI, pupil, pupil_slices, object_slices):
        "Update object ft for a single set of kins"
        x,y = PSI[0].shape
        n = max(object_slices)+1
        
        denom = np.zeros((n,x,y), dtype=np.float64)
        numer = np.zeros((n,x,y), dtype=np.complex128)
    
        for pup_sl, obj_sl, psi_i in zip(pupil_slices, object_slices, PSI):
            
            pupil_patch = pupil[pup_sl]
            
            denom[obj_sl] += np.abs(pupil_patch)**2
            
            numer[obj_sl] += np.conjugate(pupil_patch) * psi_i
        
        return numer / (denom + 1e-12)

    # @time_it
    # def update_object_fts(self, PSI_list, pupil, slices_list):
    #     "Update object ft for a multiple sets of kins"
    #     objectFT_updates = []
        
    #     for PSI, slices in zip(PSI_list, slices_list):

    #         objectFT_updates.append(self.update_object_ft_single(PSI, pupil, slices) )
                        
    #     return objectFT_updates
        
    # ~~~~___________ Pupil Update ___________~~~~ 
    # @time_it
    # def update_pupil(self, PSI_list, objectFT_list, slices_list, pupil_func, ctf):
        
    #     """Shared pupil update across all sets of kins."""
        
    #     pupil_shape = pupil_func.shape
    #     denom = np.zeros(pupil_shape, dtype=np.float64)
    #     numer = np.zeros(pupil_shape, dtype=np.complex128)

    #     for PSI, objectFT, slices in zip(PSI_list, objectFT_list, slices_list):

    #         # tmp_numer, tmp_denom = update_object_ft_single(PSI, objectFT, slices, pupil_func, ctf )
    #         for sl, psi_i in zip(slices, PSI):
                
    #             numer[sl] += np.conjugate(objectFT) * psi_i
    #             denom[sl] += np.abs(objectFT)**2
                
        
    #     pupil_func_update = numer / (denom + 1e-12)

    #     pha = np.angle(pupil_func_update) * np.abs(ctf)
    #     amp = np.abs(pupil_func_update) * np.abs(ctf)
        
    #     return  amp * np.exp(1j * pha) 
    @time_it
    def update_pupil(self, PSI, objectFT, pupil_slices, object_slices, pupil_func, ctf):
        
        pupil_shape = pupil_func.shape
        denom = np.zeros(pupil_shape, dtype=np.float64)
        numer = np.zeros(pupil_shape, dtype=np.complex128)
    
        # Pre-allocate output buffers
        for pup_sl, obj_sl, psi_i in zip(pupil_slices, object_slices, PSI):

    
            denom[pup_sl] += np.abs(objectFT[obj_sl])**2
            numer[pup_sl] += np.conjugate(objectFT[obj_sl]) * psi_i #* np.abs(ctf[lx:hx, ly:hy])
            
        pupil_func_update = numer / (denom + 1e-12)
        
        pha = np.angle(pupil_func_update) * np.abs(ctf)
        amp = np.abs(pupil_func_update) * np.abs(ctf)
        
        return amp * np.exp(1j * pha) #pupil_func_update #
        
    
    
    
    # ~~~~___________ Helpers ___________~~~~

    def compute_error(self, old_psi_list, new_psi_list):

        errors = []
        
        for old_psi, new_psi in zip(old_psi_list, new_psi_list):
            err = np.linalg.norm(new_psi - old_psi) / np.linalg.norm(old_psi)
            errors.append(err)
            
        total_error = np.mean(errors)
        
        return total_error, errors
    
    def GetKwArgs(self, obj, kwargs):
        obj_sigs = list(inspect.signature(obj).parameters.keys())
        obj_args = {k: v for k, v in kwargs.items() if k in obj_sigs}
        return obj_args


# ~~~~___________ Projections ___________~~~~
    
def project_data(images, Psi):
    '''
    measurement projection
    '''
    psi =  ifft_images(Psi) 
    
    psi_new = np.sqrt(images) * np.exp(1j*np.angle(psi))
    
    Psi_  =  fft_images(psi_new)  
    
    return Psi_


def project_model(pupil_slices, object_slices, pupil, objectFTs):
        
    object_stack = objectFTs[object_slices]
        
    # CHECK ME: the order of Xs and Ys
    Ys = np.array([np.arange(sl[0].start, sl[0].stop) for sl in pupil_slices])
    Xs = np.array([np.arange(sl[1].start, sl[1].stop) for sl in pupil_slices])
    YY = Ys[:, :, None]  
    XX = Xs[:, None, :]
    
    pupil_stack = pupil[YY, XX]
    
    Psi = pupil_stack * object_stack 
    
    return Psi 
    


def project_Zernike( pupil_slices, object_slices, pupil, objectFTs,  pupil_coords):
    
    amp = np.abs(pupil)
    pha = np.angle(pupil)
    
    zernike_pupil = amp * np.exp(
                    1j * SquarePolynomials.project_wavefront(pha, coords=pupil_coords)
                )
    
    object_stack = objectFTs[object_slices]
        
    # CHECK ME: the order of Xs and Ys
    Ys = np.array([np.arange(sl[0].start, sl[0].stop) for sl in pupil_slices])
    Xs = np.array([np.arange(sl[1].start, sl[1].stop) for sl in pupil_slices])
    YY = Ys[:, :, None]  
    XX = Xs[:, None, :]
    
    pupil_stack = zernike_pupil[YY, XX]
    
    Psi = pupil_stack * object_stack 
    
    return Psi 
    
    
def project_support(support, objectFTs, pupil, pupil_slices, object_slices):
    
    objectFTnew =  fft_2D( ifft_2D(objectFTs) * support )
    
    object_stack = objectFTnew[object_slices]
        
    # CHECK ME: the order of Xs and Ys
    Ys = np.array([np.arange(sl[0].start, sl[0].stop) for sl in pupil_slices])
    Xs = np.array([np.arange(sl[1].start, sl[1].stop) for sl in pupil_slices])
    YY = Ys[:, :, None]  
    XX = Xs[:, None, :]
    
    pupil_stack = pupil[YY, XX]
    
    Psi = pupil_stack * object_stack 
    
    return Psi 

# ~~~___________________ Algorithms ___________________~~~

class DM(AlgorithmKernel):
    def __init__(self, beta = 1.0, beta_decay = None):

        self.beta = beta

        self.beta_decay = beta_decay
        
    # def step_single(self, slices, objectFT, pupil_func, PSI, images):
        
    #     Psi_model = project_model(slices, pupil_func, objectFT)
        
    #     Psi_reflection = 2 * Psi_model - PSI
        
    #     Psi_data_reflection = project_data(images, Psi_reflection)

    #     Psi_n = PSI + self.beta * (Psi_data_reflection - Psi_model)
        
    #     if self.beta_decay is not None:
    #         self.beta *= self.beta_decay
            
    #     return Psi_n

    def step(self, pupil_slices, object_slices, objectFTs, pupil_func, PSI, images):
        
        Psi_model = project_model(pupil_slices = pupil_slices, 
                                  object_slices=object_slices, 
                                  objectFTs=objectFTs, 
                                  pupil=pupil_func)
        
        Psi_reflection = 2 * Psi_model - PSI
        
        Psi_data_reflection = project_data(images=images, Psi=Psi_reflection)

        Psi_n = PSI + self.beta * (Psi_data_reflection - Psi_model)
        
        if self.beta_decay is not None:
            self.beta *= self.beta_decay
            
        return Psi_n
    

        

class DNC(AlgorithmKernel):
    
    def __init__(self, projections: list, beta = 0.9, weights: list = [1.0,1.0,1.0,1.0]):
        
        self.beta = beta 
        self.projections = projections
        self.weights = [w / sum(weights) for w in weights]
        self.replicas = [None for _ in range(len(projections))] # list of PSI
    
    def divide(self, PSI, projections):

        return [proj(PSI) for proj in projections]
    
    def concur(self, PSI_list, weights=None):
        
        if weights is None:
            weights = np.ones(len(PSI_list)) / len(PSI_list)
        
        out = np.zeros_like(PSI_list[0], dtype=PSI_list[0].dtype)
        
        for w, psi in zip(weights, PSI_list):
            out += w * psi
        
        return out
    
        
    def step(self, PSI, pupil_slices, object_slices,
                    objectFTs, pupil_func, images, **kwargs):

        def P_model(PSI_in):
            Psi_model = project_model(
                pupil_slices=pupil_slices,
                object_slices=object_slices,
                pupil=pupil_func,
                objectFTs=objectFTs
            )
            return Psi_model

        def P_data(PSI_in):
            Psi_data = project_data(images, PSI_in)
            return Psi_data

        def P_support(PSI_in):
            # 1. apply support in object domain
            obj_supported = fft_2D(
                ifft_2D(objectFTs) * self.support
            )
            # 2. regenerate PSI from (O,P)
            Psi_sup = project_model(
                pupil_slices=pupil_slices,
                object_slices=object_slices,
                pupil=pupil_func,
                objectFTs=obj_supported
            )
            return Psi_sup

        def P_zernike(PSI_in):
            # zernike correction of probe
            amp = np.abs(pupil_func)
            pha = np.angle(pupil_func)
            P_fixed = amp * np.exp(
                1j * SquarePolynomials.project_wavefront(pha, coords=self.pupil_coords)
            )
            Psi_z = project_model(
                pupil_slices=pupil_slices,
                object_slices=object_slices,
                pupil=P_fixed,
                objectFTs=objectFTs
            )
            return Psi_z
        
        
        projections = [P_model, P_data, P_support, P_zernike]
        PD = self.divide(PSI, projections)
        PC = self.concur(PD)
        
        beta = self.beta
        PSI_reflect_C = 2 * PC - PSI        
        PSI_reflect_D = 2 * PD[0] - PSI

        # DM update:
        PSI_next = PSI + beta * (
            self.concur(self.divide(PSI_reflect_C, projections)) -
            self.concur([project(PSI_reflect_D) for project in projections])
        )
        
        # gamma_D = 1/(self.beta)
        # gamma_C = -1/(self.beta)

        # fD_model = (1+gamma_D) * project_model(slices, pupil_func, objectFT) - gamma_D * PSI
        # fD_data = (1+gamma_D) * project_data(images, PSI) - gamma_D * PSI
        # fD_zernike = (1+gamma_D) * project_Zernike(objectFT, pupil_func, slices, pupil_coords) - gamma_D * PSI
        # fD_support = (1+gamma_D) * project_support(support, objectFT, pupil_func, slices) - gamma_D * PSI
        
        # PC_o_fD = self.weights[0] * fD_model + self.weights[1] * fD_data + self.weights[2] * fD_zernike + self.weights[3] * fD_support 

        # PD_o_fC = PSI 
        
        
        # Psi_n = PSI + self.beta * ( PC_o_fD - PSI ) 
        
        return PSI_next



class DNC_wZernike(AlgorithmKernel):
    
    def __init__(self, beta = 0.9, weights: list = [1.0,1.0,1.0]):
        
        self.beta = beta 
        
        self.weights = [w / sum(weights) for w in weights ]
        
    def step_single(self, slices, objectFT, pupil_func, PSI , images, support, pupil_coords):
        
        Reflection_model = 2 * project_model(slices, pupil_func, objectFT) - PSI
        Reflection_data = 2 * project_data(images, PSI) - PSI
        Reflection_zernike = 2 * project_Zernike(objectFT, pupil_func, slices, pupil_coords) - PSI
        
        
        concur = self.weights[0] * Reflection_model + self.weights[1] * Reflection_data + self.weights[2] * Reflection_zernike 
        
        Psi_n = PSI + self.beta * ( concur - PSI ) 

        return Psi_n

class DNC_wSupport(AlgorithmKernel):
    
    def __init__(self, beta = 0.9, weights: list = [1.0,1.0,1.0]):
        
        self.beta = beta 
        
        self.weights = [w / sum(weights) for w in weights ]
        
    def step_single(self, slices, objectFT, pupil_func, PSI , images, support, pupil_coords):

        projection_model = project_model(slices, pupil_func, objectFT)
        projection_data = project_data(images, PSI)
        projection_support = project_support(support, objectFT, pupil_func, slices)

        
        Reflection_model = 2 * projection_model - PSI
        Reflection_data = 2 * projection_data - PSI
        Reflection_support = 2 * projection_support - PSI
        
        
        concur = self.weights[0] * Reflection_model + self.weights[1] * Reflection_data + self.weights[2] * Reflection_support 
        
        Psi_n = PSI + self.beta * ( concur - PSI ) 

        return Psi_n



class RAAR(AlgorithmKernel):
    
    def __init__(self, beta = 0.9, beta_decay = None):

        self.beta = beta

        self.beta_decay = beta_decay
        
    
    def step_single(self, slices, objectFT, pupil_func, PSI, images):
         
        Psi_reflection_model = 2*project_model(slices, pupil_func, objectFT) - PSI

        Psi_reflection_data = 2*project_data(images, PSI) - PSI
        # Psi_project_model = self.project_data(images, Psi_reflection_model)

        average_reflections = 0.5 * (Psi_reflection_data + Psi_reflection_model)

        
        Psi_n = PSI + self.beta * average_reflections #* Psi_project_model + (1-self.beta) * PSI
            
        return Psi_n
    


class AAR(AlgorithmKernel):
    
    def step_single(self, slices, objectFT, pupil_func, PSI, images):
        
        Psi_model = project_model(slices, pupil_func, objectFT)

        Psi_data_reflection = project_data(images, 2*Psi_model - PSI)

        Psi_n = 0.5 * (PSI + Psi_data_reflection)

        return Psi_n




class HPR(AlgorithmKernel):
    def __init__(self, beta = 0.9):

        self.beta = beta
    def step_single(self, slices, objectFT, pupil_func, PSI, images):
        Psi_model = project_model(slices, pupil_func, objectFT)

        Psi_data_reflection = project_data(images, 2*Psi_model - PSI)

        Psi_n = PSI + self.beta * (Psi_data_reflection - PSI)

        return Psi_n


class DM_PIE(AlgorithmKernel):
    def __init__(self, alpha=0.9):

        self.alpha = alpha

    def step_single(self, slices, objectFT, pupil_func, PSI, images):
        Psi_model = project_model(slices, pupil_func, objectFT)

        Psi_reflection = (1 + self.alpha) * Psi_model - self.alpha * PSI

        Psi_data_reflection = project_data(images, Psi_reflection)

        Psi_n = PSI + Psi_data_reflection - Psi_model

        return Psi_n

class ePIE(AlgorithmKernel):
    def __init__(self, alpha_obj=0.9):

        self.alpha_obj = alpha_obj

    def step_single(self, slices, objectFT, pupil_func, PSI, images):
        
        Psi_model = project_model(slices, pupil_func, objectFT)

        Psi_data = project_data(images, Psi_model)

        Psi_n = PSI + self.alpha_obj * (Psi_data - Psi_model)

        return Psi_n

    def compute_weight_fac(self, func):
        """Compute weighting factor for phase retrieval update."""
        
        mod = np.abs(func) ** 2
        return np.conjugate(func) / (mod.max() + 1e-23)
  
        
        
