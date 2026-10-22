import numpy as np
from joblib import Parallel, delayed
from .utils_pr import time_it
from .zernike import SquarePolynomials
import inspect

fft_images = lambda imgs: np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(imgs, axes=(-2, -1)), axes=(-2, -1)), axes = (-2,-1))
ifft_images = lambda imgs: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(imgs, axes=(-2, -1)), axes=(-2, -1)), axes = (-2,-1))

# fft_2D = lambda img: np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))
# ifft_2D = lambda img: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(img)))

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

    @time_it
    def update_object_fts(self, PSI, pupil, pupil_slices, object_slices, objectFTs=None, eps=1e-6):
        """LSQ object spectra for a fixed pupil (the projection onto {P*O}):

            O = (sum conj(P) psi + mu O_prev) / (sum |P|^2 + mu),  mu = eps * max(sum |P|^2)

        mu is relative, so it does not depend on the free P/O scale. Poorly
        constrained pixels keep O_prev (0 if objectFTs is None) instead of
        blowing up as psi/P; well-constrained ones are effectively exact."""
        x,y = PSI[0].shape
        n = max(object_slices)+1

        denom = np.zeros((n,x,y), dtype=np.float64)
        numer = np.zeros((n,x,y), dtype=np.complex128)

        for pup_sl, obj_sl, psi_i in zip(pupil_slices, object_slices, PSI):

            pupil_patch = pupil[pup_sl]

            denom[obj_sl] += np.abs(pupil_patch)**2

            numer[obj_sl] += np.conjugate(pupil_patch) * psi_i

        mu = eps * denom.max() + 1e-30
        if objectFTs is not None:
            numer += mu * objectFTs
        return numer / (denom + mu)

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
    def update_pupil(self, PSI, objectFT, pupil_slices, object_slices, pupil_func, ctf=None, eps=1e-3):
        """LSQ pupil for fixed objects; pixels with little object power keep
        their previous value (mu = eps * max denom). Zero outside the aperture
        `ctf` if given."""
        pupil_shape = pupil_func.shape
        denom = np.zeros(pupil_shape, dtype=np.float64)
        numer = np.zeros(pupil_shape, dtype=np.complex128)

        for pup_sl, obj_sl, psi_i in zip(pupil_slices, object_slices, PSI):

            denom[pup_sl] += np.abs(objectFT[obj_sl])**2
            numer[pup_sl] += np.conjugate(objectFT[obj_sl]) * psi_i

        mu = eps * denom.max() + 1e-30
        pupil_func_update = (numer + mu * pupil_func) / (denom + mu)

        if ctf is not None:
            pupil_func_update = pupil_func_update * (np.abs(ctf) > 0)

        return pupil_func_update
        
    
    
    
    # ~~~~___________ Helpers ___________~~~~

    def compute_error(self, old_psi_list, new_psi_list):

        errors = []
        
        for old_psi, new_psi in zip(old_psi_list, new_psi_list):
            norm_old = np.linalg.norm(old_psi)
            if norm_old == 0:      # all-zero (padding) image: skip
                continue
            err = np.linalg.norm(new_psi - old_psi) / norm_old
            errors.append(err)

        total_error = np.mean(errors) if errors else 0.0

        return total_error, errors

    def compute_data_misfit(self, pupil_slices, object_slices, pupil, objectFTs, images):
        """Amplitude misfit of the current model P*O against the data:
            E = sum_j || |ifft(P_j O_c(j))| - sqrt(I_j) ||^2 / sum_j I_j
        Uses the model (object + pupil), not the iterate PSI."""
        psi_model = ifft_images(project_model(pupil_slices, object_slices, pupil, objectFTs))
        amp_data = np.sqrt(images)
        err_per_image = np.sum((np.abs(psi_model) - amp_data)**2, axis=(-2, -1))
        total_error = np.sum(err_per_image) / np.sum(images)
        return total_error, err_per_image / np.sum(images, axis=(-2, -1))

    def compute_gap(self, PSI, pupil_slices, object_slices, pupil, images):
        """Gap between the two constraint sets at the iterate PSI:
            gap = || P_D(PSI) - P_M(PSI) || / || PSI ||
        P_M fits the objects to PSI with the pupil held fixed (the exact
        projection onto {P*O} for that pupil), then re-evaluates P*O."""
        Psi_M, _ = project_model_fit(self, PSI, pupil_slices, object_slices, pupil)
        Psi_D = project_data(images, PSI)
        return np.linalg.norm(Psi_D - Psi_M) / np.linalg.norm(PSI)

    def solution_estimate(self, PSI, images):
        """Exit waves the object/pupil should be fitted to after a step.
        The iterate itself by default; kernels whose solution lives in a
        projection of the iterate (e.g. RAAR_standard -> P_D(Ψ)) override."""
        return PSI

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


def project_model_fit(kernel, PSI, pupil_slices, object_slices, pupil, objectFTs=None):
    """P_M(PSI) for a fixed pupil: refit the objects to PSI, then re-predict.
    Exact projection onto {P*O} (up to the small Tikhonov term in
    update_object_fts). Returns (Psi_M, objectFTs_fit)."""
    objectFTs_fit = kernel.update_object_fts(PSI, pupil, pupil_slices, object_slices,
                                             objectFTs=objectFTs)
    return project_model(pupil_slices, object_slices, pupil, objectFTs_fit), objectFTs_fit


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
    
    objectFTnew =  fft_images( ifft_images(objectFTs) * support )
    
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
    """Difference Map (Thibault 2008), relaxed:

        Ψ <- Ψ + β [ P_D(2 P_M Ψ - Ψ) - P_M Ψ ]  =  (1-β) Ψ + β T_DM(Ψ)

    β only shortens the step; the fixed points are those of β = 1. P_M refits
    the objects to Ψ inside the step (exact projection for the current pupil).

    estimate : what the engine fits object/pupil to after the step.
        'data'  -> the shadow P_D(2 P_M Ψ - Ψ). It lies on the data set
                   (|ψ| = sqrt(I)), so it cannot drift when the sets do not
                   intersect, and equals P_M Ψ at a fixed point.
        'model' -> the iterate Ψ itself (fitting to it gives P_M Ψ, Thibault's
                   read-out); Ψ drifts for inconsistent data.
    """
    def __init__(self, beta = 1.0, beta_decay = None, beta_min = 0.5, estimate = 'data'):

        self.beta = beta

        self.beta_decay = beta_decay
        self.beta_min = beta_min

        if estimate not in ('data', 'model'):
            raise ValueError("estimate must be 'data' or 'model'")
        self.estimate = estimate
        self._shadow = None

    # def step_single(self, slices, objectFT, pupil_func, PSI, images):
        
    #     Psi_model = project_model(slices, pupil_func, objectFT)
        
    #     Psi_reflection = 2 * Psi_model - PSI
        
    #     Psi_data_reflection = project_data(images, Psi_reflection)

    #     Psi_n = PSI + self.beta * (Psi_data_reflection - Psi_model)
        
    #     if self.beta_decay is not None:
    #         self.beta *= self.beta_decay
            
    #     return Psi_n

    def step(self, pupil_slices, object_slices, objectFTs, pupil_func, PSI, images):

        Psi_model, _ = project_model_fit(self, PSI, pupil_slices, object_slices,
                                         pupil_func, objectFTs)   # P_M(Ψ)

        Psi_reflection = 2 * Psi_model - PSI

        Psi_data_reflection = project_data(images=images, Psi=Psi_reflection)
        self._shadow = Psi_data_reflection

        Psi_n = PSI + self.beta * (Psi_data_reflection - Psi_model)

        if self.beta_decay is not None:
            self.beta = max(self.beta * self.beta_decay, self.beta_min)

        return Psi_n

    def solution_estimate(self, PSI, images):
        if self.estimate == 'data' and self._shadow is not None:
            return self._shadow
        return PSI
    

        

class DNC(AlgorithmKernel):
    
    def __init__(self, beta = 0.9, weights: list = [1.0,1.0,1.0,1.0], support = None):
        
        self.beta = beta 
        self.weights = [w / sum(weights) for w in weights]
        self.support = support 
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
            self.support = np.ones_like(PSI_in[0]) 
            obj_supported = fft_images(
                ifft_images(objectFTs) * self.support
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
                1j * SquarePolynomials.project_wavefront(pha, coords=kwargs['pupil_coords'])
            )
            Psi_z = project_model(
                pupil_slices=pupil_slices,
                object_slices=object_slices,
                pupil=P_fixed,
                objectFTs=objectFTs
            )
            return Psi_z
        
        gamma_D = 1/self.beta
        gamma_C = -1/self.beta
        projections = [P_model, P_data, P_support, P_zernike]
        
        fD = (1+gamma_D) * self.divide(PSI, projections) - gamma_D * PSI
        
        PC = self.concur(PD, weights = self.weights)
        
        beta = self.beta
        PSI_reflect_C = 2 * PC - PSI        
        PSI_reflect_D = 2 * PD[0] - PSI

        # DM update:
        PSI_next = PSI + beta * (
            self.concur(self.divide(PSI_reflect_C, projections), weights = self.weights) -
            self.concur([project(PSI_reflect_D) for project in projections], weights = self.weights)
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
    """Non-standard RAAR variant: beta·P_D(R_M(psi)) + (1-beta)·psi.

    This is a relaxed reflect-then-project step, NOT the Luke (2004) RAAR.
    It works well in practice; see RAAR_standard for the textbook version.
    """

    def __init__(self, beta = 0.9, beta_decay = None):

        self.beta = beta

        self.beta_decay = beta_decay


    def step(self,PSI, pupil_slices, object_slices,
                    objectFTs, pupil_func, images, **kwargs):

        Psi_reflection_model = 2*project_model(
                pupil_slices=pupil_slices,
                object_slices=object_slices,
                pupil=pupil_func,
                objectFTs=objectFTs
                ) - PSI

        Psi_project_model = project_data(images, Psi_reflection_model)

        Psi_n = self.beta * Psi_project_model + (1-self.beta) * PSI

        return Psi_n


class RAAR_standard(AlgorithmKernel):
    """Standard RAAR (Luke 2004), with the data (modulus) set D as the inner
    set and the model set M = {P*O} as the outer one:

        Ψ_{n+1} = (β/2)·(R_M(R_D(Ψ)) + Ψ) + (1-β)·P_D(Ψ)
                = β·Ψ + β·P_M(R_D(Ψ)) + (1-2β)·P_D(Ψ)

    β = 1 is Douglas-Rachford (Fienup HIO with β = 1). P_M is the exact
    projection for the current pupil: the objects are refitted to R_D(Ψ)
    inside the step. The solution is read from P_D(Ψ), so the engine fits
    object/pupil to `solution_estimate(Ψ)` rather than to the iterate.

    beta_decay multiplies β every step, but never below beta_min: as β→0
    the update collapses to Ψ = P_D(Ψ) and stops making progress.
    """

    def __init__(self, beta=0.9, beta_decay=None, beta_min=0.5):
        self.beta = beta
        self.beta_decay = beta_decay
        self.beta_min = beta_min

    def step(self, PSI, pupil_slices, object_slices,
             objectFTs, pupil_func, images, **kwargs):

        Psi_data = project_data(images, PSI)           # P_D(Ψ)
        R_D = 2 * Psi_data - PSI                       # R_D(Ψ)

        P_M_RD, _ = project_model_fit(self, R_D, pupil_slices, object_slices,
                                      pupil_func, objectFTs)   # P_M(R_D(Ψ))

        Psi_n = self.beta * PSI + self.beta * P_M_RD + (1 - 2 * self.beta) * Psi_data

        if self.beta_decay is not None:
            self.beta = max(self.beta * self.beta_decay, self.beta_min)

        return Psi_n

    def solution_estimate(self, PSI, images):
        return project_data(images, PSI)
    


class ER(AlgorithmKernel):
    """Error Reduction (alternating projections): Ψ = P_D(P_M(Ψ)).

    Maximally stable; slowest convergence. Useful as a baseline or to finish
    a run started with a faster algorithm.
    """

    def step(self, PSI, pupil_slices, object_slices,
             objectFTs, pupil_func, images, **kwargs):

        Psi_model = project_model(
            pupil_slices=pupil_slices,
            object_slices=object_slices,
            pupil=pupil_func,
            objectFTs=objectFTs
        )
        return project_data(images, Psi_model)


class AAR(AlgorithmKernel):
    """Averaged project-reflect step: Ψ = (1/2)·(Ψ + P_D(R_M(Ψ)))."""

    def step(self, PSI, pupil_slices, object_slices,
             objectFTs, pupil_func, images, **kwargs):

        Psi_model = project_model(
            pupil_slices=pupil_slices,
            object_slices=object_slices,
            pupil=pupil_func,
            objectFTs=objectFTs
        )

        Psi_data_reflection = project_data(images, 2*Psi_model - PSI)

        Psi_n = 0.5 * (PSI + Psi_data_reflection)

        return Psi_n




class HPR(AlgorithmKernel):
    def __init__(self, beta = 0.9):

        self.beta = beta
    def step(self, pupil_slices, object_slices, objectFTs, pupil_func, PSI, images):
        
        Psi_model = project_model(
                pupil_slices=pupil_slices,
                object_slices=object_slices,
                pupil=pupil_func,
                objectFTs=objectFTs
                )

        Psi_data_reflection = project_data(images, 2*Psi_model - PSI)

        Psi_n = PSI + self.beta * (Psi_data_reflection - PSI)

        return Psi_n


class DM_PIE(AlgorithmKernel):
    def __init__(self, alpha=0.9):

        self.alpha = alpha

    def step(self, PSI, pupil_slices, object_slices,
             objectFTs, pupil_func, images, **kwargs):

        Psi_model = project_model(
            pupil_slices=pupil_slices,
            object_slices=object_slices,
            pupil=pupil_func,
            objectFTs=objectFTs
        )

        Psi_reflection = (1 + self.alpha) * Psi_model - self.alpha * PSI

        Psi_data_reflection = project_data(images, Psi_reflection)

        Psi_n = PSI + Psi_data_reflection - Psi_model

        return Psi_n

class ePIE(AlgorithmKernel):
    def __init__(self, alpha_obj=0.9):

        self.alpha_obj = alpha_obj

    def step(self, PSI, pupil_slices, object_slices,
             objectFTs, pupil_func, images, **kwargs):

        Psi_model = project_model(
            pupil_slices=pupil_slices,
            object_slices=object_slices,
            pupil=pupil_func,
            objectFTs=objectFTs
        )

        Psi_data = project_data(images, Psi_model)

        Psi_n = PSI + self.alpha_obj * (Psi_data - Psi_model)

        return Psi_n

    def compute_weight_fac(self, func):
        mod = np.abs(func) ** 2
        return np.conjugate(func) / (mod.max() + 1e-23)
  
        
        
