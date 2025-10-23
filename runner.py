from typing import Tuple, Sequence, Callable, Any
import numpy as np
from .phase_abstract import Plot, LivePlot, PhaseRetrievalBase
from .utils_pr import *
import inspect
from IPython.display import display, clear_output
from ZernikePolynomials import SquarePolynomials

from .algorithms import AlgorithmKernel

class FourierPtychoRunner(PhaseRetrievalBase, Plot, LivePlot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.liveplot_init = True
        
    def solve(self, kernel: AlgorithmKernel, iterations: int, 
              pupil_update_step: int = 1, 
              zernike_projection = False,
              live_plot: bool = False):
        
        
        if live_plot and self.liveplot_init:
            self._initialize_live_plot()
            self.hr_obj_image = self.inverse_fft(self.hr_fourier_image)
            self._update_live_plot()
            self.liveplot_init = False
            
        for it in range(iterations):
            
            old_PSI = self.PSI.copy()
            
            
            # Update PSI for each image
            self.PSI = kernel.step(PSI=self.PSI, 
                                  objectFT = self.hr_fourier_image, 
                                  pupil_func = self.pupil_func, 
                                  bounds = self.patch_bounds, 
                                  images = self.coherent_imgs_upsampled,
                                  n_jobs = self.num_jobs, 
                                   backend = self.backend
                                  )
            
            # Update Fourier Spectrum
            self.hr_fourier_image = kernel.update_object_ft(PSI = self.PSI, 
                                                            pupil = self.pupil_func, 
                                                            bounds = self.patch_bounds, 
                                                            n_jobs = self.num_jobs, 
                                                            backend = self.backend)
            
            # Update pupil
            if pupil_update_step > 0 and ((self.iters_passed + 1) % pupil_update_step == 0):
                self.pupil_func = kernel.update_pupil(PSI = self.PSI, 
                                                      objectFT = self.hr_fourier_image, 
                                                      bounds = self.patch_bounds, 
                                                      pupil_func = self.pupil_func, 
                                                      ctf = self.ctf,
                                                      n_jobs = self.num_jobs, 
                                                      backend= self.backend)

                if zernike_projection:
                    
                    pha = np.angle(self.pupil_func)
                    amp = np.abs(self.pupil_func)
                    
                    self.pupil_func = amp * np.exp(1j*SquarePolynomials.project(pha))
                    
                        
            err_tot = kernel.compute_error(old_PSI, self.PSI)
            
            self.iter_loss = err_tot / max(1, self.num_images)
            self.losses.append(self.iter_loss)
            
            self.iters_passed += 1

            if live_plot:
                self.hr_obj_image = self.inverse_fft(self.hr_fourier_image)
                self._update_live_plot()
    
    def get_state(self):
        return dict(
        objectFT=self.hr_fourier_image,
        pupil=self.pupil_func,
        PSI=self.PSI,
        iters_passed=self.iters_passed,
        losses=self.losses
        )

    def set_state(self, state: dict):
        self.hr_fourier_image = state['objectFT']
        self.pupil_func = state['pupil']
        self.PSI = state['PSI']
        self.iters_passed = state.get('iters_passed', 0)
        self.losses = state.get('losses', [])
    
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

    
    def compute_error(self, image, PSI):

        image_cal = np.abs( self.inverse_fft(PSI) )**2
        
        err_image = np.sum ( np.abs( image_cal - image ) )/ np.sum(image)

        return err_image
    

class Pipeline:
    
    def __init__(self, runner: FourierPtychoRunner, steps: Sequence[Tuple[AlgorithmKernel, int]]):
        
        self.runner = runner
        self.steps = steps

    
    def run(self, live_plot=False, pupil_update_step=1):
        
        for kernel, n in self.steps:
            
            print(f"Running {kernel.__class__.__name__} for {n} iters")
            
            self.runner.solve(kernel, iterations=n, pupil_update_step=pupil_update_step, live_plot=live_plot)

    def cycle(self, total_iterations: int, live_plot=False, pupil_update_step=1):
        
        iterations_done = 0
        
        while iterations_done < total_iterations:
            
            for kernel, n in self.steps:
                
                print(f"Running {kernel.__class__.__name__} for {n} iters")
                
                remaining = total_iterations - iterations_done
                
                iter_to_run = min(n, remaining)
                
                
                self.runner.solve(kernel, iterations=n, pupil_update_step=pupil_update_step, live_plot=live_plot)
                
                iterations_done += iter_to_run

