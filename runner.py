from typing import Tuple, Sequence, Callable, Any
import numpy as np
from .phase_abstract import Plot, LivePlot, PhaseRetrievalBase
from .utils_pr import *
import inspect
from IPython.display import display, clear_output
from .zernike import SquarePolynomials

from .algorithms import AlgorithmKernel

class FourierPtychoEngine(PhaseRetrievalBase, Plot, LivePlot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.liveplot_init = True
        
    def solve(self, kernel: AlgorithmKernel, 
              iterations: int, 
              object_steps = None,
              pupil_steps = None, 
              zernike_steps = None,
              live_plot: bool = False):
        
        
        if live_plot and self.liveplot_init:
            
            self.rec_obj_images = self.inverse_fft(self.rec_fourier_images)
            self._initialize_live_plot()
            self._update_live_plot()
            self.liveplot_init = False
        
        for it in range(iterations):

            
            
            old_PSI = self.PSI.copy()
            
            # ________ Update PSI ________
            self.PSI = kernel.step(PSI_list=self.PSI, 
                                  objectFT_list = self.rec_fourier_images, 
                                  pupil_func = self.pupil_func, 
                                  slices_list=self.pupil_slices,
                                    images_list = self.images,
                                  n_jobs = self.num_jobs, 
                                   backend = self.backend
                                  )

            print('Step done')


            # ________ Object update ________
            if object_steps is not None:
                if (it % (object_steps + pupil_steps)) < object_steps:
                    self.rec_fourier_images = kernel.update_object_fts(
                        PSI_list=self.PSI, 
                        pupil=self.pupil_func, 
                        slices_list= self.pupil_slices 
                    )
                    print('Object Update')
        
                # ________ Pupil update ________
                else:
                    self.pupil_func = kernel.update_pupil(
                        PSI_list=self.PSI, 
                        objectFT_list=self.rec_fourier_images, 
                        slices_list = self.pupil_slices,
                        pupil_func=self.pupil_func, 
                        ctf=self.ctf,
                    )
                    
                    print('Pupil Update')
                    
                    if zernike_steps is not None and it % zernike_steps == 0 :
                        pha = np.angle(self.pupil_func)
                        amp = np.abs(self.pupil_func)
                        self.pupil_func = amp * np.exp(
                            1j * SquarePolynomials.project_wavefront(pha, coords=self.pupil_coords)
                        )
                        print("Zernike Update")
            else: 
                self.rec_fourier_images = kernel.update_object_fts(
                        PSI_list=self.PSI, 
                        pupil=self.pupil_func, 
                        slices_list= self.pupil_slices 
                    )
                
                print('Object Update')
                
                self.pupil_func = kernel.update_pupil(
                        PSI_list=self.PSI, 
                        objectFT_list=self.rec_fourier_images, 
                        slices_list = self.pupil_slices,
                        pupil_func=self.pupil_func, 
                        ctf=self.ctf,
                    )
                    
                print('Pupil Update')
                    
                if zernike_steps is not None and it % zernike_steps == 0 :
                    pha = np.angle(self.pupil_func)
                    amp = np.abs(self.pupil_func)
                    self.pupil_func = amp * np.exp(
                        1j * SquarePolynomials.project_wavefront(pha, coords=self.pupil_coords)
                    )
                    print("Zernike Update")
                        
            # ________ Error ________
            err_tot, _ = kernel.compute_error(old_PSI, self.PSI)
            
            self.iter_loss = err_tot #/ max(1, self.num_images)
            self.losses.append(self.iter_loss)
            
            self.iters_passed += 1

            if live_plot:
                central_idx = self.num_streaks//2
                self.rec_obj_images[central_idx] = self.inverse_fft(self.rec_fourier_images[central_idx])
                self._update_live_plot()

    def _post_process(self):

        self.rec_obj_images = [self.inverse_fft(img) for img in self.rec_fourier_images]
    
    def get_state(self):
        return dict(
        objectFT=self.rec_fourier_images,
        pupil=self.pupil_func,
        PSI=self.PSI,
        iters_passed=self.iters_passed,
        losses=self.losses
        )

    def set_state(self, state: dict):
        self.rec_fourier_images = state['objectFT']
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
    
    def __init__(self, engine: FourierPtychoEngine, steps: Sequence[Tuple[AlgorithmKernel, int]]):
        
        self.engine = engine
        self.steps = steps

    
    def run(self, live_plot=False, pupil_steps=None, object_steps=None, zernike_steps = None):
        
        for kernel, n in self.steps:
            
            print(f"Running {kernel.__class__.__name__} for {n} iters")
            self.engine.solve(kernel, iterations=n, object_steps=object_steps, pupil_steps=pupil_steps, 
                              zernike_steps=zernike_steps, live_plot=live_plot)

        self.engine._post_process()

    def cycle(self, total_iterations: int, live_plot=False, pupil_steps=None, object_steps=None, zernike_steps=None):
        
        iterations_done = 0
        
        while iterations_done < total_iterations:
            
            for kernel, n in self.steps:
                
                print(f"Running {kernel.__class__.__name__} for {n} iters")
                
                remaining = total_iterations - iterations_done
                
                iter_to_run = min(n, remaining)
                
                
                self.engine.solve(kernel, iterations=n, object_steps=object_steps, pupil_steps=pupil_steps, 
                                  zernike_steps=zernike_steps, live_plot=live_plot)
                
                iterations_done += iter_to_run
        
        self.engine._post_process()