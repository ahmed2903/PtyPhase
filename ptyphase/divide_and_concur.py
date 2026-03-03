from typing import Tuple, Sequence, Callable, Any
import numpy as np
import imageio.v2 as imageio
from dataclasses import dataclass

from .phase_abstract import Plot, LivePlot, PhaseRetrievalBase
from .utils_pr import *
import inspect
from IPython.display import display, clear_output
from .zernike import SquarePolynomials

from .algorithms import AlgorithmKernel

@dataclass
class FourierPtychoState:
    objectFT: np.ndarray
    pupil: np.ndarray
    PSI: np.ndarray
    iters_passed: int = 0
    losses: list = None

    def copy(self):
        return FourierPtychoState(
            objectFT=self.objectFT.copy(),
            pupil=self.pupil.copy(),
            PSI=self.PSI.copy(),
            iters_passed=self.iters_passed,
            losses=list(self.losses) if self.losses is not None else []
        )
        
class FourierPtychoEngine(PhaseRetrievalBase, Plot, LivePlot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.liveplot_init = True
        self._gif_frames_amp = []
        self._gif_frames_pha = []
        self._gif_frames_pupil = []
        
    def solve(self, kernel: AlgorithmKernel, 
              iterations: int, 
              object_steps = None,
              pupil_steps = None, 
              zernike_steps = None,
              live_plot: bool = False,
              save_gif: bool= False,
              **kwargs):
        
        
        if live_plot and self.liveplot_init:
            
            self.rec_obj_images = self.inverse_fft(self.rec_fourier_images)
            self._initialize_live_plot()
            self._update_live_plot()
            self.liveplot_init = False
        
        step_args = self.GetKwArgs(kernel.step, kwargs)
                
        for it in range(iterations):
            
            old_PSI = self.PSI.copy()
            
            # ________ Update PSI ________
            self.PSI = kernel.step(PSI=self.PSI, 
                                  objectFTs = self.rec_fourier_images, 
                                  pupil_func = self.pupil_func, 
                                  pupil_slices=self.pupil_slices,
                                  object_slices = self.object_slices,
                                    images = self.images,
                                  #n_jobs = self.num_jobs, 
                                   #backend = self.backend, 
                                   # pupil_coords = self.pupil_coords,
                                   **step_args
                                  )

            print('Step done')

 
            self.rec_fourier_images = kernel.update_object_fts(
                    PSI=self.PSI, 
                    pupil=self.pupil_func, 
                    pupil_slices= self.pupil_slices,
                    object_slices=self.object_slices
                )
            
            print('Object Update')
            
            self.pupil_func = kernel.update_pupil(
                    PSI=self.PSI, 
                    objectFT=self.rec_fourier_images, 
                    pupil_slices = self.pupil_slices,
                    pupil_func=self.pupil_func,
                    object_slices=self.object_slices, 
                    ctf=self.ctf,
                )
                
            print('Pupil Update')
                        
            # ________ Error ________
            err_tot, _ = kernel.compute_error(old_PSI, self.PSI)
            
            self.iter_loss = err_tot #/ max(1, self.num_images)
            self.losses.append(self.iter_loss)
            
            self.iters_passed += 1

            if live_plot:
                central_idx = self.num_streaks//2
                self.rec_obj_images[central_idx] = self.inverse_fft(self.rec_fourier_images[central_idx])
                self._update_live_plot()

            if save_gif and self.iters_passed%20 == 0 : 
                central_idx = self.num_streaks//2
                self.rec_obj_images[central_idx] = self.inverse_fft(self.rec_fourier_images[central_idx])
                # Normalization for visualization (optional but recommended)
                frame = np.abs(self.rec_obj_images[central_idx])
                frame = frame / frame.max()
                frame_uint8 = (frame * 255).astype(np.uint8)
                self._gif_frames_amp.append(frame_uint8)

                # Normalization for visualization (optional but recommended)
                phase = np.angle(self.rec_obj_images[central_idx]) 
                phase_norm = (phase + np.pi) / (2 * np.pi)
                phase_uint8 = (phase_norm * 255).astype(np.uint8)
                self._gif_frames_pha.append(phase_uint8)

                # Normalization for visualization (optional but recommended)
                phase = np.angle(self.pupil_func) 
                phase_norm = (phase + np.pi) / (2 * np.pi)
                phase_uint8 = (phase_norm * 255).astype(np.uint8)
                self._gif_frames_pupil.append(phase_uint8)
            
            
    def _post_process(self, save_gif = False):

        self.rec_obj_images = [self.inverse_fft(img) for img in self.rec_fourier_images]

        if save_gif: 
            imageio.mimsave("phase_reconstruction.gif", self._gif_frames_pha, fps=10)
            imageio.mimsave("amp_reconstruction.gif", self._gif_frames_amp, fps=10)
            imageio.mimsave("pupil_reconstruction.gif", self._gif_frames_pupil, fps=10)
    
    def get_state(self) -> FourierPtychoState:
        return FourierPtychoState(
            objectFT=self.rec_fourier_images,
            pupil=self.pupil_func,
            PSI=self.PSI,
            iters_passed=self.iters_passed,
            losses=self.losses
        )

    def set_state(self, state: FourierPtychoState):
        self.rec_fourier_images = state.objectFT
        self.pupil_func = state.pupil
        self.PSI = state.PSI
        self.iters_passed = state.iters_passed
        self.losses = state.losses


    def GetKwArgs(self, obj, kwargs):
        obj_sigs = list(inspect.signature(obj).parameters.keys())
        obj_args = {k: v for k, v in kwargs.items() if k in obj_sigs}
        return obj_args


class DnC:
    
    def __init__(self, engine: FourierPtychoEngine, projections: Sequence):
        self.engine = engine
        self.projections = projections
        self.n_replicas = len(projections)

        self.PSIs = [engine.PSI for _ in range(self.n_replicas)]
        
        
    def divide_projection(self, psis):
        
        state = np.array([proj(psi) for proj, psi in zip(self.projections, psis) ])
        
        return state
    
    def concur_projection(self, psis):
        
        avg = np.mean(psis, axis = 0, keepdims = True)
        
        return avg
        
    
    def step(self,
        beta = .9,
        **kwargs):
        
            
        PSIs  = self.PSIs.copy()
        f_divide = (1+1/beta) * self.divide_projection(PSIs) - 1/beta * PSIs
        pc_o_fd = self.concur_projection(f_divide)
        
        pd = self.divide_projection(PSIs) 
        
        self.PSIs = PSIs + beta * (pc_o_fd - pd)




class Pipeline:
    
    def __init__(self, engine: FourierPtychoEngine, steps: Sequence[Tuple[AlgorithmKernel, int]]):
        
        self.engine = engine
        self.steps = steps

    
    def run(self, live_plot=False, save_gif = False, pupil_steps=None, object_steps=None, zernike_steps = None, **kwargs):
        for kernel, n in self.steps:
            
            print(f"Running {kernel.__class__.__name__} for {n} iters")
            self.engine.solve(kernel, iterations=n, object_steps=object_steps, pupil_steps=pupil_steps, 
                              zernike_steps=zernike_steps, live_plot=live_plot, save_gif = save_gif, **kwargs)

        self.engine._post_process(save_gif=save_gif)

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