from typing import Tuple, Sequence, Callable, Any
import numpy as np
import imageio.v2 as imageio

from .phase_abstract import Plot, LivePlot, PhaseRetrievalBase
from .utils_pr import *
import inspect
from IPython.display import display, clear_output
from .zernike import SquarePolynomials

from .algorithms import AlgorithmKernel, project_model_fit, normalise_pupil_scale
ifft_images_z = lambda imgs: np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(imgs, axes=0), axis=0), axes = (0))

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
              constrain_pupil: bool = True,
              normalise_pupil: bool = True,
              pupil_iters: int = 0,
              independent_pupils: bool = False,
              **kwargs):
        """
        independent_pupils : False (default) -> one pupil shared by all streaks.
            True -> every streak gets its own pupil, and the streaks are reconstructed
            as separate problems (own object, own pupil, own Tikhonov weights and
            P/O scale). On the first call with True the current shared pupil is copied
            to every streak, so a Pipeline can run shared stages first and independent
            ones after. self.pupil_func then has shape (N_streaks, P0, P1) and
            self.pupil_slices entries become (streak, slice0, slice1). Going back to a
            shared pupil is not supported (the per-streak phases/scales of P_s are
            arbitrary, so there is no unique way to merge them).
        constrain_pupil : keep the pupil zero outside the aperture (ctf).
        normalise_pupil : after each pupil update rescale P -> P/c, O -> O·c so that
            rms|P| = 1 on the covered pupil (removes the free P/O scale drift).
        pupil_iters : 0 -> P_M fits the objects for the current pupil, and after the
            step the objects then the pupil are fit once to the estimate.
            n > 0 -> P_M also frees the pupil (n rounds of pupil <-> object LSQ, see
            algorithms.project_model_fit), and the update after the step is the same
            joint fit to the estimate. On object-only iterations (object_steps cycle,
            or a fixed pupil) the pupil stays fixed in both.
        """
        
        self._set_pupil_mode(independent_pupils)

        if live_plot and self.liveplot_init:

            self.rec_obj_images = self.inverse_fft(self.rec_fourier_images)
            self._initialize_live_plot()
            self._update_live_plot()
            self.liveplot_init = False
        
        step_args = self.GetKwArgs(kernel.step, kwargs)
        ctf = self.ctf if constrain_pupil else None
        cover = self.pupil_cover_mask()
        kernel.pupil_ctf = ctf
        kernel.pupil_cover = cover if normalise_pupil else None

        for it in range(iterations):

            old_PSI = self.PSI.copy()

            pupil_free = object_steps is None or (it % (object_steps + pupil_steps)) >= object_steps
            kernel.pupil_iters = pupil_iters if pupil_free else 0

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

            # Exit waves to fit object/pupil to (P_D(PSI) for RAAR_standard, PSI otherwise)
            PSI_est = kernel.solution_estimate(self.PSI, self.images)

            # ________ Joint object + pupil fit (same as P_M) ________
            if pupil_iters > 0:
                _, self.rec_fourier_images, self.pupil_func = project_model_fit(
                    kernel, PSI_est, self.pupil_slices, self.object_slices,
                    self.pupil_func, self.rec_fourier_images)

                if pupil_free and zernike_steps is not None and it % zernike_steps == 0:
                    self._zernike_project_pupil()
                    print("Zernike Update")

            # ________ Object update ________
            elif object_steps is not None:
                if (it % (object_steps + pupil_steps)) < object_steps:
                    self.rec_fourier_images = kernel.update_object_fts(
                        PSI=PSI_est, 
                        pupil=self.pupil_func, 
                        pupil_slices= self.pupil_slices,
                        object_slices=self.object_slices,
                        objectFTs=self.rec_fourier_images,
                    )
                    print('Object Update')
        
                # ________ Pupil update ________
                else:
                    self.pupil_func = kernel.update_pupil(
                        PSI=PSI_est, 
                        objectFT=self.rec_fourier_images, 
                        pupil_slices = self.pupil_slices,
                        pupil_func=self.pupil_func,
                        object_slices=self.object_slices, 
                        ctf=ctf,
                    )
                    
                    if normalise_pupil:
                        self.pupil_func, self.rec_fourier_images = normalise_pupil_scale(
                            self.pupil_func, self.rec_fourier_images, cover)
                    print('Pupil Update')

                    if zernike_steps is not None and it % zernike_steps == 0 :
                        self._zernike_project_pupil()
                        print("Zernike Update")
            else: 
                self.rec_fourier_images = kernel.update_object_fts(
                        PSI=PSI_est, 
                        pupil=self.pupil_func, 
                        pupil_slices= self.pupil_slices,
                        object_slices=self.object_slices,
                        objectFTs=self.rec_fourier_images,
                    )
                
                print('Object Update')
                
                self.pupil_func = kernel.update_pupil(
                        PSI=PSI_est, 
                        objectFT=self.rec_fourier_images, 
                        pupil_slices = self.pupil_slices,
                        pupil_func=self.pupil_func,
                        object_slices=self.object_slices, 
                        ctf=ctf,
                    )
                    
                if normalise_pupil:
                    self.pupil_func, self.rec_fourier_images = normalise_pupil_scale(
                        self.pupil_func, self.rec_fourier_images, cover)
                print('Pupil Update')

                if zernike_steps is not None and it % zernike_steps == 0 :
                    self._zernike_project_pupil()
                    print("Zernike Update")
                        
            # ________ Error ________
            err_tot, _ = kernel.compute_error(old_PSI, self.PSI)
            
            self.iter_loss = err_tot #/ max(1, self.num_images)
            self.losses.append(self.iter_loss)

            misfit_tot, _ = kernel.compute_data_misfit(
                pupil_slices=self.pupil_slices,
                object_slices=self.object_slices,
                pupil=self.pupil_func,
                objectFTs=self.rec_fourier_images,
                images=self.images,
            )
            self.data_misfits.append(misfit_tot)

            gap = kernel.compute_gap(
                PSI=self.PSI,
                pupil_slices=self.pupil_slices,
                object_slices=self.object_slices,
                pupil=self.pupil_func,
                images=self.images,
            )
            self.gaps.append(gap)

            print(f"Iter {self.iters_passed}: step change = {err_tot:.4e}, "
                  f"data misfit = {misfit_tot:.4e}, gap = {gap:.4e}")

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
                phase = np.angle(self.pupil_display())
                phase_norm = (phase + np.pi) / (2 * np.pi)
                phase_uint8 = (phase_norm * 255).astype(np.uint8)
                self._gif_frames_pupil.append(phase_uint8)
            

    

    def pupil_cover_mask(self):
        """Bool mask of the pupil pixels the data constrains (inside the aperture)."""
        if getattr(self, "pupil_cover", None) is None:
            cover = np.zeros(self.pupil_func.shape, dtype=bool)
            for sl in self.pupil_slices:
                cover[sl] = True
            self.pupil_cover = cover & (np.abs(self.ctf) > 0)
        return self.pupil_cover

    # ________ Shared / independent pupil ________

    @property
    def independent_pupils(self):
        return np.ndim(self.pupil_func) == 3

    def _set_pupil_mode(self, independent):
        if independent and not self.independent_pupils:
            self.split_pupil()
        elif not independent and self.independent_pupils:
            raise ValueError(
                "The pupils are already independent (one per streak); a shared pupil "
                "cannot be recovered from them. Pass independent_pupils=True.")

    def split_pupil(self):
        """Shared pupil -> one copy per streak, shape (N_streaks, P0, P1).
        Image j then uses pupil_func[object_slices[j]]."""
        n_streaks = len(self.rec_fourier_images)
        self.pupil_func = np.repeat(np.asarray(self.pupil_func)[None], n_streaks, axis=0)
        self._index_pupil_slices_by_streak()
        print(f"Independent pupils: {n_streaks} streaks, pupil stack {self.pupil_func.shape}")

    def _index_pupil_slices_by_streak(self):
        """(slice0, slice1) -> (streak, slice0, slice1); no-op if already done."""
        self.pupil_slices = [sl if len(sl) == 3 else (int(s), sl[0], sl[1])
                             for sl, s in zip(self.pupil_slices, self.object_slices)]
        self.pupil_cover = None        # rebuilt with the new shape on next use

    def _zernike_project_pupil(self):
        """Project the pupil phase onto the Zernike basis (each streak's pupil
        separately when they are independent)."""
        pupils = self.pupil_func if self.independent_pupils else self.pupil_func[None]
        out = np.empty_like(pupils)
        for s, P in enumerate(pupils):
            out[s] = np.abs(P) * np.exp(
                1j * SquarePolynomials.project_wavefront(np.angle(P), coords=self.pupil_coords))
        self.pupil_func = out if self.independent_pupils else out[0]

    def _post_process(self, save_gif = False):

        self.rec_obj_images = [self.inverse_fft(img) for img in self.rec_fourier_images]
        
        self.obj_3D = ifft_images_z(np.asarray(self.rec_obj_images))
        obj3D = np.pad(self.obj_3D, ((1, 1), (1, 1), (1, 1)))
        
        if save_gif: 
            imageio.mimsave("phase_reconstruction.gif", self._gif_frames_pha, fps=10)
            imageio.mimsave("amp_reconstruction.gif", self._gif_frames_amp, fps=10)
            imageio.mimsave("pupil_reconstruction.gif", self._gif_frames_pupil, fps=10)
        
    def get_state(self):
        return dict(
        objectFT=self.rec_fourier_images,
        pupil=self.pupil_func,
        PSI=self.PSI,
        iters_passed=self.iters_passed,
        losses=self.losses,
        data_misfits=self.data_misfits,
        gaps=self.gaps
        )

    def set_state(self, state: dict):
        self.rec_fourier_images = state['objectFT']
        self.pupil_func = state['pupil']
        if np.ndim(self.pupil_func) == 3:     # state saved with independent pupils
            self._index_pupil_slices_by_streak()
        else:                                  # shared pupil: plain (slice0, slice1)
            self.pupil_slices = [sl[-2:] for sl in self.pupil_slices]
            self.pupil_cover = None
        self.PSI = state['PSI']
        self.iters_passed = state.get('iters_passed', 0)
        self.losses = state.get('losses', [])
        self.data_misfits = state.get('data_misfits', [])
        self.gaps = state.get('gaps', [])

        
    # def GetKwArgs(self, obj, kwargs):
    #     obj_sigs = []
    #     obj_args = {}
    #     for arg in inspect.signature(obj).parameters.values():
    #         if not arg.default is inspect._empty:
    #             obj_sigs.append(arg.name)
    #     for key, value in kwargs.items():
    #         #if key in obj_sigs:
    #         obj_args[key] = value
    #     return obj_args

    def GetKwArgs(self, obj, kwargs):
        obj_sigs = list(inspect.signature(obj).parameters.keys())
        obj_args = {k: v for k, v in kwargs.items() if k in obj_sigs}
        return obj_args
    
    def compute_error(self, image, PSI):

        image_cal = np.abs( self.inverse_fft(PSI) )**2
        
        err_image = np.sum ( np.abs( image_cal - image ) )/ np.sum(image)

        return err_image
    

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