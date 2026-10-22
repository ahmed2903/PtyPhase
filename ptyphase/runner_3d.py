import numpy as np

from .phase_abstract_3d import PhaseRetrieval3DBase, LivePlot3D, Plot3D
from .algorithms_3d import AlgorithmKernel3D, normalise_pupil_scale
from .runner import Pipeline          # works unchanged with this engine
from .zernike import SquarePolynomials


class FourierPtycho3DEngine(PhaseRetrieval3DBase, Plot3D, LivePlot3D):
    """3D-coupled engine. Same flow as FourierPtychoEngine:

        engine = FourierPtycho3DEngine(images=..., kins=..., kouts=..., pupil_func=...,
                                       ks_pupil=..., lr_psize=..., wavelength=...)
        engine.prepare(extend='triple', dqz=None, z_extent_nm=None)
        Pipeline(engine, [(DM3D(beta=1.0), 200), (ER3D(), 50)]).run(object_steps=..., pupil_steps=...)
        engine.obj_3D        # [z, x, y]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.liveplot_init = True

    def solve(self, kernel: AlgorithmKernel3D,
              iterations: int,
              object_steps=None,
              pupil_steps=None,
              zernike_steps=None,
              live_plot: bool = False,
              save_gif: bool = False,
              constrain_pupil: bool = True,
              normalise_pupil: bool = True,
              **kwargs):
        """
        object_steps/pupil_steps : cycle `object_steps` object-only iterations with
            `pupil_steps` pupil-only iterations (as in the 2D engine). pupil_steps=0
            keeps the pupil fixed. None -> object and pupil every iteration.
        constrain_pupil : keep the pupil zero outside the aperture (ctf).
        normalise_pupil : after each pupil update rescale P -> P/c, S -> S·c so that
            rms|P| = 1 on the covered pupil (removes the free P/S scale drift).
        """
        if live_plot and self.liveplot_init:
            self._initialize_live_plot()
            self.liveplot_init = False

        step_args = self.GetKwArgs(kernel.step, kwargs)
        ctf = self.ctf if constrain_pupil else None
        cover = self.pupil_cover_mask()

        for it in range(iterations):

            old_PSI = self.PSI.copy()

            # ________ Update PSI ________
            self.PSI = kernel.step(PSI=self.PSI,
                                   geom=self.geom,
                                   objectFT=self.rec_fourier_images,
                                   pupil_func=self.pupil_func,
                                   images=self.images,
                                   **step_args)

            PSI_est = kernel.solution_estimate(self.PSI, self.images)

            if object_steps is not None:
                do_object = (it % (object_steps + pupil_steps)) < object_steps
                do_pupil = not do_object
            else:
                do_object = do_pupil = True

            # ________ Object update ________
            if do_object:
                self.rec_fourier_images = kernel.update_object_fts(
                    PSI=PSI_est, geom=self.geom, pupil=self.pupil_func,
                    objectFT=self.rec_fourier_images)

            # ________ Pupil update ________
            if do_pupil:
                self.pupil_func = kernel.update_pupil(
                    PSI=PSI_est, geom=self.geom, objectFT=self.rec_fourier_images,
                    pupil_func=self.pupil_func, ctf=ctf)

                if normalise_pupil:
                    self.pupil_func, self.rec_fourier_images = normalise_pupil_scale(
                        self.pupil_func, self.rec_fourier_images, cover)

                if zernike_steps is not None and it % zernike_steps == 0:
                    amp, pha = np.abs(self.pupil_func), np.angle(self.pupil_func)
                    self.pupil_func = amp * np.exp(
                        1j * SquarePolynomials.project_wavefront(pha, coords=self.pupil_coords))

            # ________ Error ________
            err_tot, _ = kernel.compute_error([old_PSI], [self.PSI])
            self.iter_loss = err_tot
            self.losses.append(err_tot)

            misfit, _ = kernel.compute_data_misfit(self.geom, self.pupil_func,
                                                   self.rec_fourier_images, self.images)
            self.data_misfits.append(misfit)

            gap = kernel.compute_gap(self.PSI, self.geom, self.pupil_func,
                                     self.rec_fourier_images, self.images)
            self.gaps.append(gap)

            print(f"Iter {self.iters_passed}: step change = {err_tot:.4e}, "
                  f"data misfit = {misfit:.4e}, gap = {gap:.4e}")

            self.iters_passed += 1

            if live_plot and self.iters_passed % 5 == 0:
                self._update_live_plot()

    def pupil_cover_mask(self):
        """Bool mask of the pupil pixels the data constrains (inside the aperture)."""
        if getattr(self, "pupil_cover", None) is None:
            from .algorithms_3d import pupil_coverage
            self.pupil_cover = pupil_coverage(self.geom, self.pupil_func.shape) & (np.abs(self.ctf) > 0)
            print(f"Pupil coverage: {100 * self.pupil_cover.sum() / max((np.abs(self.ctf) > 0).sum(), 1):.1f}% "
                  f"of the aperture is constrained by the data")
        return self.pupil_cover

    def _post_process(self, save_gif=False):
        self.obj_3D = self.inverse_fft_3d(self.rec_fourier_images)
        self.rec_obj_images = self.obj_3D

    def get_state(self):
        return dict(objectFT=self.rec_fourier_images, pupil=self.pupil_func, PSI=self.PSI,
                    iters_passed=self.iters_passed, losses=self.losses,
                    data_misfits=self.data_misfits, gaps=self.gaps)

    def set_state(self, state: dict):
        self.rec_fourier_images = state['objectFT']
        self.pupil_func = state['pupil']
        self.PSI = state['PSI']
        self.iters_passed = state.get('iters_passed', 0)
        self.losses = state.get('losses', [])
        self.data_misfits = state.get('data_misfits', [])
        self.gaps = state.get('gaps', [])

    def GetKwArgs(self, obj, kwargs):
        import inspect
        sig = list(inspect.signature(obj).parameters.keys())
        return {k: v for k, v in kwargs.items() if k in sig}
