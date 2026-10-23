import numpy as np

from .phase_abstract_ewald import PhaseRetrievalEwaldBase
from .runner import FourierPtychoEngine, Pipeline      # Pipeline works unchanged
from .algorithms_ewald import slices_to_object, shrinkwrap_support, DnCEwald


class FourierPtychoEwaldEngine(PhaseRetrievalEwaldBase, FourierPtychoEngine):
    """Per-streak engine with curved Ewald slices and a 3D real-space support:

        engine = FourierPtychoEwaldEngine(images=..., pupil_func=..., kins=..., ks_pupil=...,
                                          lr_psize=..., wavelength=lam_nm, dqz=roi.delta_qz)
        engine.prepare(extend='triple', curvature=True, support=None)
        Pipeline(engine, [(DMEwald(beta=0.9), 200), (EREwald(), 50)]).run(shrinkwrap_steps=20)
        engine.obj_3D        # [z, x, y]

    Use the *Ewald kernels from algorithms_ewald; plain 2D kernels run without the support.
    """

    def solve(self, kernel, iterations, shrinkwrap_steps=None, sw_sigma=1.0, sw_threshold=0.1,
              **kwargs):
        """
        shrinkwrap_steps : every this many iterations replace the support by the blurred
            |obj_3D| thresholded at sw_threshold * max (sigma sw_sigma voxels). If no
            support was given, start from the full box. None -> support stays fixed.
        Other arguments as FourierPtychoEngine.solve. With object_steps/pupil_steps,
        keep shrinkwrap_steps a multiple of object_steps + pupil_steps (the cycle
        restarts at every chunk).
        """
        kernel.geom = self.geom
        if isinstance(kernel, DnCEwald):
            run = self._solve_dnc
        else:
            run = lambda k, n, **kw: FourierPtychoEngine.solve(self, k, n, **kw)

        if shrinkwrap_steps is None:
            return run(kernel, iterations, **kwargs)

        if self.geom.support is None:
            self.geom.support = np.ones(self.geom.obj_shape, dtype=bool)

        done = 0
        while done < iterations:
            n = min(shrinkwrap_steps, iterations - done)
            run(kernel, n, **kwargs)
            done += n
            obj = slices_to_object(self.geom, self.rec_fourier_images)
            self.geom.support = shrinkwrap_support(obj, sw_sigma, sw_threshold)
            print(f"Shrinkwrap: support = {100 * self.geom.support.mean():.1f}% of the box")

    def _solve_dnc(self, kernel, iterations, object_steps=None, pupil_steps=None,
                   zernike_steps=None, live_plot=False, constrain_pupil=True,
                   normalise_pupil=True, **kwargs):
        """Loop for DnCEwald. The kernel keeps its replicas between iterations; the
        engine holds the concurred estimate (PSI, F) and updates the pupil from it.
        F moves every iteration; object_steps/pupil_steps only gate the pupil update
        (pupil_steps=0 keeps it fixed)."""
        if live_plot or zernike_steps is not None:
            print("DnCEwald: live_plot and zernike_steps are not supported, ignoring them")

        if kernel.x is None or kernel._iter != self.iters_passed:
            # match the energy of the start to the data (Parseval: ||PSI||^2 = n_pix sum I);
            # concur only averages replicas, so a badly scaled start takes long to recover
            s = np.sqrt(self.images[0].size * self.images.sum()) / np.linalg.norm(self.PSI)
            if np.isfinite(s) and s > 0:
                self.PSI = self.PSI * s
                self.rec_fourier_images = self.rec_fourier_images * s
            kernel.reset(self.PSI, self.rec_fourier_images)

        ctf = self.ctf if constrain_pupil else None
        cover = self.pupil_cover_mask()

        for it in range(iterations):

            old_PSI = self.PSI

            # ________ DnC step ________
            self.PSI, self.rec_fourier_images = kernel.step(
                pupil_func=self.pupil_func,
                pupil_slices=self.pupil_slices,
                object_slices=self.object_slices,
                images=self.images)

            # ________ Pupil update ________
            if object_steps is None or (it % (object_steps + pupil_steps)) >= object_steps:
                self.pupil_func = kernel.update_pupil(
                    PSI=self.PSI,
                    objectFT=self.rec_fourier_images,
                    pupil_slices=self.pupil_slices,
                    pupil_func=self.pupil_func,
                    object_slices=self.object_slices,
                    ctf=ctf)

                if normalise_pupil:
                    c = np.sqrt(np.mean(np.abs(self.pupil_func[cover])**2))
                    if np.isfinite(c) and c > 0:
                        self.pupil_func = self.pupil_func / c
                        self.rec_fourier_images = self.rec_fourier_images * c
                        kernel.rescale(c)

            # ________ Error ________
            err_tot, _ = kernel.compute_error(old_PSI, self.PSI)
            self.iter_loss = err_tot
            self.losses.append(err_tot)

            misfit_tot, _ = kernel.compute_data_misfit(
                pupil_slices=self.pupil_slices,
                object_slices=self.object_slices,
                pupil=self.pupil_func,
                objectFTs=self.rec_fourier_images,
                images=self.images)
            self.data_misfits.append(misfit_tot)
            self.gaps.append(kernel.residual)

            print(f"Iter {self.iters_passed}: step change = {err_tot:.4e}, "
                  f"data misfit = {misfit_tot:.4e}, DnC residual = {kernel.residual:.4e}")

            self.iters_passed += 1
            kernel._iter = self.iters_passed

    def _post_process(self, save_gif=False):
        super()._post_process(save_gif=save_gif)
        self.obj_3D = slices_to_object(self.geom, self.rec_fourier_images)
