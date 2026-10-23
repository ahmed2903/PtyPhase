import numpy as np

from .phase_abstract_ewald import PhaseRetrievalEwaldBase
from .runner import FourierPtychoEngine, Pipeline      # Pipeline works unchanged
from .algorithms_ewald import slices_to_object, shrinkwrap_support


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

        if shrinkwrap_steps is None:
            return FourierPtychoEngine.solve(self, kernel, iterations, **kwargs)

        if self.geom.support is None:
            self.geom.support = np.ones(self.geom.obj_shape, dtype=bool)

        done = 0
        while done < iterations:
            n = min(shrinkwrap_steps, iterations - done)
            FourierPtychoEngine.solve(self, kernel, n, **kwargs)
            done += n
            obj = slices_to_object(self.geom, self.rec_fourier_images)
            self.geom.support = shrinkwrap_support(obj, sw_sigma, sw_threshold)
            print(f"Shrinkwrap: support = {100 * self.geom.support.mean():.1f}% of the box")

    def _post_process(self, save_gif=False):
        super()._post_process(save_gif=save_gif)
        self.obj_3D = slices_to_object(self.geom, self.rec_fourier_images)
