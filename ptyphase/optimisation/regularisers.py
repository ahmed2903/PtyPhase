"""
Regulariser interface. A regulariser is a weighted penalty on the current model, added to
the data loss:

    total = loss_data + sum_r  w_r(epoch) * r.penalty(ctx)

`ctx` gives the spectrum S (Nz, n0, n1), the object obj = IFFT3c(S) [z, x, y] (computed
lazily, once per step, shared by all regularisers) and the pupil P. The weight is constant,
or follows a schedule with `get_alpha(epoch)` such as `ptyphase.schedulers.WeightScheduler`.

Subclass and implement `penalty`:

    class L1Object(Regulariser):
        name = "l1_obj"
        def penalty(self, ctx):
            return ctx.obj.abs().mean()
"""
from .forward import ifft3c


class RegContext:
    def __init__(self, S, P):
        self.S = S
        self.P = P
        self._obj = None

    @property
    def obj(self):
        if self._obj is None:
            self._obj = ifft3c(self.S)
        return self._obj


class Regulariser:
    name = "reg"

    def __init__(self, weight=1.0, scheduler=None):
        self.weight = weight
        self.scheduler = scheduler

    def weight_at(self, epoch):
        return self.scheduler.get_alpha(epoch) if self.scheduler is not None else self.weight

    def penalty(self, ctx):
        raise NotImplementedError
