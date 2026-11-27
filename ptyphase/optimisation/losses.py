"""
Data-fidelity losses. Each takes the predicted and measured intensities of a batch of images
and a normaliser fixed over the whole data set, so that summing the batch losses of one
epoch gives the full-data loss.

    l2_intensity : sum (I - I_meas)^2         / sum I_meas^2
    l2_amplitude : sum (sqrt I - sqrt I_meas)^2 / sum I_meas      (= the projection engines'
                                                                    data misfit)
    poisson      : sum (I - I_meas + I_meas log(I_meas / I))  / sum I_meas
                   Poisson negative log-likelihood up to a data-only constant (generalised KL /
                   half the Poisson deviance): 0 at a perfect fit, every term >= 0, and
                   0 log 0 = 0, so zero-count pixels just ask for I -> 0. Correct weighting for
                   photon-counting data (each pixel's error relative to its own shot noise).

Optional per-pixel weights w (same shape as I, 0 = pixel excluded): every sum, including the
normaliser, runs over w * (...), so masked pixels neither pull the model nor count in the scale.

per_image=True returns the loss of every image separately, shape (B,) (the sum over the last two
axes only, same normaliser), so that their sum is the batch loss.
"""
import torch

_EPS = 1e-12


def _wsum(x, w, per_image=False):
    if w is not None:
        x = w * x
    return torch.sum(x, dim=(-2, -1)) if per_image else torch.sum(x)


def l2_intensity(I_pred, I_meas, norm, w=None, per_image=False):
    return _wsum((I_pred - I_meas)**2, w, per_image) / norm


def l2_amplitude(I_pred, I_meas, norm, w=None, per_image=False):
    return _wsum((torch.sqrt(I_pred + _EPS) - torch.sqrt(I_meas))**2, w, per_image) / norm


_PEPS = 1e-9          # keeps log(I) finite where the model is exactly dark (images have mean 1)


def poisson(I_pred, I_meas, norm, w=None, per_image=False):
    Ip = I_pred + _PEPS
    # I_meas log I_meas is a data-only constant (0 log 0 = 0); kept separate so the gradient
    # is exactly 1 - I_meas / I, finite at zero-count pixels
    return _wsum(Ip - I_meas + torch.xlogy(I_meas, I_meas) - I_meas * torch.log(Ip), w, per_image) / norm


LOSSES = {
    "l2_intensity": (l2_intensity, lambda I, w=None: _wsum(I**2, w)),
    "l2_amplitude": (l2_amplitude, lambda I, w=None: _wsum(I, w)),
    "poisson": (poisson, lambda I, w=None: _wsum(I, w)),
}


def get_loss(name):
    """(loss_fn, normaliser_fn(I_meas_all))"""
    if name not in LOSSES:
        raise ValueError(f"loss must be one of {list(LOSSES)}, got '{name}'")
    return LOSSES[name]
