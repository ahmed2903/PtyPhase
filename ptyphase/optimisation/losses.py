"""
Data-fidelity losses. Each takes the predicted and measured intensities of a batch of images
and a normaliser fixed over the whole data set, so that summing the batch losses of one
epoch gives the full-data loss.

    l2_intensity : sum (I - I_meas)^2         / sum I_meas^2
    l2_amplitude : sum (sqrt I - sqrt I_meas)^2 / sum I_meas      (= the projection engines'
                                                                    data misfit)

Optional per-pixel weights w (same shape as I, 0 = pixel excluded): every sum, including the
normaliser, runs over w * (...), so masked pixels neither pull the model nor count in the scale.
"""
import torch

_EPS = 1e-12


def _wsum(x, w):
    return torch.sum(x) if w is None else torch.sum(w * x)


def l2_intensity(I_pred, I_meas, norm, w=None):
    return _wsum((I_pred - I_meas)**2, w) / norm


def l2_amplitude(I_pred, I_meas, norm, w=None):
    return _wsum((torch.sqrt(I_pred + _EPS) - torch.sqrt(I_meas))**2, w) / norm


LOSSES = {
    "l2_intensity": (l2_intensity, lambda I, w=None: _wsum(I**2, w)),
    "l2_amplitude": (l2_amplitude, lambda I, w=None: _wsum(I, w)),
}


def get_loss(name):
    """(loss_fn, normaliser_fn(I_meas_all))"""
    if name not in LOSSES:
        raise ValueError(f"loss must be one of {list(LOSSES)}, got '{name}'")
    return LOSSES[name]
