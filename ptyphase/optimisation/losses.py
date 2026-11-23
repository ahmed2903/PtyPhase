"""
Data-fidelity losses. Each takes the predicted and measured intensities of a batch of images
and a normaliser fixed over the whole data set, so that summing the batch losses of one
epoch gives the full-data loss.

    l2_intensity : sum (I - I_meas)^2         / sum I_meas^2
    l2_amplitude : sum (sqrt I - sqrt I_meas)^2 / sum I_meas      (= the projection engines'
                                                                    data misfit)
"""
import torch

_EPS = 1e-12


def l2_intensity(I_pred, I_meas, norm):
    return torch.sum((I_pred - I_meas)**2) / norm


def l2_amplitude(I_pred, I_meas, norm):
    return torch.sum((torch.sqrt(I_pred + _EPS) - torch.sqrt(I_meas))**2) / norm


LOSSES = {
    "l2_intensity": (l2_intensity, lambda I: torch.sum(I**2)),
    "l2_amplitude": (l2_amplitude, lambda I: torch.sum(I)),
}


def get_loss(name):
    """(loss_fn, normaliser_fn(I_meas_all))"""
    if name not in LOSSES:
        raise ValueError(f"loss must be one of {list(LOSSES)}, got '{name}'")
    return LOSSES[name]
