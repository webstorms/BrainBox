import sys
import logging

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("util")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def normalize_rfs(rfs):
    n_dim, h_dim, w_dim = rfs.shape
    max_values = rfs.abs().amax(dim=(1, 2))
    max_values = torch.clamp(max_values, min=1e-5, max=np.inf)
    max_values = max_values.repeat(h_dim, w_dim, 1).permute(2, 0, 1)

    return rfs / max_values


def normalize_strfs(strf):
    n_dim, t_dim, h_dim, w_dim = strf.shape
    max_values = strf.abs().amax(dim=(1, 2, 3))
    max_values = torch.clamp(max_values, min=1e-5, max=np.inf)
    max_values = max_values.repeat(h_dim, w_dim, t_dim, 1).permute(3, 2, 0, 1)

    return strf / max_values


def get_highest_power_spatial_rf(spatiotemporal_rf):
    """
    Obtain the spatial RF from the spatiotemporal RF with the largest mean power. The mean power is calculated as the
    mean over both spatial dimensions of the squared values of the spatial RFs at every time step.

    :param spatiotemporal_rf: A tensor of shape rf_len x rf_w x rf_h
    :return: A tensor of shape  rf_w x rf_h
    """

    power_at_timesteps = torch.pow(spatiotemporal_rf, 2).mean(dim=(1, 2))
    t = power_at_timesteps.argmax().item()
    spatial_rf = spatiotemporal_rf[t]

    return spatial_rf


def get_all_highest_power_spatial_rf(spatiotemporal_rfs):
    """
    Obtain the spatial RFs with largest mean power from the provided set of spatiotemporal RFs. The mean power is
    calculated as the mean over both spatial dimensions of the squared values of the spatial RFs at every time step.

    :param spatiotemporal_rfs: A tensor of shape n_units x rf_len x rf_w x rf_h
    :return: A tensor of shape n_units x rf_w x rf_h
    """

    rfs = []

    for i in range(len(spatiotemporal_rfs)):
        spatial_rf = get_highest_power_spatial_rf(
            spatiotemporal_rfs[i].detach().cpu().float()
        )
        rfs.append(spatial_rf)
    rfs = torch.stack(rfs)

    return rfs


def get_temporal_power_profile(spatiotemporal_rfs):
    power_profile = (
        torch.pow(spatiotemporal_rfs, 2).mean(dim=(0, 2, 3)).detach().cpu().float()
    )
    power_profile = power_profile / power_profile.sum()

    return power_profile


def sta(
    input_to_spikes,
    rf_len,
    rf_h,
    rf_w,
    t_len,
    noise_var=10,
    samples=10,
    batch_size=2000,
    device="cuda",
):

    model_rfs = 0

    for i in range(samples // batch_size):
        logger.info(f"Processing batch {i} out of {samples // batch_size}...")
        noise = torch.normal(
            0, noise_var, (min(batch_size, samples - i * batch_size), t_len, rf_h, rf_w)
        ).to(device)
        model_output = input_to_spikes(noise)

        off = noise.shape[1] - model_output.shape[2]
        model_output = F.pad(model_output, (off, 0))
        count = 0

        for t in range(rf_len - 1, t_len):
            _noise = noise[:, t - rf_len + 1 : t + 1]
            _model_output = model_output[:, :, t]
            batch_model_rfs = torch.einsum("bthw, bn->bnthw", _noise, _model_output)
            count += batch_model_rfs.shape[0]
            model_rfs += batch_model_rfs.sum(dim=0)

    model_rfs = model_rfs / count
    model_rfs = model_rfs.detach().cpu()

    return model_rfs
