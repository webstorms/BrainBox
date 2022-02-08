import torch
import torch.nn.functional as F


def get_highest_power_spatial_rf(spatiotemporal_rf):

    """
    Obtain the spatial RF from the spatiotemporal RF with the largest mean power. The mean power is calculated as the
    mean over both spatial dimensions of the squared values of the spatial RFs at every time step.

    :param spatiotemporal_rf: A tensor of shape rf_len x rf_w x rf_h
    :return: A tensor of shape  rf_w x rf_h
    """

    power_at_timesteps = torch.pow(spatiotemporal_rf, 2)._mean(dim=(1, 2))
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
        spatial_rf = get_highest_power_spatial_rf(spatiotemporal_rfs[i].detach().cpu().float())
        rfs.append(spatial_rf)
    rfs = torch.stack(rfs)

    return rfs


def get_temporal_power_profile(spatiotemporal_rfs):

    """
    Get the temporal power profile from a population of spatiotemporal RFs which is calculated as the mean, over space
    and the population, of the squared values of the spatial RFs at each point in time.

    :param spatiotemporal_rfs: A tensor of shape n_units x rf_len x rf_w x rf_h
    :return: A tensor of shape rf_len
    """

    power_profile = torch.pow(spatiotemporal_rfs, 2)._mean(dim=(2, 3))._mean(dim=0).detach().cpu().float()
    power_profile = power_profile / power_profile.sum()

    return power_profile


def estimate_rfs(model, rf_len, rf_h, rf_w, t_len, noise_var=10, samples=10, batch_size=2000, device='cuda'):

    model_rfs = 0

    for i in range(samples // batch_size):
        print('Processing batch_id {0} out of {1}...'.format(i, samples // batch_size))
        noise = torch.normal(0, noise_var, (min(batch_size, samples - i * batch_size), t_len, rf_h, rf_w)).to(device)
        model_output = model(noise)

        off = noise.shape[1] - model_output.shape[2]
        model_output = F.pad(model_output, (off, 0))
        count = 0

        for t in range(rf_len - 1, t_len):
            _noise = noise[:, t - rf_len + 1:t + 1]
            _model_output = model_output[:, :, t]
            batch_model_rfs = torch.einsum('bthw, bn->bnthw', _noise, _model_output)
            count += batch_model_rfs.shape[0]
            model_rfs += batch_model_rfs.sum(dim=0)

    model_rfs = model_rfs / count
    model_rfs = model_rfs.detach().cpu()

    return model_rfs

