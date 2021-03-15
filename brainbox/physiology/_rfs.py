import torch


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

    power_profile = torch.pow(spatiotemporal_rfs, 2).mean(dim=(2, 3)).mean(dim=0).detach().cpu().float()
    power_profile = power_profile / power_profile.sum()

    return power_profile

