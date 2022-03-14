import torch

from brainbox.physiology import util


def compute_isi_cvs_for_dataset(input_to_spikes, bin_dt, window_dt, stride_dt, n_spikes_thresh, dataset, batch_size, device='cuda', dtype=torch.float, max_batches=None, on_cpu=False, **kwargs):
    get_isi_cvs = lambda data, _: compute_isi_cvs(compute_isis_tensor(util.batchify_time_dimension(input_to_spikes(data), bin_dt, window_dt, stride_dt)), n_spikes_thresh, on_cpu)
    return util.run_function_on_batch(get_isi_cvs, dataset, batch_size, device, dtype, max_batches, **kwargs)


def compute_isi_cvs(isis_tensor, n_spikes_thresh, on_cpu):
    isis_tensor = isis_tensor.to("cpu" if on_cpu else isis_tensor.device)
    mean = util.get_mean(isis_tensor)
    std = util.get_std(isis_tensor, mean=None)
    cvs = std / mean

    invalid_cv_mask = (isis_tensor > 0).sum(dim=2) < n_spikes_thresh
    cvs[invalid_cv_mask] = 0

    return cvs


def compute_isis_tensor(spike_trains):
    # spike_trains: b x n x t
    spike_trains = spike_trains.type(torch.int16)
    b_dim, n_dim, t_dim = spike_trains.shape
    t_of_last_spike = 32767 * torch.ones(b_dim, n_dim).to(spike_trains.device).type(torch.int16)
    isis = []

    for t in range(t_dim):
        spike_mask = spike_trains[:, :, t] > 0

        isi = torch.zeros(b_dim, n_dim).to(spike_trains.device).type(torch.int16)
        isi[spike_mask] = t - t_of_last_spike[spike_mask]

        t_of_last_spike[spike_mask] = t
        isis.append(isi)

    isis_tensor = torch.stack(isis, dim=-1)
    isis_tensor[isis_tensor < 0] = 0  # Remove first spike occurrence

    return isis_tensor
