import torch

from .rate import bin_spikes
from brainbox.physiology import util


def compute_synchronization_for_dataset(input_to_spikes, n_pairs, dt, bin_dt, window_dt, stride_dt, dataset, batch_size, device='cuda', dtype=torch.float, max_batches=None, **kwargs):
    get_synchronization = lambda data, _: compute_synchronization(input_to_spikes(data), n_pairs, dt, bin_dt, window_dt, stride_dt)
    return util.run_function_on_batch(get_synchronization, dataset, batch_size, device, dtype, max_batches, **kwargs)


def compute_synchronization(spike_trains, n_pairs, dt=8, bin_dt=25, window_dt=2000, stride_dt=200):
    # Bin spikes
    binned_spikes = bin_spikes(spike_trains, dt, bin_dt, bin_dt)

    # Generate pairing tensors
    from_binned_spikes, to_binned_spikes = generate_spike_pairing_tensors(binned_spikes, n_pairs, dt*(bin_dt//dt), window_dt, stride_dt)
    del spike_trains
    del binned_spikes

    # Compute cross covariance matrix
    cross_covariance_matrix = util.cross_covariance_matrix(from_binned_spikes, to_binned_spikes, normalize=True)
    del from_binned_spikes
    del to_binned_spikes

    return cross_covariance_matrix


def generate_spike_pairing_tensors(binned_spikes, n_pairs, dt, window_dt, stride_dt):
    # Generate pairings
    n_neurons = binned_spikes.shape[1]
    from_idxs, to_idxs = generate_pairs(n_neurons, n_pairs)

    # Create pairing tensors
    from_binned_spikes = util.batchify_time_dimension(binned_spikes[:, from_idxs], dt, window_dt, stride_dt)
    to_binned_spikes = util.batchify_time_dimension(binned_spikes[:, to_idxs], dt, window_dt, stride_dt)

    return from_binned_spikes, to_binned_spikes


def generate_pairs(n_neurons, n_pairs):
    from_idxs = torch.randint(0, n_neurons, (n_pairs,))
    to_idxs = torch.randint(0, n_neurons, (n_pairs,))

    return from_idxs, to_idxs
