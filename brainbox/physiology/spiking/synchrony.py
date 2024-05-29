import torch
import pandas as pd

from .rate import bin_spikes


def compute_synchronization_df(cross_covariance_matrix, dt=25):

    def time_pair_to_lag(i, j, dt):
        return (i - j) * dt

    cross_covariance_matrix = cross_covariance_matrix.cpu().detach()
    t_dim = cross_covariance_matrix.shape[0]

    synchrony = []

    for i in range(t_dim):
        for j in range(t_dim):
            lag = time_pair_to_lag(i, j, dt)
            correlation = cross_covariance_matrix[i, j].item()
            synchrony.append({"lag": lag, "correlation": correlation})

    return pd.DataFrame(synchrony)


def compute_synchronization(spike_trains, n_pairs, dt=8, bin_dt=25):
    from_idxs, to_idxs = generate_pairs(spike_trains.shape[1], n_pairs)

    binned_spikes = bin_spikes(spike_trains, dt, bin_dt, bin_dt)

    from_binned_spikes, to_binned_spikes = generate_spike_pairing_tensors(
        binned_spikes, from_idxs, to_idxs
    )

    return cross_covariance_matrix(from_binned_spikes, to_binned_spikes)


def generate_spike_pairing_tensors(binned_spikes, from_idxs, to_idxs):
    # b x n x t
    from_binned_spikes = binned_spikes[:, from_idxs]
    to_binned_spikes = binned_spikes[:, to_idxs]

    return from_binned_spikes, to_binned_spikes


def generate_pairs(n_neurons, n_pairs):
    from_idxs = torch.randint(0, n_neurons, (n_pairs,))
    to_idxs = torch.randint(0, n_neurons, (n_pairs,))

    # Avoid sampling identical idxs for pairing
    for i in range(n_pairs):
        if from_idxs[i] == to_idxs[i]:
            print(from_idxs[i], to_idxs[i])
            to_idxs[i] += 1

    return from_idxs, to_idxs


def cross_covariance_matrix(x, y, normalize=True):
    x_min_mean = x - x.mean(0)
    y_min_mean = y - y.mean(0)

    cross_covariance_matrix_batch = torch.einsum(
        "bni, bnj -> bnij", x_min_mean, y_min_mean
    )
    cross_covariance_matrix_batch = cross_covariance_matrix_batch.mean(0)

    if normalize:
        inv_x_std = torch.nan_to_num(1 / x.std(0, unbiased=False))
        inv_y_std = torch.nan_to_num(1 / y.std(0, unbiased=False))
        return torch.einsum(
            "nij, ni, nj -> nij", cross_covariance_matrix_batch, inv_x_std, inv_y_std
        )

    return cross_covariance_matrix_batch
