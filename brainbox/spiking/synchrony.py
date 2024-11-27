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


def compute_synchronization(
    spike_trains, n_pairs, dt=8, bin_dt=25, normalize=True, from_idxs=None, to_idxs=None
):
    if from_idxs is None and to_idxs is None:
        from_idxs, to_idxs = _generate_pairs(spike_trains.shape[1], n_pairs)
    else:
        from_idxs, to_idxs = _generate_pairs_from_idxs(n_pairs, from_idxs, to_idxs)
    binned_spikes = bin_spikes(spike_trains, dt, bin_dt, bin_dt)

    from_binned_spikes, to_binned_spikes = _generate_spike_pairing_tensors(
        binned_spikes, from_idxs, to_idxs
    )

    return _cross_covariance_matrix(
        from_binned_spikes, to_binned_spikes, normalize=normalize
    )


def _generate_spike_pairing_tensors(binned_spikes, from_idxs, to_idxs):
    # b x n x t
    from_binned_spikes = binned_spikes[:, from_idxs]
    to_binned_spikes = binned_spikes[:, to_idxs]

    return from_binned_spikes, to_binned_spikes


def _generate_pairs(n_neurons, n_pairs):
    from_idxs = torch.randint(0, n_neurons, (n_pairs,))
    to_idxs = torch.randint(0, n_neurons, (n_pairs,))

    # Avoid sampling identical idxs for pairing
    for i in range(n_pairs):
        if from_idxs[i] == to_idxs[i]:
            to_idxs[i] -= 1

    return from_idxs, to_idxs


def _generate_pairs_from_idxs(n_pairs, from_idxs, to_idxs):
    random_from_idxs = torch.randint(0, len(from_idxs), (n_pairs,))
    random_to_idxs = torch.randint(0, len(to_idxs), (n_pairs,))

    sampled_from_idxs = from_idxs[random_from_idxs].long()
    sampled_to_idxs = to_idxs[random_to_idxs].long()

    # Avoid sampling identical idxs for pairing
    for i in range(n_pairs):
        if sampled_from_idxs[i] == sampled_to_idxs[i]:
            sampled_to_idxs[i] = to_idxs[(random_to_idxs[i] + 1) % len(to_idxs)]

    return sampled_from_idxs, sampled_to_idxs


def _cross_covariance_matrix(x, y, normalize=True):
    x_min_mean = x - x.mean((0, 2)).view(1, -1, 1)
    y_min_mean = y - y.mean((0, 2)).view(1, -1, 1)

    cross_covariance_matrix_batch = torch.einsum(
        "bni, bnj -> bnij", x_min_mean, y_min_mean
    )
    cross_covariance_matrix_batch = cross_covariance_matrix_batch.mean(0)

    if normalize:
        inv_x_std = torch.nan_to_num(1 / x.std((0, 2), unbiased=False))
        inv_y_std = torch.nan_to_num(1 / y.std((0, 2), unbiased=False))
        return torch.einsum(
            "nij, n, n -> nij", cross_covariance_matrix_batch, inv_x_std, inv_y_std
        )

    return cross_covariance_matrix_batch
