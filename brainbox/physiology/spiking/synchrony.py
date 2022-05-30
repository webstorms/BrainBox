import torch

from .rate import bin_spikes
from brainbox.physiology import util


def compute_synchronization_for_dataset(
    input_to_spikes,
    n_neurons,
    n_pairs,
    dt,
    bin_dt,
    dataset,
    batch_size,
    device="cuda",
    dtype=torch.float,
    max_batches=None,
    **kwargs
):
    # Generate pairings
    from_idxs, to_idxs = generate_pairs(n_neurons, n_pairs)

    sample_binned_spikes = lambda data, _: compute_synchronization(
        input_to_spikes(data), from_idxs, to_idxs, dt, bin_dt
    )
    binned_spikes = util.run_function_on_batch(
        sample_binned_spikes, dataset, batch_size, device, dtype, max_batches, **kwargs
    )

    from_binned_spikes, to_binned_spikes = binned_spikes[:, 0], binned_spikes[:, 1]

    return util.cross_covariance_matrix(
        from_binned_spikes, to_binned_spikes, normalize=True
    )


def compute_synchronization(spike_trains, from_idxs, to_idxs, dt=8, bin_dt=25):
    binned_spikes = bin_spikes(spike_trains, dt, bin_dt, bin_dt)
    from_binned_spikes, to_binned_spikes = generate_spike_pairing_tensors(
        binned_spikes, from_idxs, to_idxs
    )

    return torch.stack([from_binned_spikes, to_binned_spikes]).permute(1, 0, 2, 3)


def generate_spike_pairing_tensors(binned_spikes, from_idxs, to_idxs):
    from_binned_spikes = binned_spikes[:, from_idxs]
    to_binned_spikes = binned_spikes[:, to_idxs]

    return from_binned_spikes, to_binned_spikes


def generate_pairs(n_neurons, n_pairs):
    from_idxs = torch.randint(0, n_neurons, (n_pairs,))
    to_idxs = torch.randint(0, n_neurons, (n_pairs,))

    return from_idxs, to_idxs
