import torch
import torch.nn.functional as F

from brainbox.physiology import util


def compute_firing_rates_for_dataset(
    input_to_spikes,
    dt,
    window_dt,
    stride_dt,
    pad_input,
    dataset,
    batch_size,
    device="cuda",
    dtype=torch.float,
    max_batches=None,
    **kwargs
):
    get_firing_rates = lambda data, _: compute_firing_rates(
        input_to_spikes(data), dt, window_dt, stride_dt, pad_input
    )
    return util.run_function_on_batch(
        get_firing_rates, dataset, batch_size, device, dtype, max_batches, **kwargs
    )


def count_bursts(spike_trains, dt, burst_size, stride_dt=None, pad_input=False):
    # spike_trains: b x n x t
    burst_window_dt = burst_size * dt
    spike_counts = bin_spikes(spike_trains, dt, burst_window_dt, stride_dt, pad_input)
    burst_counts = (spike_counts == burst_size).sum(dim=2)

    window_dt = spike_trains.shape[2] * dt
    burst_frequency = burst_counts * (1000 / window_dt)

    return burst_frequency
