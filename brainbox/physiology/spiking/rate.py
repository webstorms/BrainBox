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


def compute_firing_rates(spike_trains, dt, window_dt, stride_dt=None, pad_input=False):
    # spike_trains: b x n x t
    spike_counts = bin_spikes(spike_trains, dt, window_dt, stride_dt, pad_input)
    kernel_length = window_dt // dt
    actual_window_dt = kernel_length * dt
    firing_rates = spike_counts * (1000 / actual_window_dt)
    del spike_counts

    return firing_rates


def bin_spikes(spike_trains, dt, window_dt, stride_dt=None, pad_input=False):
    # spike_trains: b x n x t
    spike_trains = spike_trains.unsqueeze(1)
    kernel_length = window_dt // dt
    kernel = torch.ones(1, 1, 1, kernel_length).to(spike_trains.device)
    stride = (1, 1) if stride_dt is None else (1, stride_dt // dt)

    if pad_input:
        spike_trains = F.pad(spike_trains, (kernel_length - 1, 0))
    spike_counts = F.conv2d(spike_trains, kernel, stride=stride)

    return spike_counts.squeeze(1)
