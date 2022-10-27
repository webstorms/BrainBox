import torch
import torch.nn.functional as F
import numpy as np

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
    kernel_length = int(window_dt // dt)
    actual_window_dt = int(kernel_length * dt)
    firing_rates = spike_counts * (1000 / actual_window_dt)
    del spike_counts

    return firing_rates


def bin_spikes(
    spike_trains,
    dt,
    window_dt,
    stride_dt=None,
    pad_input=False,
    gaussian=False,
    sigma=1,
):
    def get_guassian_kernel(sigma, width):
        kernel = [
            (1 / ((np.sqrt(2 * np.pi) * sigma)) * np.exp(-(x**2) / (2 * sigma**2)))
            for x in np.arange(-width // 2 + 1, width // 2 + 1, 1)
        ]
        return torch.Tensor(kernel)

    # spike_trains: b x n x t
    spike_trains = spike_trains.unsqueeze(1)
    kernel_length = int(window_dt // dt)
    kernel = torch.ones(1, 1, 1, kernel_length).to(spike_trains.device)

    if gaussian:
        kernel[0, 0, 0, :] = get_guassian_kernel(sigma, kernel_length)

    stride = (1, 1) if stride_dt is None else (1, int(stride_dt // dt))

    if pad_input:
        spike_trains = F.pad(spike_trains, (kernel_length - 1, 0))
    spike_counts = F.conv2d(spike_trains, kernel, stride=stride)

    return spike_counts.squeeze(1)
