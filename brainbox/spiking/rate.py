import torch
import torch.nn.functional as F
import numpy as np


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

    kernel = kernel / kernel.sum()  # Normalize the kernel

    if pad_input:
        spike_trains = F.pad(spike_trains, (kernel_length - 1, 0))
    spike_counts = F.conv2d(spike_trains, kernel, stride=stride)

    return spike_counts.squeeze(1)
