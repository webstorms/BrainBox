import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidFitter:
    def __init__(
        self, model_output, amplitudes, phases, frequencies, offsets, n_iterations
    ):
        self._model_output = model_output  # n x t
        self._n_iterations = n_iterations

        (
            init_amplitudes,
            init_phases,
            init_frequencies,
            init_offsets,
        ) = SinusoidFitter._build_initial_params(
            amplitudes, phases, frequencies, offsets
        )
        self._init_amplitudes = init_amplitudes
        self._init_phases = init_phases
        self._init_frequencies = init_frequencies
        self._init_offsets = init_offsets

    def fit(self):
        def predicted_sinusoid():
            return amplitude * torch.sin(frequency * t + phase) + offset

        # Set correct dimensions on all variables
        n_units, t_len = self._model_output.shape
        n_params = len(self._init_amplitudes)
        amplitudes_init = self._init_amplitudes.view(-1, 1, 1).repeat(1, n_units, 1)
        phases_init = self._init_phases.view(-1, 1, 1).repeat(1, n_units, 1)
        frequencies_init = self._init_frequencies.view(-1, 1, 1).repeat(1, n_units, 1)
        # offset_init = self._model_output.mean(dim=1).view(1, -1, 1).repeat(n_params, 1, 1)
        offset_init = self._init_offsets.view(-1, 1, 1).repeat(1, n_units, 1)

        # Set trainable parameters
        t = torch.arange(t_len)
        amplitude = nn.Parameter(data=torch.Tensor(amplitudes_init), requires_grad=True)
        phase = nn.Parameter(data=torch.Tensor(phases_init), requires_grad=True)
        frequency = nn.Parameter(
            data=torch.Tensor(frequencies_init), requires_grad=True
        )
        offset = nn.Parameter(data=offset_init, requires_grad=True)

        model_output = self._model_output.unsqueeze(0).repeat(n_params, 1, 1)

        optimizer = torch.optim.SGD([amplitude, phase, frequency, offset], 10**-3)

        for _ in range(self._n_iterations):
            optimizer.zero_grad()
            # loss = F.mse_loss(torch.relu(predicted_sinusoid()), model_output)
            loss = F.mse_loss(predicted_sinusoid(), model_output)
            # torch.relu(predicted_sinusoid())
            # loss = (model_output - predicted_sinusoid()).abs().sum()
            loss.backward()
            optimizer.step()
            print(loss.item())

        return predicted_sinusoid(), amplitude, phase, frequency, offset

    @staticmethod
    def _build_initial_params(amplitudes, phases, frequencies, offsets):
        params_product = list(
            itertools.product(amplitudes, phases, frequencies, offsets)
        )
        amplitudes_init = torch.Tensor([v[0] for v in params_product])
        phases_init = torch.Tensor([v[1] for v in params_product])
        frequencies_init = torch.Tensor([v[2] for v in params_product])
        offsets_init = torch.Tensor([v[3] for v in params_product])

        return amplitudes_init, phases_init, frequencies_init, offsets_init
