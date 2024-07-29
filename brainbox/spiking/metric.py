import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VanRossum(nn.Module):
    def __init__(self, len, tau=20, dt=1):
        super().__init__()
        self.len = len
        self.tau = tau
        self.dt = dt
        self.kernel = nn.Parameter(
            torch.Tensor([np.exp(-t / tau) for t in range(len - 1, -1, -1)]).view(
                1, 1, -1
            ),
            requires_grad=False,
        )

    def forward(self, input, target):
        diff_vec = input - target
        diff_vec = F.pad(diff_vec, (self.len - 1, 0))
        diff_vec = F.conv1d(diff_vec, self.kernel)

        return torch.sqrt((1 / self.tau) * torch.pow(diff_vec, 2).sum() * self.dt)


class SpikeToPSTH(nn.Module):
    def __init__(self, len, sig=20):
        super().__init__()
        self.len = len

        start_idx = -int(len / 2)
        end_idx = int(len / 2) + (0 if len % 2 == 0 else 1)
        self.kernel = nn.Parameter(
            torch.Tensor(
                [SpikeToPSTH.gaus(t, sig) for t in range(start_idx, end_idx)]
            ).view(1, 1, -1),
            requires_grad=False,
        )

    def forward(self, input):
        input = F.pad(input, (self.len - 1, self.len - 1))
        input = F.conv1d(input, self.kernel)
        input = input[:, :, int(self.len / 2) : -int(self.len / 2) + 1]

        return input

    @staticmethod
    def gaus(x, sig):
        return (1 / (sig * (2 * np.pi) ** 2)) * np.exp(-0.5 * (x / sig) ** 2)
