import torch.nn as nn

from .._model import BBModel


class LinearModel(BBModel):

    def __init__(self, rf_len, in_channels, n_neurons, width, height, bias=False):
        super().__init__()
        self.rf_len = rf_len
        self.in_channels = in_channels
        self.width = width
        self.height = height
        self.model = nn.Conv3d(in_channels, n_neurons, (rf_len, self.height, self.width), bias=bias)

    @property
    def hyperparams(self):
        return {**super().hyperparams, 'rf_len': self.rf_len, 'in_channels': self.in_channels, 'width': self.width,
                'height': self.height}

    def forward(self, x):
        # x: batch x c x t x h x w
        return self.model(x)[..., 0, 0]


class LNModel(LinearModel):

    def __init__(self, nonlinearity, rf_len, in_channels, n_neurons, width, height, bias=False):
        super().__init__(rf_len, in_channels, n_neurons, width, height, bias)
        self.nonlinearity = nonlinearity

    @property
    def hyperparams(self):
        return {**super().hyperparams, 'nonlinearity': self.nonlinearity.__name__}

    def forward(self, x):
        # x: batch x c x t x h x w
        return self.nonlinearity(self.model(x)[..., 0, 0])
