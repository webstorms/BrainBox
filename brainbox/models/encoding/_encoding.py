import torch.nn as nn

from .._model import BBModel


class SpatioTemporalLinearModel(BBModel):

    def __init__(self, rf_len, in_channels, n_neurons, width, height, bias=False):
        super().__init__()
        self.rf_len = rf_len
        self.in_channels = in_channels
        self.n_neurons = n_neurons
        self.width = width
        self.height = height
        self.bias = bias
        self.model = nn.Conv3d(in_channels, n_neurons, (rf_len, self.height, self.width), bias=bias)

    @property
    def hyperparams(self):
        return {**super().hyperparams, 'rf_len': self.rf_len, 'in_channels': self.in_channels, 'n_neurons': self.n_neurons,
                'width': self.width, 'height': self.height, 'bias': self.bias}

    def forward(self, x):
        # x: batch x c x t x h x w
        x = self.model(x)
        n_batch, n_chanel, n_timesteps, h, w = x.shape
        assert h == 1 and w == 1, 'Spatial dimensions too large.'

        # ensure output is of shape b x 1 x t x n_neurons
        x = x.view(n_batch, n_chanel, n_timesteps, 1)
        x = x.permute(0, 3, 2, 1)

        return x


class SpatioTemporalLNModel(SpatioTemporalLinearModel):

    def __init__(self, nonlinearity, rf_len, in_channels, n_neurons, width, height, bias=False):
        super().__init__(rf_len, in_channels, n_neurons, width, height, bias)
        self.nonlinearity = nonlinearity

    @property
    def hyperparams(self):
        return {**super().hyperparams, 'nonlinearity': self.nonlinearity.__name__}

    def forward(self, x):
        # x: batch x c x t x h x w
        x = super(SpatioTemporalLNModel, self).forward(x)

        return self.nonlinearity(x)


class SpatialLinearModel(BBModel):

    def __init__(self, in_channels, n_neurons, width, height, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.n_neurons = n_neurons
        self.width = width
        self.height = height
        self.bias = bias
        self.model = nn.Conv2d(in_channels, n_neurons, (self.height, self.width), bias=bias)

    @property
    def hyperparams(self):
        return {**super().hyperparams, 'in_channels': self.in_channels, 'n_neurons': self.n_neurons,
                'width': self.width, 'height': self.height, 'bias': self.bias}

    def forward(self, x):
        # x: batch x c x h x w
        x = self.model(x)
        n_batch, n_chanel, h, w = x.shape
        assert h == 1 and w == 1, 'Spatial dimensions too large.'

        # ensure output is of shape b x 1 x n_neurons
        x = x.view(n_batch, n_chanel, 1)
        x = x.permute(0, 2, 1)

        return x


class SpatialLNModel(SpatialLinearModel):

    def __init__(self, nonlinearity, in_channels, n_neurons, width, height, bias=False):
        super().__init__(in_channels, n_neurons, width, height, bias)
        self.nonlinearity = nonlinearity

    @property
    def hyperparams(self):
        return {**super().hyperparams, 'nonlinearity': self.nonlinearity.__name__}

    def forward(self, x):
        # x: batch x c x h x w
        x = super(SpatialLNModel, self).forward(x)

        return self.nonlinearity(x)

