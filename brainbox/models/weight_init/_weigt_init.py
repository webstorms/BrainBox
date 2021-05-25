import torch
import torch.nn as nn


class WeightInit:

    def __init__(self, name):
        self.name = name


class Constant(WeightInit):

    def __init__(self, name, c=0.42):
        super(Constant, self).__init__(name)
        self.c = c

    def __call__(self, weight):
        nn.init.constant_(weight, self.c)

    @property
    def hyperparams(self):
        return {'init_{0}'.format(self.name): {'name': 'constant', 'c': self.c}}


class Uniform(WeightInit):

    def __init__(self, name, a=0, b=1):
        super(Uniform, self).__init__(name)
        self.a = a
        self.b = b

    def __call__(self, weight):
        nn.init.uniform_(weight, self.a, self.b)

    @property
    def hyperparams(self):
        return {'init_{0}'.format(self.name): {'name': 'uniform', 'a': self.a, 'b': self.b}}


class GlorotUniform(WeightInit):

    def __init__(self, name):
        super(GlorotUniform, self).__init__(name)

    def __call__(self, weight):
        nn.init.xavier_uniform_(weight)

    @property
    def hyperparams(self):
        return {'init_{0}'.format(self.name): {'name': 'glorotuniform'}}


class Normal(WeightInit):

    def __init__(self, name, mean=0, std=1):
        super(Normal, self).__init__(name)
        self.mean = mean
        self.std = std

    def __call__(self, weight):
        nn.init.normal_(weight, mean=self.mean, std=self.std)

    @property
    def hyperparams(self):
        return {'init_{0}'.format(self.name): {'name': 'normal', 'mean': self.mean, 'std': self.std}}


class GlorotNormal(WeightInit):

    def __init__(self, name):
        super(GlorotNormal, self).__init__(name)

    def __call__(self, weight):
        nn.init.xavier_normal_(weight)

    @property
    def hyperparams(self):
        return {'init_{0}'.format(self.name): {'name': 'glorotnormal'}}


class Identity(WeightInit):

    def __init__(self, name, c=1):
        super(Identity, self).__init__(name)
        self.c = c

    def __call__(self, weight):
        nn.init.eye_(weight)
        with torch.no_grad():
            weight *= self.c

    @property
    def hyperparams(self):
        return {'init_{0}'.format(self.name): {'name': 'identity', 'c': self.c}}

