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
        return {'init_{0}_constant'.format(self.name): {'c': self.c}}


class Uniform(WeightInit):

    def __init__(self, name, a=0, b=1):
        super(Uniform, self).__init__(name)
        self.a = a
        self.b = b

    def __call__(self, weight):
        nn.init.uniform_(weight, self.a, self.b)

    @property
    def hyperparams(self):
        return {'init_{0}_uniform'.format(self.name): {'a': self.a, 'b': self.b}}


class GlorotUniform(WeightInit):

    def __init__(self, name, a=0, b=1):
        super(GlorotUniform, self).__init__(name)
        self.a = a
        self.b = b

    def __call__(self, weight):
        nn.init.xavier_uniform_(weight, self.a, self.b)

    @property
    def hyperparams(self):
        return {'init_{0}_glorotuniform'.format(self.name): {'a': self.a, 'b': self.b}}


class Normal(WeightInit):

    def __init__(self, name, mean=0, std=1):
        super(Normal, self).__init__(name)
        self.mean = mean
        self.std = std

    def __call__(self, weight):
        nn.init.normal_(weight, mean=self.mean, std=self.std)

    @property
    def hyperparams(self):
        return {'init_{0}_normal'.format(self.name): {'mean': self.mean, 'std': self.std}}


class GlorotNormal(WeightInit):

    def __init__(self, name):
        super(GlorotNormal, self).__init__(name)

    def __call__(self, weight):
        nn.init.xavier_normal_(weight)

    @property
    def hyperparams(self):
        return {'init_{0}_glorotnormal'.format(self.name): {}}

# TODO: Gamma

# TODO: Identity

