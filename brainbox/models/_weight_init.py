import torch
import torch.nn as nn


class WeightInit:
    def __init__(self, weight_name):
        self.weight_name = weight_name

    @property
    def hyperparams(self):
        hyperparams = {"name": self.__class__.__name__, "weight_name": self.weight_name}

        return hyperparams


class Constant(WeightInit):
    def __init__(self, weight_name, c=0.42):
        super(Constant, self).__init__(weight_name)
        self.c = c

    def __call__(self, weight):
        nn.init.constant_(weight, self.c)

    @property
    def hyperparams(self):
        hyperparams = {**super().hyperparams, "c": self.c}

        return hyperparams


class Uniform(WeightInit):
    def __init__(self, weight_name, a=0, b=1):
        super(Uniform, self).__init__(weight_name)
        self.a = a
        self.b = b

    def __call__(self, weight):
        nn.init.uniform_(weight, self.a, self.b)

    @property
    def hyperparams(self):
        hyperparams = {**super().hyperparams, "a": self.a, "b": self.b}

        return hyperparams


class GlorotUniform(WeightInit):
    def __init__(self, weight_name):
        super(GlorotUniform, self).__init__(weight_name)

    def __call__(self, weight):
        nn.init.xavier_uniform_(weight)


class Normal(WeightInit):
    def __init__(self, weight_name, mean=0, std=1):
        super(Normal, self).__init__(weight_name)
        self.mean = mean
        self.std = std

    def __call__(self, weight):
        nn.init.normal_(weight, mean=self.mean, std=self.std)

    @property
    def hyperparams(self):
        hyperparams = {**super().hyperparams, "mean": self.mean, "std": self.std}

        return hyperparams


class GlorotNormal(WeightInit):
    def __init__(self, weight_name):
        super(GlorotNormal, self).__init__(weight_name)

    def __call__(self, weight):
        nn.init.xavier_normal_(weight)


class Identity(WeightInit):
    def __init__(self, weight_name, c=1):
        super(Identity, self).__init__(weight_name)
        self.c = c

    def __call__(self, weight):
        nn.init.eye_(weight)
        with torch.no_grad():
            weight *= self.c

    @property
    def hyperparams(self):
        hyperparams = {**super().hyperparams, "c": self.c}

        return hyperparams
