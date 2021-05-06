import torch
import torch.nn as nn

from . import weight_init


class BBModel(nn.Module):

    def __init__(self):
        super(BBModel, self).__init__()
        self.__hyperparams = {'name': self.__class__.__name__}

    @property
    def hyperparams(self):
        return self.__hyperparams

    def _get_variable_name(self, src_param):

        for name, param in self.named_parameters():

            if torch.equal(param, src_param):
                return name

        return None

    def init_weight(self, weight, init_type, **kwargs):

        weight_name = self._get_variable_name(weight)

        if init_type == 'constant':
            w_initializer = weight_init.Constant(weight_name, kwargs.get('c', 0.42))
        elif init_type == 'uniform':
            w_initializer = weight_init.Uniform(weight_name, kwargs.get('a', 0), kwargs.get('b', 1))
        elif init_type == 'glorot_uniform':
            w_initializer = weight_init.GlorotUniform(weight_name)
        elif init_type == 'normal':
            w_initializer = weight_init.Normal(weight_name, kwargs.get('mean', 0), kwargs.get('std', 1))
        elif init_type == 'glorot_normal':
            w_initializer = weight_init.GlorotNormal(weight_name)
        elif init_type == 'identity':
            w_initializer = weight_init.Identity(weight_name, kwargs.get('c', 1))

        w_initializer(weight)

        if weight_name is not None:
            self.__hyperparams = {**self.__hyperparams, **w_initializer.hyperparams}
