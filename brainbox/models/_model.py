import torch.nn as nn


class BBModel(nn.Module):

    @property
    def hyperparams(self):
        return {'name': self.__class__.__name__}
