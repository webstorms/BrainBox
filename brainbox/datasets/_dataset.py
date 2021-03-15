import torch


class BBDataset(torch.utils.data.Dataset):

    def __init__(self, transform, target_transform):
        self.transform = transform
        self.target_transform = target_transform

    @staticmethod
    def _append_prefix_to_hyperparams(prefix, hyperparams):
        return {'_'.join([prefix, key]): value for key, value in hyperparams.items()}

    @property
    def hyperparams(self):

        hyperparams = {'name': self.__class__.__name__}

        if self.transform is not None:
            transform_hyperparams = BBDataset._append_prefix_to_hyperparams('trans', self.transform.hyperparams)
            hyperparams = {**hyperparams, **transform_hyperparams}
        if self.target_transform is not None:
            target_transform_hyperparams = BBDataset._append_prefix_to_hyperparams('targ_trans', self.target_transform.hyperparams)
            hyperparams = {**hyperparams, **target_transform_hyperparams}

        return hyperparams
