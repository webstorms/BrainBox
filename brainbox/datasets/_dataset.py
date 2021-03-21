import torch


class BBDataset(torch.utils.data.Dataset):

    def __init__(self, transform=None, target_transform=None):
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


class TemporalDataset(BBDataset):

    def __init__(self, t_len, dt, transform=None, target_transform=None):
        super().__init__(transform, target_transform)
        self.t_len = t_len
        self.dt = dt

        self.ids = []
        for clip_id in range(self.n_clips):
            t_id = 0
            while t_id + t_len <= self.n_timesteps:
                self.ids.append((clip_id, t_id))
                t_id += dt

    @property
    def hyperparams(self):
        return {**super().hyperparams, 't_len': self.t_len, 'dt': self.dt}

    @property
    def n_clips(self):
        # channel x timesteps x ...
        raise NotImplementedError

    @property
    def n_timesteps(self):
        raise NotImplementedError

    def load_clip(self, i):
        raise NotImplementedError

    def __getitem__(self, i):
        i, t = self.ids[i]
        x, y = self.load_clip(i)
        x = x[:, t:t + self.t_len]
        y = y[:, t:t + self.t_len]

        return x, y

    def __len__(self):
        return len(self.ids)
