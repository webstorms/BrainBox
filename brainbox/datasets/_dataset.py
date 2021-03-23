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

    def __init__(self, t_len, dt, n_clips, n_timesteps, transform=None, target_transform=None):
        super().__init__(transform, target_transform)
        self.t_len = t_len
        self.dt = dt

        self.ids = []
        for clip_id in range(n_clips):
            for t_id in range(0, n_timesteps - t_len + 1, dt):
                self.ids.append((clip_id, t_id))

    @property
    def hyperparams(self):
        return {**super().hyperparams, 't_len': self.t_len, 'dt': self.dt}

    def load_clip(self, i):
        # x: channel x timesteps x ...
        # y: channel x timesteps x ...
        raise NotImplementedError

    def __getitem__(self, i):
        clip_idx, t_idx = self.ids[i]
        x, y = self.load_clip(clip_idx)
        x = x[:, t_idx:t_idx + self.t_len]
        y = y[:, t_idx:t_idx + self.t_len]

        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y

    def __len__(self):
        return len(self.ids)
