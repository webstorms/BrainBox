import torch


class BBDataset(torch.utils.data.Dataset):

    def __init__(self, transform=None, target_transform=None):
        self._transform = transform
        self._target_transform = target_transform

    @property
    def hyperparams(self):
        hyperparams = {'name': self.__class__.__name__}

        if self._transform is not None:
            hyperparams['transform'] = self._transform.hyperparams

        if self._target_transform is not None:
            hyperparams['target_transform'] = self._target_transform.hyperparams

        return hyperparams


class TemporalDataset(BBDataset):

    def __init__(self, sample_length, dt, n_clips, clip_length, transform=None, target_transform=None):
        super().__init__(transform, target_transform)
        assert sample_length <= clip_length, f'sample_length {sample_length} needs to be less or equal to n_timesteps {clip_length}'
        self._sample_length = sample_length
        self._dt = dt
        self._n_clips = n_clips
        self._clip_length = clip_length

        self._ids = []
        for clip_id in range(n_clips):
            for t_id in range(0, clip_length - sample_length + 1, dt):
                self._ids.append((clip_id, t_id))

    @property
    def hyperparams(self):
        return {**super().hyperparams, 'sample_length': self._sample_length, 'dt': self._dt, 'n_clips': self._n_clips, 'clip_length': self._clip_length}

    def load_clips(self, i):
        # x: channel x timesteps x ...
        # y: channel x timesteps x ...
        raise NotImplementedError

    def __getitem__(self, i):
        clip_id, t_id = self._ids[i]
        x, y = self.load_clips(clip_id)
        x = x[:, t_id:t_id + self._sample_length]
        y = y[:, t_id:t_id + self._sample_length]

        if self._transform is not None:
            x = self._transform(x)

        if self._target_transform is not None:
            y = self._target_transform(y)

        return x, y

    def __len__(self):
        return len(self._ids)


class PredictionTemporalDataset(TemporalDataset):

    def __init__(self, sample_length, dt, n_clips, clip_length, pred_horizon, transform=None, target_transform=None):
        super().__init__(sample_length, dt, n_clips, clip_length - pred_horizon, transform, target_transform)
        self._pred_horizon = pred_horizon

    @property
    def hyperparams(self):
        return {**super().hyperparams, 'pred_horizon': self._pred_horizon}

    def load_clips(self, i):
        clip = self.load_clip(i)

        x = clip[:, :-self.pred_horizon]
        y = clip[:, self.pred_horizon:]

        return x, y

    def load_clip(self, i):
        raise NotImplementedError