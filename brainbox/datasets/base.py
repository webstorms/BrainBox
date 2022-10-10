import torch


class BBDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        train=True,
        preprocess=None,
        transform=None,
        target_transform=None,
        push_gpu=False,
    ):
        self._root = root
        self._train = train
        self._transform = transform
        self._target_transform = target_transform

        self._dataset, self._targets = self._load_dataset(train)

        if preprocess is not None:
            self._dataset = preprocess(self._dataset)

        if push_gpu:
            self._dataset = self._dataset.cuda()
            self._targets = self._targets.cuda()

    def __getitem__(self, i):
        x, y = self._dataset[i], self._targets[i]

        if self._transform is not None:
            x = self._transform(x)

        return x, y

    def __len__(self):
        return len(self._dataset)

    @property
    def hyperparams(self):
        hyperparams = {"name": self.__class__.__name__, "train": self._train}

        if self._transform is not None:
            hyperparams["transform"] = self._transform.hyperparams

        if self._target_transform is not None:
            hyperparams["target_transform"] = self._target_transform.hyperparams

        return hyperparams

    def dims(self):
        return self._dataset.shape

    def _load_dataset(self, train):
        raise NotImplementedError


class TemporalDataset(BBDataset):
    def __init__(
        self,
        root,
        sample_length,
        dt,
        n_clips,
        clip_length,
        train=True,
        preprocess=None,
        transform=None,
        target_transform=None,
        push_gpu=False,
    ):
        super().__init__(root, train, preprocess, transform, target_transform, push_gpu)
        assert (
            sample_length <= clip_length
        ), f"sample_length {sample_length} needs to be less or equal to n_timesteps {clip_length}"
        self._sample_length = sample_length
        self._dt = dt
        self._n_clips = n_clips
        self._clip_length = clip_length

        self._ids = []
        for clip_id in range(n_clips):
            for t_id in range(0, clip_length - sample_length + 1, dt):
                self._ids.append((clip_id, t_id))

    def __getitem__(self, i):
        clip_id, t_id = self._ids[i]
        x, y = self.load_clips(clip_id)
        x = x[:, t_id : t_id + self._sample_length]
        y = y[:, t_id : t_id + self._sample_length]

        if self._transform is not None:
            x = self._transform(x)

        if self._target_transform is not None:
            y = self._target_transform(y)

        return x, y

    def __len__(self):
        return len(self._ids)

    @property
    def hyperparams(self):
        return {
            **super().hyperparams,
            "sample_length": self._sample_length,
            "dt": self._dt,
            "n_clips": self._n_clips,
            "clip_length": self._clip_length,
        }

    @property
    def fps(self):
        return self._get_fps()

    def load_clips(self, i):
        # x: channel x timesteps x ...
        # y: channel x timesteps x ...
        raise NotImplementedError

    def _get_fps(self):
        raise NotImplementedError


class PredictionTemporalDataset(TemporalDataset):
    def __init__(
        self,
        root,
        sample_length,
        dt,
        n_clips,
        clip_length,
        pred_horizon,
        train=True,
        preprocess=None,
        transform=None,
        target_transform=None,
        push_gpu=False,
    ):
        super().__init__(
            root,
            sample_length,
            dt,
            n_clips,
            clip_length - pred_horizon,
            train,
            preprocess,
            transform,
            target_transform,
            push_gpu,
        )
        self._pred_horizon = pred_horizon

    @property
    def hyperparams(self):
        return {**super().hyperparams, "pred_horizon": self._pred_horizon}

    def load_clips(self, i):
        clip = self.load_clip(i)

        x = clip[:, : -self._pred_horizon]
        y = clip[:, self._pred_horizon :]

        return x, y

    def load_clip(self, i):
        raise NotImplementedError


class PatchPredictionTemporalDataset(PredictionTemporalDataset):
    def __init__(
        self,
        root,
        sample_length,
        dt,
        in_k,
        out_k,
        stride,
        n_clips,
        clip_length,
        clip_h,
        clip_w,
        pred_horizon,
        train=True,
        preprocess=None,
        transform=None,
        target_transform=None,
        push_gpu=False,
    ):
        super().__init__(
            root,
            sample_length,
            dt,
            n_clips,
            clip_length,
            pred_horizon,
            train,
            preprocess,
            transform,
            target_transform,
            push_gpu,
        )

        self._in_k = in_k
        self._out_k = out_k
        self._stride = stride

        self._clip_h = clip_h
        self._clip_w = clip_w

        self._ids = []
        for clip_id in range(n_clips):
            for t_id in range(0, clip_length - sample_length - pred_horizon + 1, dt):
                for h_id in range(0, clip_h, self._stride):
                    for w_id in range(0, clip_w, self._stride):
                        if h_id + in_k <= clip_h and w_id + in_k <= clip_w:
                            self._ids.append((clip_id, t_id, h_id, w_id))

    @property
    def hyperparams(self):
        return {
            **super().hyperparams,
            "in_k": self._in_k,
            "out_k": self._out_k,
            "stride": self._stride,
            "clip_h": self._clip_h,
            "clip_w": self._clip_w,
        }

    def __getitem__(self, i):
        clip_id, t_id, h_id, w_id = self._ids[i]
        x, y = self.load_clips(clip_id)
        p = int((self._in_k - self._out_k) / 2)
        x = x[
            :,
            t_id : t_id + self._sample_length,
            h_id : h_id + self._in_k,
            w_id : w_id + self._in_k,
        ]
        y = y[
            :,
            t_id : t_id + self._sample_length,
            h_id + p : h_id + self._in_k - p,
            w_id + p : w_id + self._in_k - p,
        ]

        if self._transform is not None:
            x = self._transform(x)

        if self._target_transform is not None:
            y = self._target_transform(y)

        return x, y
