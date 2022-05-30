import os
import pickle

import torch

from brainbox.datasets.base import PredictionTemporalDataset


class BBCWild(PredictionTemporalDataset):

    _N_TRAIN_CLIPS = 812
    _N_TEST_CLIPS = 203
    _CLIP_LEN = 50

    def __init__(
        self,
        root,
        sample_length,
        dt,
        pred_horizon=1,
        train=True,
        preprocess=None,
        transform=None,
        target_transform=None,
    ):
        n_clips = BBCWild._N_TRAIN_CLIPS if train else BBCWild._N_TEST_CLIPS
        super().__init__(
            root,
            sample_length,
            dt,
            n_clips,
            BBCWild._CLIP_LEN,
            pred_horizon,
            train,
            preprocess,
            transform,
            target_transform,
        )

    @property
    def model_outputs_path(self):
        return os.path.join(self._root, "normalized_concattrain.pkl")

    def load_clip(self, i):
        x = self.dataset[i]

        return x

    def _load_dataset(self, train):
        file = open(self.model_outputs_path, "rb")
        dataset = pickle.load(file)
        file.close()

        if train:
            dataset = dataset[: BBCWild._N_TRAIN_CLIPS]
        else:
            dataset = dataset[BBCWild._N_TRAIN_CLIPS :]

        dataset = torch.from_numpy(dataset)
        dataset = dataset.permute(0, 3, 1, 2)
        dataset = dataset.unsqueeze(1)
        dataset = dataset.type(torch.FloatTensor)

        return dataset, None
