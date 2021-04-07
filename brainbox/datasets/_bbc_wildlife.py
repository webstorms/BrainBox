import os
import pickle

import torch

from ._dataset import TemporalDataset


class BBCWild(TemporalDataset):

    _N_TRAIN_CLIPS = 812
    _N_TEST_CLIPS = 203
    _CLIP_LEN = 50

    def __init__(self, root, t_len, dt, train=True, transform=None, target_transform=None):
        n_clips = BBCWild._N_TRAIN_CLIPS if train else BBCWild._N_TEST_CLIPS
        super().__init__(t_len, dt, n_clips, BBCWild._CLIP_LEN, transform, target_transform)
        self.root = root

        file = open(self.model_outputs_path, 'rb')
        dataset = pickle.load(file)
        file.close()

        if train:
            dataset = dataset[:BBCWild._N_TRAIN_CLIPS]
        else:
            dataset = dataset[BBCWild._N_TRAIN_CLIPS:]

        self.dataset = torch.from_numpy(dataset)
        self.dataset = self.dataset.permute(0, 3, 1, 2)
        self.dataset = self.dataset.unsqueeze(1)
        self.dataset = self.dataset.type(torch.FloatTensor)

    def load_clip(self, i):
        x = self.dataset[i]
        y = self.dataset[i, :, 1:]  # We shift the clip by one frame

        return x, y

    @property
    def model_outputs_path(self):
        return os.path.join(self.root, 'preprocessed_dataset.pkl')
