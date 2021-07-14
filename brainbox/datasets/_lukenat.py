import os
import h5py

import torch
import numpy as np

from ._dataset import PredictionTemporalDataset


class MouseNat(PredictionTemporalDataset):

    _N_TRAIN_CLIPS = 241
    _N_TEST_CLIPS = 27
    _CLIP_LEN = 240
    # filtered: m=1.1744,std=264.695
    # non-filtered: m=160.24, std=60.71

    def __init__(self, root, t_len, dt, resample_step, pred_horizon, filtered=False, train=True, transform=None,
                 target_transform=None):
        n_clips = MouseNat._N_TRAIN_CLIPS if train else MouseNat._N_TEST_CLIPS
        super().__init__(t_len, dt, n_clips, MouseNat._CLIP_LEN, resample_step, pred_horizon, transform,
                         target_transform)
        self.root = root
        self.filtered = filtered

        hf = h5py.File(self.model_outputs_path, 'r')
        dataset = np.array(hf.get('dataset'))
        hf.close()

        if train:
            dataset = dataset[:MouseNat._N_TRAIN_CLIPS]
        else:
            dataset = dataset[MouseNat._N_TRAIN_CLIPS:]
        self.dataset = torch.from_numpy(dataset)
        self.dataset = self.dataset.unsqueeze(1)
        self.dataset = self.dataset.type(torch.FloatTensor)

    def load_clip(self, i):
        x = self.dataset[i]

        return x

    @property
    def model_outputs_path(self):
        name = 'filtered_mousenat.hdf5' if self.filtered else 'mousenat.hdf5'
        return os.path.join(self.root, name)


class HumanNat(PredictionTemporalDataset):

    _N_TRAIN_CLIPS = 579
    _N_TEST_CLIPS = 64
    _CLIP_LEN = 240
    # filtered: m=0.75,std=324.89
    # non-filtered: m=155.01, std=55.39

    def __init__(self, root, t_len, dt, resample_step, pred_horizon, filtered=False, train=True, transform=None,
                 target_transform=None):
        n_clips = MouseNat._N_TRAIN_CLIPS if train else MouseNat._N_TEST_CLIPS
        super().__init__(t_len, dt, n_clips, MouseNat._CLIP_LEN, resample_step, pred_horizon, transform,
                         target_transform)
        self.root = root
        self.filtered = filtered

        hf = h5py.File(self.model_outputs_path, 'r')
        dataset = np.array(hf.get('dataset'))
        hf.close()

        if train:
            dataset = dataset[:HumanNat._N_TRAIN_CLIPS]
        else:
            dataset = dataset[HumanNat._N_TRAIN_CLIPS:]
        self.dataset = torch.from_numpy(dataset)
        self.dataset = self.dataset.unsqueeze(1)
        self.dataset = self.dataset.type(torch.FloatTensor)

    def load_clip(self, i):
        x = self.dataset[i]

        return x

    @property
    def model_outputs_path(self):
        name = 'filtered_humannat.hdf5' if self.filtered else 'humannat.hdf5'
        return os.path.join(self.root, name)
