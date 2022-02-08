import os
import h5py

import torch
import numpy as np

from ._dataset import PredictionTemporalDataset


class Natural(PredictionTemporalDataset):

    _N_TRAIN_CLIPS = 31
    _N_TEST_CLIPS = 8
    _CLIP_LENGTH = 1200

    def __init__(self, root, sample_length, dt, pred_horizon, train=True, transform=None, target_transform=None):
        n_clips = Natural._N_TRAIN_CLIPS if train else Natural._N_TEST_CLIPS
        super().__init__(sample_length, dt, n_clips, Natural._CLIP_LENGTH, pred_horizon, transform, target_transform)
        self._root = root

        self._dataset = self._load_dataset(train).pin_memory()

    @property
    def model_outputs_path(self):
        return os.path.join(self._root, 'filtered_natural.hdf5')

    def load_clip(self, i):
        x = self._dataset[i]

        return x

    def _load_dataset(self, is_training):
        hf = h5py.File(self.model_outputs_path, 'r')
        dataset_name = 'train' if is_training else 'test'
        dataset = np.array(hf.get(dataset_name))
        hf.close()

        dataset = torch.from_numpy(dataset)
        dataset = dataset.unsqueeze(1)
        dataset = dataset.type(torch.FloatTensor)

        return dataset


class MouseNat(PredictionTemporalDataset):

    _N_TRAIN_CLIPS = 241
    _N_TEST_CLIPS = 27
    _CLIP_LENGTH = 240
    # filtered: m=1.1744,std=264.695
    # non-filtered: m=160.24, std=60.71

    def __init__(self, root, sample_length, dt, pred_horizon, train=True, preprocess=None, transform=None, target_transform=None):
        n_clips = MouseNat._N_TRAIN_CLIPS if train else MouseNat._N_TEST_CLIPS
        super().__init__(sample_length, dt, n_clips, MouseNat._CLIP_LENGTH, pred_horizon, transform, target_transform)
        self._root = root

        self._dataset = self._load_dataset(train)

        if preprocess is not None:
            self._dataset = preprocess(self._dataset)

    @property
    def model_outputs_path(self):
        return os.path.join(self._root, 'filtered_mousenat.hdf5')

    def load_clip(self, i):
        x = self._dataset[i]

        return x

    def _load_dataset(self, is_training):
        hf = h5py.File(self.model_outputs_path, 'r')
        dataset = np.array(hf.get('dataset'))
        hf.close()

        if is_training:
            dataset = dataset[:MouseNat._N_TRAIN_CLIPS]
        else:
            dataset = dataset[MouseNat._N_TRAIN_CLIPS:]
        dataset = torch.from_numpy(dataset)
        dataset = dataset.unsqueeze(1)
        dataset = dataset.type(torch.FloatTensor)

        return dataset


class HumanNat(PredictionTemporalDataset):

    _N_TRAIN_CLIPS = 579
    _N_TEST_CLIPS = 64
    _CLIP_LENGTH = 240
    # filtered: m=0.75,std=324.89
    # non-filtered: m=155.01, std=55.39

    def __init__(self, root, sample_length, dt, pred_horizon, train=True, transform=None, target_transform=None):
        n_clips = HumanNat._N_TRAIN_CLIPS if train else HumanNat._N_TEST_CLIPS
        super().__init__(sample_length, dt, n_clips, HumanNat._CLIP_LENGTH, pred_horizon, transform, target_transform)
        self._root = root

        self._dataset = self.load_dataset(train)

    @property
    def model_outputs_path(self):
        return os.path.join(self._root, 'filtered_humannat.hdf5')

    def load_clip(self, i):
        x = self._dataset[i]

        return x

    def _load_dataset(self, is_training):
        hf = h5py.File(self.model_outputs_path, 'r')
        dataset = np.array(hf.get('dataset'))
        hf.close()

        if is_training:
            dataset = dataset[:HumanNat._N_TRAIN_CLIPS]
        else:
            dataset = dataset[HumanNat._N_TRAIN_CLIPS:]

        dataset = torch.from_numpy(dataset)
        dataset = dataset.unsqueeze(1)
        dataset = dataset.type(torch.FloatTensor)

        return dataset