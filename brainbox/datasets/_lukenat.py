import os
import h5py

import torch
import numpy as np

from ._dataset import PredictionTemporalDataset, FramePredictionTemporalDataset


# class Natural(PredictionTemporalDataset):
#
#     _N_TRAIN_CLIPS = 31
#     _N_TEST_CLIPS = 8
#     _CLIP_LEN = 1200
#
#     def __init__(self, root, sample_length, dt, pred_horizon, filtered=False, train=True, transform=None,
#                  target_transform=None):
#         n_clips = Natural._N_TRAIN_CLIPS if train else Natural._N_TEST_CLIPS
#         super().__init__(sample_length, dt, n_clips, Natural._CLIP_LEN, pred_horizon, transform, target_transform)
#         self.root = root
#         self.filtered = filtered
#
#         hf = h5py.File(self.model_outputs_path, 'r')
#         dataset_name = 'train' if train else 'test'
#         dataset = np.array(hf.get(dataset_name))
#         hf.close()
#
#         self.dataset = torch.from_numpy(dataset)
#         self.dataset = self.dataset.unsqueeze(1)
#         self.dataset = self.dataset.type(torch.FloatTensor)
#
#     def load_clip(self, i):
#         x = self.dataset[i]
#
#         return x
#
#     @property
#     def model_outputs_path(self):
#         name = 'filtered_natural.hdf5' if self.filtered else 'natural.hdf5'
#         return os.path.join(self.root, name)

# class Natural(FramePredictionTemporalDataset):
#
#     _N_TRAIN_CLIPS = 31
#     _N_TEST_CLIPS = 8
#     _CLIP_LEN = 1200
#
#     def __init__(self, root, sample_length, dt, sample_height, sample_width, pred_horizon, n_samples_per_clip, train, transform=None, target_transform=None):
#         n_clips = Natural._N_TRAIN_CLIPS if train else Natural._N_TEST_CLIPS
#         super().__init__(sample_length, dt, sample_height, sample_width, pred_horizon, n_samples_per_clip, n_clips, Natural._CLIP_LEN, transform, target_transform)
#         self.root = root
#
#         hf = h5py.File(self.model_outputs_path, 'r')
#         dataset_name = 'train' if train else 'test'
#         dataset = np.array(hf.get(dataset_name))
#         hf.close()
#
#         self.dataset = torch.from_numpy(dataset)
#         self.dataset = self.dataset.unsqueeze(1)
#         self.dataset = self.dataset.type(torch.FloatTensor)
#
#         for i in range(n_clips):
#             self.dataset[i] = self.transform(self.dataset[i])
#
#     def load_clip(self, i):
#         x = self.dataset[i]
#
#         return x
#
#     @property
#     def model_outputs_path(self):
#         return os.path.join(self.root, 'filtered_natural.hdf5')


class Natural(PredictionTemporalDataset):

    _N_TRAIN_CLIPS = 31
    _N_TEST_CLIPS = 8
    _CLIP_LEN = 1200

    def __init__(self, root, sample_length, dt, pred_horizon, train=True, transform=None, target_transform=None):
        n_clips = Natural._N_TRAIN_CLIPS if train else Natural._N_TEST_CLIPS
        super().__init__(sample_length, dt, n_clips, Natural._CLIP_LEN, pred_horizon, transform, target_transform)
        self.root = root

        hf = h5py.File(self.model_outputs_path, 'r')
        dataset_name = 'train' if train else 'test'
        dataset = np.array(hf.get(dataset_name))
        hf.close()

        self.dataset = torch.from_numpy(dataset)
        self.dataset = self.dataset.unsqueeze(1)
        self.dataset = self.dataset.type(torch.FloatTensor)

        # for i in range(n_clips):
        #     self.dataset[i] = self.transform(self.dataset[i])

    def load_clips(self, i):
        x = self.dataset[i]

        return x

    @property
    def model_outputs_path(self):
        return os.path.join(self.root, 'filtered_natural.hdf5')


class MouseNat(PredictionTemporalDataset):

    _N_TRAIN_CLIPS = 241
    _N_TEST_CLIPS = 27
    _CLIP_LEN = 240
    # filtered: m=1.1744,std=264.695
    # non-filtered: m=160.24, std=60.71

    # t_len, dt, n_clips, n_timesteps, pred_horizon, transform=None, target_transform=None

    def __init__(self, root, sample_length, dt, pred_horizon, filtered=False, train=True, transform=None,
                 target_transform=None):
        n_clips = MouseNat._N_TRAIN_CLIPS if train else MouseNat._N_TEST_CLIPS
        super().__init__(sample_length, dt, n_clips, MouseNat._CLIP_LEN, pred_horizon, transform, target_transform)
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

    def load_clips(self, i):
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

    def __init__(self, root, sample_length, dt, pred_horizon, filtered=False, train=True, transform=None,
                 target_transform=None):
        n_clips = MouseNat._N_TRAIN_CLIPS if train else MouseNat._N_TEST_CLIPS
        super().__init__(sample_length, dt, n_clips, MouseNat._CLIP_LEN, pred_horizon, transform,
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

    def load_clips(self, i):
        x = self.dataset[i]

        return x

    @property
    def model_outputs_path(self):
        name = 'filtered_humannat.hdf5' if self.filtered else 'humannat.hdf5'
        return os.path.join(self.root, name)
