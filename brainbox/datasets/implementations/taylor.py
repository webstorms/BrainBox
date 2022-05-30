import os
import h5py

import torch
import numpy as np

from brainbox.datasets.base import PredictionTemporalDataset


class Natural(PredictionTemporalDataset):

    _N_TRAIN_CLIPS = 31
    _N_TEST_CLIPS = 8
    _CLIP_LENGTH = 1200

    def __init__(
        self,
        root,
        sample_length,
        dt,
        pred_horizon,
        train=True,
        preprocess=None,
        transform=None,
        target_transform=None,
        push_gpu=False,
    ):
        n_clips = Natural._N_TRAIN_CLIPS if train else Natural._N_TEST_CLIPS
        super().__init__(
            root,
            sample_length,
            dt,
            n_clips,
            Natural._CLIP_LENGTH,
            pred_horizon,
            train,
            preprocess,
            transform,
            target_transform,
            push_gpu,
        )

    @property
    def model_outputs_path(self):
        return os.path.join(self._root, "filtered_natural.hdf5")

    def load_clip(self, i):
        x = self._dataset[i]

        return x

    def _load_dataset(self, train):
        hf = h5py.File(self.model_outputs_path, "r")
        dataset_name = "train" if train else "test"
        dataset = np.array(hf.get(dataset_name))
        hf.close()

        dataset = torch.from_numpy(dataset)
        dataset = dataset.unsqueeze(1)
        dataset = dataset.type(torch.FloatTensor)

        return dataset, None


class MouseNat(PredictionTemporalDataset):

    _N_TRAIN_CLIPS = 241
    _N_TEST_CLIPS = 27
    _CLIP_LENGTH = 240
    # filtered: m=1.1744,std=264.695
    # non-filtered: m=160.24, std=60.71

    def __init__(
        self,
        root,
        sample_length,
        dt,
        pred_horizon,
        train=True,
        preprocess=None,
        transform=None,
        target_transform=None,
        push_gpu=False,
    ):
        n_clips = MouseNat._N_TRAIN_CLIPS if train else MouseNat._N_TEST_CLIPS
        super().__init__(
            root,
            sample_length,
            dt,
            n_clips,
            MouseNat._CLIP_LENGTH,
            pred_horizon,
            train,
            preprocess,
            transform,
            target_transform,
            push_gpu,
        )

    @property
    def model_outputs_path(self):
        return os.path.join(self._root, "filtered_mousenat.hdf5")

    def load_clip(self, i):
        x = self._dataset[i]

        return x

    def _load_dataset(self, train):
        hf = h5py.File(self.model_outputs_path, "r")
        dataset = np.array(hf.get("dataset"))
        hf.close()

        if train:
            dataset = dataset[: MouseNat._N_TRAIN_CLIPS]
        else:
            dataset = dataset[MouseNat._N_TRAIN_CLIPS :]
        dataset = torch.from_numpy(dataset)
        dataset = dataset.unsqueeze(1)
        dataset = dataset.type(torch.FloatTensor)

        return dataset, None


class HumanNat(PredictionTemporalDataset):

    _N_TRAIN_CLIPS = 579
    _N_TEST_CLIPS = 64
    _CLIP_LENGTH = 240
    # filtered: m=0.75,std=324.89
    # non-filtered: m=155.01, std=55.39

    def __init__(
        self,
        root,
        sample_length,
        dt,
        pred_horizon,
        train=True,
        preprocess=None,
        transform=None,
        target_transform=None,
        push_gpu=False,
    ):
        n_clips = HumanNat._N_TRAIN_CLIPS if train else HumanNat._N_TEST_CLIPS
        super().__init__(
            root,
            sample_length,
            dt,
            n_clips,
            HumanNat._CLIP_LENGTH,
            pred_horizon,
            train,
            preprocess,
            transform,
            target_transform,
            push_gpu,
        )

    @property
    def model_outputs_path(self):
        return os.path.join(self._root, "filtered_humannat.hdf5")

    def load_clip(self, i):
        x = self._dataset[i]

        return x

    def _load_dataset(self, train):
        hf = h5py.File(self.model_outputs_path, "r")
        dataset = np.array(hf.get("dataset"))
        hf.close()

        if train:
            dataset = dataset[: HumanNat._N_TRAIN_CLIPS]
        else:
            dataset = dataset[HumanNat._N_TRAIN_CLIPS :]
        dataset = torch.from_numpy(dataset)
        dataset = dataset.unsqueeze(1)
        dataset = dataset.type(torch.FloatTensor)

        return dataset, None
