import os

import torch

from ._dataset import BBDataset


# TODO Ensure response tensor is of shape b x 1 x t x n_neurons

class PVC1Dataset(BBDataset):

    N_NEURONS_TRIAL_SINGLE = 96
    N_NEURONS_TRIAL_MULTI = 23

    TRIAL_SINGLE = 'single'
    TRIAL_MULTI = 'multi'
    FOLD_TRAIN = 'train'
    FOLD_VAL = 'val'
    FOLD_TEST = 'test'

    _MAX_MOVIE = 4
    _MAX_SEG = 30
    _MAX_FRAMES = 900

    # Split: 64 train, 16 val, 20 test
    _TRIAL_FOLD_MOVIE_IDS = {
        'multi': {
            'all': [(0, 0), (0, 1), (0, 2), (0, 3), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 7), (1, 8), (1, 9)],
            'train': [(1, 8), (1, 2), (1, 5), (1, 1), (1, 4), (1, 9), (0, 8), (1, 7), (0, 1), (0, 7), (1, 3)],
            'val': [(0, 2), (0, 3), (0, 6)],
            'test': [(0, 5), (0, 0), (1, 0), (0, 9)]},
        'single': {
            'all': [(0, 0), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (0, 13), (0, 14), (0, 16), (0, 17), (0, 18), (0, 19), (0, 20), (0, 21), (0, 22), (0, 23), (0, 24), (0, 25), (0, 26), (0, 27), (0, 28), (0, 29), (1, 0), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (1, 14), (1, 16), (1, 17), (1, 18), (1, 19), (1, 20), (1, 21), (1, 22), (1, 23), (1, 24), (1, 25), (1, 26), (1, 27), (1, 28), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (2, 15), (2, 16), (2, 17), (2, 18), (2, 19), (2, 21), (2, 22), (2, 23), (2, 24), (2, 25), (2, 26), (2, 27), (2, 28), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (3, 12), (3, 13), (3, 15), (3, 16), (3, 17), (3, 18), (3, 19), (3, 20), (3, 21), (3, 22), (3, 23), (3, 24), (3, 25), (3, 26), (3, 28)],
            'train': [(2, 6), (3, 26), (2, 2), (2, 23), (1, 5), (2, 27), (1, 9), (3, 17), (2, 24), (2, 4), (1, 4), (2, 0), (3, 21), (2, 5), (0, 28), (1, 23), (2, 8), (3, 18), (1, 8), (3, 2), (0, 11), (2, 9), (0, 22), (3, 0), (0, 4), (0, 13), (1, 24), (3, 15), (1, 16), (2, 15), (3, 23), (1, 10), (1, 2), (2, 22), (1, 22), (1, 6), (3, 12), (1, 28), (2, 21), (1, 11), (2, 12), (3, 28), (0, 26), (3, 3), (2, 26), (0, 14), (3, 1), (1, 3), (1, 13), (3, 10), (0, 24), (3, 8), (1, 27), (1, 19), (2, 28), (2, 1), (0, 6), (3, 22), (2, 7), (2, 25), (3, 25), (0, 12), (0, 25), (3, 7), (2, 3), (2, 11), (1, 12), (2, 10), (0, 23)],
            'val': [(0, 0), (1, 17), (1, 21), (0, 20), (2, 17), (3, 19), (1, 18), (0, 18), (3, 20), (0, 17), (0, 8), (0, 21), (3, 13), (1, 7), (0, 10), (2, 19), (3, 9)],
            'test': [(1, 14), (1, 20), (0, 2), (3, 16), (2, 18), (1, 25), (3, 5), (0, 5), (0, 3), (0, 19), (1, 26), (0, 16), (2, 16), (3, 24), (0, 27), (2, 13), (1, 0), (0, 9), (0, 7), (3, 4), (3, 11), (0, 29)]}
    }

    def __init__(self, root, t_len, start_pred_on_step, trial_type, fold_type, transform=None, target_transform=None):
        super().__init__(transform, target_transform)
        assert trial_type in [PVC1Dataset.TRIAL_SINGLE, PVC1Dataset.TRIAL_MULTI], 'Please provide a valid response type.'
        assert fold_type in [PVC1Dataset.FOLD_TRAIN, PVC1Dataset.FOLD_VAL, PVC1Dataset.FOLD_TEST], 'Please provide a valid fold type.'
        assert start_pred_on_step <= t_len, 'start_pred_on_step needs to <= t_len'
        self.root = root
        self.t_len = t_len
        self.start_pred_on_step = start_pred_on_step
        self.trial_type = trial_type
        self.fold_type = fold_type

        # Load neural responses and movies
        self.neural_responses = torch.load(self.responses_path)[self._neural_response_ids(trial_type, fold_type)]
        self.neural_responses = self.neural_responses.permute(0, 2, 1, 3).float()
        self.movies = torch.load(self.movies_path)[self._trial_fold_movie_ids(trial_type, fold_type)]
        self.movies = self.movies.permute(0, 2, 1, 3, 4)

        # Build the indices of the dataset
        self.index = []
        for movie_id in range(self.movies.shape[0]):
            t_id = 0
            while t_id + t_len + 1 <= PVC1Dataset._MAX_FRAMES:
                self.index.append((movie_id, t_id))
                t_id += t_len - start_pred_on_step + 1

    def __getitem__(self, i):
        movie_id, t_id = self.index[i]
        end_clip_id = t_id + self.t_len

        x = self.movies[movie_id, :, t_id:end_clip_id]
        y = self.neural_responses[movie_id, :, :, t_id + self.start_pred_on_step - 1:end_clip_id]

        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y

    def __len__(self):
        return len(self.index)

    @property
    def hyperparams(self):
        return {**super().hyperparams, 't_len': self.t_len, 'start_pred_on_step': self.start_pred_on_step,
                'trial_type': self.trial_type, 'fold_type': self.fold_type}

    @property
    def responses_path(self):
        return os.path.join(self.root, 'processed', 'responses', '{0}_trial_responses.pt'.format(self.trial_type))

    @property
    def movies_path(self):
        return os.path.join(self.root, 'processed', 'movies.pt')

    def _neural_response_ids(self, trial_type, fold_type):
        neural_response_ids = []
        mapping = {neural_id: i for i, neural_id in enumerate(PVC1Dataset._TRIAL_FOLD_MOVIE_IDS[trial_type]['all'])}

        for neural_response_id in PVC1Dataset._TRIAL_FOLD_MOVIE_IDS[trial_type][fold_type]:
            neural_response_ids.append(mapping[neural_response_id])

        return neural_response_ids

    def _trial_fold_movie_ids(self, trial_type, fold_type):
        movie_ids = []
        mapping = {(i, j): i * PVC1Dataset._MAX_SEG + j for i in range(PVC1Dataset._MAX_MOVIE) for j in range(PVC1Dataset._MAX_SEG)}

        for movie_id in PVC1Dataset._TRIAL_FOLD_MOVIE_IDS[trial_type][fold_type]:
            movie_ids.append(mapping[movie_id])

        return movie_ids

