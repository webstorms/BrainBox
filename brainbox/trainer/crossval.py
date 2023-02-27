import os
import sys
import copy
import logging

import torch
import pandas as pd

from .validator import compute_metric

logger = logging.getLogger("crossval")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class FoldDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, idxs):
        self._dataset = dataset
        self._idxs = idxs

    def __getitem__(self, i):
        i = self._idxs[i]
        x, y = self._dataset[i]

        return x, y

    def __len__(self):
        return len(self._idxs)

    @property
    def x(self):
        return self._dataset.x[self._idxs]

    @property
    def y(self):
        return self._dataset.y[self._idxs]


class BaseValidationTrainer:

    def __init__(self, root, model, train_dataset, trainer_class, trainer_kwargs, lambdas, minimise_score=True, final_repeat=1):
        self._root = root
        self._model = model
        self._train_dataset = train_dataset
        self._trainer_class = trainer_class
        self._trainer_kwargs = trainer_kwargs
        self._lambdas = lambdas
        self._minimise_score = minimise_score
        self._final_repeat = final_repeat

    @property
    def train_path(self):
        path = os.path.join(self._root, "models")
        if not os.path.exists(path):
            os.makedirs(path)
        
        return path

    def train(self):
        cv_results = []

        best_val_score = 0
        best_lam = None

        # Train over the different lambda values
        for lam in self._lambdas:
            logger.info(f"Fitting for l1={lam}...")
            # Train CV
            train_score, val_score = self._fit_over_folds(lam)
            cv_results.append({"lam": lam, "train": train_score, "val": val_score})

            val_score_improvement = val_score < best_val_score if self._minimise_score else val_score > best_val_score
            if val_score_improvement or best_lam is None:
                best_lam = lam
                best_val_score = val_score

        # Train on all data and save results and model
        logger.info(f"Fitting for best l1={best_lam}...")
        for i in range(self._final_repeat):
            self._fit_for_fold(best_lam, self._train_dataset, save=True, id=str(i))

        cv_results = pd.DataFrame(cv_results)
        cv_results.to_csv(os.path.join(self._root, "cv.csv"), index=False)

    def _fit_over_folds(self, lam):
        raise NotImplementedError

    def _fit_for_fold(self, lam, train_dataset, val_dataset=None, save=False, id=None):
        model = copy.deepcopy(self._model)
        trainer = self._trainer_class(self.train_path, model, train_dataset, lam=lam, **self._trainer_kwargs, id=id)
        trainer.train(save)

        if val_dataset is None:
            return self._compute_score(model, train_dataset)
        else:
            train_score = self._compute_score(model, train_dataset)
            val_score = self._compute_score(model, val_dataset)

            return train_score, val_score

    def _compute_score(self, model, dataset):
        trainer = self._trainer_class(self.train_path, model, dataset, lam=0, **self._trainer_kwargs)
        loss = lambda output, target: trainer.loss(output, target, model)
        score = compute_metric(model, dataset, loss, trainer.batch_size, trainer.device, trainer.dtype)

        return torch.Tensor(score).sum().item()


class SplitValidationTrainer(BaseValidationTrainer):

    def __init__(self, root, model, train_dataset, trainer_class, trainer_kwargs, lambdas, train_idxs, val_idxs, minimise_score=True, final_repeat=1):
        super().__init__(root, model, train_dataset, trainer_class, trainer_kwargs, lambdas, minimise_score, final_repeat)
        self._train_idxs = train_idxs
        self._val_idxs = val_idxs

    def _fit_over_folds(self, lam):
        train_dataset = FoldDataset(self._train_dataset, self._train_idxs)
        val_dataset = FoldDataset(self._train_dataset, self._val_idxs)
        train_score, val_score = self._fit_for_fold(lam, train_dataset, val_dataset)

        return train_score, val_score


class KFoldValidationTrainer(BaseValidationTrainer):

    def __init__(self, root, model, train_dataset, trainer_class, trainer_kwargs, lambdas, k, minimise_score=True, final_repeat=1):
        super().__init__(root, model, train_dataset, trainer_class, trainer_kwargs, lambdas, minimise_score, final_repeat)
        self._k = k

    def _fit_over_folds(self, lam):
        kf = KFold(n_splits=self._k, shuffle=False)

        train_scores_list = []
        val_scores_list = []

        for i, (train_idxs, val_idxs) in enumerate(kf.split(self._train_dataset)):
            train_dataset = FoldDataset(self._train_dataset, train_idxs)
            val_dataset = FoldDataset(self._train_dataset, val_idxs)
            train_score, val_score = self._fit_for_fold(lam, train_dataset, val_dataset)
            train_scores_list.append(train_score)
            val_scores_list.append(val_score)

        train_score = torch.stack(train_scores_list).mean().item()
        val_score = torch.stack(val_scores_list).mean().item()

        return train_score, val_score
