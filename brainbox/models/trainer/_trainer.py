import os
import glob
import uuid
from datetime import datetime

import torch
import numpy as np
import pandas as pd


class Trainer:

    GRAD_VALUE_CLIP_PRE = 'GRAD_VALUE_CLIP_PRE'
    GRAD_VALUE_CLIP_POST = 'GRAD_VALUE_CLIP_POST'
    GRAD_NORM_CLIP = 'GRAD_NORM_CLIP'

    def __init__(self, root, model, train_dataset, n_epochs, batch_size, lr, optimizer_func=torch.optim.Adam, device='cuda', dtype=torch.float, grad_clip_type=None, grad_clip_value=None):
        self.root = root
        self.model = model
        self.train_dataset = train_dataset
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer_func = optimizer_func
        self.device = device
        self.dtype = dtype
        self.grad_clip_type = grad_clip_type
        self.grad_clip_value = grad_clip_value

        # Instantiate housekeeping variables
        self.id = str(uuid.uuid4().hex)
        self.log = {'train_loss': []}
        self.train_data_loader = torch.utils.data.DataLoader(self.train_dataset, self.batch_size)

        if self.dtype == torch.float:
            self.optimizer = self.optimizer_func(self.model.parameters(), 10 ** self.lr)
        elif self.dtype == torch.half:
            self.optimizer = self.optimizer_func(self.model.parameters(), 10 ** self.lr, eps=1e-4)

        # Register grad clippings
        if self.grad_clip_type == Trainer.GRAD_VALUE_CLIP_PRE:

            for p in self.model.parameters():
                p.register_hook(lambda grad: torch.clamp(grad, -grad_clip_value, grad_clip_value))

        # Initialise the model
        self.model = self.model.to(device)
        if dtype == torch.float:
            self.model = self.model.float()
        elif dtype == torch.half:
            self.model = self.model.half()
        self.model.train()

        self.date = datetime.today().strftime('%Y%m%d')

    @property
    def name(self):
        return '{0}_{1}'.format(self.date, self.id)

    @staticmethod
    def _append_prefix_to_hyperparams(prefix, hyperparams):
        return {'_'.join([prefix, key]): value for key, value in hyperparams.items()}

    @property
    def hyperparams(self):
        return {'n_epochs': self.n_epochs, 'batch_size': self.batch_size, 'lr': self.lr, 'dtype': self.dtype, 'grad_clip_type': self.grad_clip_type, 'grad_clip_value': self.grad_clip_value}

    @property
    def model_path(self):
        name = '{0}_{1}'.format(self.name, 'model.pt')

        return os.path.join(self.root, name)

    @property
    def model_hyperparams_path(self):
        name = '{0}_{1}'.format(self.name, 'hyperparams.csv')

        return os.path.join(self.root, name)

    @property
    def model_log_path(self):
        name = '{0}_{1}'.format(self.name, 'log.csv')

        return os.path.join(self.root, name)

    def save_model(self):
        torch.save(self.model, self.model_path)

    def save_hyperparams(self):
        dataset_hyperparams = Trainer._append_prefix_to_hyperparams('dataset', self.train_dataset.hyperparams)
        train_hyperparams = Trainer._append_prefix_to_hyperparams('train', self.hyperparams)
        model_hyperparams = Trainer._append_prefix_to_hyperparams('model', self.model.hyperparams)

        hyperparams = {**dataset_hyperparams, **train_hyperparams, **model_hyperparams}
        hyperparams = pd.Series(hyperparams)
        hyperparams.to_csv(self.model_hyperparams_path, index=True)

    def save_model_log(self):
        log_df = pd.DataFrame(self.log)
        log_df.to_csv(self.model_log_path, index=False)

    def loss(self, output, target, model):
        raise NotImplementedError

    def val_loss(self, output, target, model):
        raise NotImplementedError

    def on_epoch_complete(self, save):
        if save:
            self.save_model_log()

    def on_training_start(self, save):
        if save:
            self.save_hyperparams()

    def on_training_complete(self, save):
        if save:
            self.save_model()

    def train_for_single_epoch(self):

        epoch_loss = 0

        for batch_id, (data, target) in enumerate(self.train_data_loader):
            data = data.to(self.device).type(self.dtype)
            target = target.to(self.device).type(self.dtype)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target, self.model)
            epoch_loss += loss.item()
            loss.backward()

            if self.grad_clip_type is not None:

                if self.grad_clip_type == Trainer.GRAD_NORM_CLIP:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)

                elif self.grad_clip_type == Trainer.GRAD_VALUE_CLIP_POST:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clip_value)

            self.optimizer.step()

        return epoch_loss

    def train(self, save=False):

        self.on_training_start(save)

        for epoch in range(self.n_epochs):
            print('Starting epoch {0}...'.format(epoch))
            # Train the model
            epoch_loss = self.train_for_single_epoch()
            print('epoch_loss', epoch_loss)
            self.log['train_loss'].append(epoch_loss)

            self.on_epoch_complete(save)

        self.on_training_complete(save)


class DecayTrainer(Trainer):

    def __init__(self, root, model, train_dataset, val_dataset, n_epochs, batch_size, lr, lr_decay=0.5, max_train_steps=10, max_decay_steps=4, optimizer_func=torch.optim.Adam, device='cuda', dtype=torch.float, grad_clip_type=None, grad_clip_value=None):
        super().__init__(root, model, train_dataset, n_epochs, batch_size, lr, optimizer_func, device, dtype, grad_clip_type, grad_clip_value)
        self.val_dataset = val_dataset
        self.lr_decay = lr_decay
        self.max_train_steps = max_train_steps
        self.max_decay_steps = max_decay_steps

        self.val_data_loader = torch.utils.data.DataLoader(val_dataset, self.batch_size)
        self.log = {'train_loss': [], 'val_loss': []}
        self.min_val_loss = np.inf
        self.train_steps_counter = 0
        self.decay_steps_counter = 0
        self._lr = 10 ** lr

    @property
    def hyperparams(self):
        return {**super().hyperparams, 'lr_decay': self.lr_decay, 'max_train_steps': self.max_train_steps, 'max_decay_steps': self.max_decay_steps}

    def save_hyperparams(self):
        train_dataset_hyperparams = Trainer._append_prefix_to_hyperparams('train_dataset', self.train_dataset.hyperparams)
        val_dataset_hyperparams = Trainer._append_prefix_to_hyperparams('val_dataset', self.val_dataset.hyperparams)
        train_hyperparams = Trainer._append_prefix_to_hyperparams('train', self.hyperparams)
        model_hyperparams = Trainer._append_prefix_to_hyperparams('model', self.model.hyperparams)

        hyperparams = {**train_dataset_hyperparams, **val_dataset_hyperparams, **train_hyperparams, **model_hyperparams}
        hyperparams = pd.Series(hyperparams)
        hyperparams.to_csv(self.model_hyperparams_path, index=True)

    def on_epoch_complete(self, save):

        val_loss = self.get_total_val_loss()
        self.log['val_loss'].append(val_loss)
        if save:
            self.save_model_log()

        print('Train loss', self.log['train_loss'][-1])
        print('Val loss', self.log['val_loss'][-1])

        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.train_steps_counter = 0

            if save:
                self.save_model()

        else:
            self.train_steps_counter += 1

            # We decay the learning rate if we have reached the specified number of training steps
            # and have not obtained any improvement in the validation loss
            if self.train_steps_counter == self.max_train_steps:

                # Decay the lr
                print('Decaying the lr... {0}/{1}'.format(self.decay_steps_counter + 1, self.max_decay_steps))
                self.train_steps_counter = 0
                self.decay_steps_counter += 1
                self._lr *= self.lr_decay

                # Load the best model and re-instantiated the optimizer
                if save:
                    self.model = torch.load(self.model_path)

                if self.dtype == torch.float:
                    self.optimizer = self.optimizer_func(self.model.parameters(), self._lr)
                elif self.dtype == torch.half:
                    self.optimizer = self.optimizer_func(self.model.parameters(), self._lr, eps=1e-4)

    def on_training_start(self, save):
        if save:
            self.save_model()
            self.save_hyperparams()

    def on_training_complete(self, save):
        pass

    def get_total_val_loss(self):
        self.model.eval()

        total_loss = 0

        with torch.no_grad():
            for batch_id, (data, target) in enumerate(self.val_data_loader):
                data = data.to(self.device).type(self.dtype)
                target = target.to(self.device).type(self.dtype)

                output = self.model(data)
                total_loss += self.val_loss(output, target, self.model).item()

        self.model.train()

        return total_loss

    def train(self, save=False):

        self.on_training_start(save)

        for epoch in range(self.n_epochs):
            print('Starting epoch {0}...'.format(epoch))
            # Train the model
            epoch_loss = self.train_for_single_epoch()

            self.log['train_loss'].append(epoch_loss)

            self.on_epoch_complete(save)

            # Halt the training process if we have reached the maximum number of lr decay steps
            if self.decay_steps_counter > self.max_decay_steps:
                print('Training stopped.')
                return

        self.on_training_complete(save)


def get_trainer(trainer_class, loss_function, *args):
    model_trainer = trainer_class(*args)
    model_trainer.loss = loss_function

    return model_trainer


def get_all_model_hyperparams(root):
    all_model_hyperparam_paths = glob.glob('{0}/*hyperparams.csv'.format(root))

    model_hyperparam_dfs = []

    for model_hyperparam_path in all_model_hyperparam_paths:
        model_id = '_'.join(model_hyperparam_path.split('/')[-1].split('_')[:2])

        model_hyperparam_df = pd.read_csv(model_hyperparam_path, names=[0, model_id]).set_index(0)
        model_hyperparam_dfs.append(model_hyperparam_df)

    return pd.concat(model_hyperparam_dfs, axis=1).T


def query_model_ids(root, **kwargs):
    all_model_hyperparam = get_all_model_hyperparams(root)

    query = all_model_hyperparam.index != None
    for key, value in kwargs.items():
        query &= (all_model_hyperparam[key] == str(value))

    return all_model_hyperparam[query]


def load_model(root, model_id=None, **kwargs):

    if model_id is None:
        model_ids = query_model_ids(root, **kwargs)
        assert len(model_ids) == 1, 'Multiple models match the query criteria'
        model_id = model_ids.index[0]

    model_path = os.path.join(root, '{0}_model.pt'.format(model_id))

    model = torch.load(model_path)
    model.eval()

    return model


def load_model_log(root, model_id=None, **kwargs):

    if model_id is None:
        model_ids = query_model_ids(root, **kwargs)
        assert len(model_ids) == 1, 'Multiple models match the query criteria'
        model_id = model_ids.index[0]

    model_log_path = os.path.join(root, '{0}_log.csv'.format(model_id))

    return pd.read_csv(model_log_path)


def remove_model(root, model_id):

    def remove(path):
        try:
            os.remove(path)
            print('Removed {0}'.format(path))
        except:
            print('Could not remove {0}'.format(path))

    model_hyperparams = os.path.join(root, '{0}_hyperparams.csv'.format(model_id))
    model_path = os.path.join(root, '{0}_model.pt'.format(model_id))
    model_log_path = os.path.join(root, '{0}_log.csv'.format(model_id))

    remove(model_hyperparams)
    remove(model_path)
    remove(model_log_path)


def remove_models(root, model_ids):
    for model_id in model_ids:
        remove_model(root, model_id)
