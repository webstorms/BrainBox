import os
import sys
import glob
import uuid
import json
import logging
import time
from datetime import datetime

import torch
import numpy as np
import pandas as pd

logger = logging.getLogger("trainer")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class Trainer:

    GRAD_VALUE_CLIP_PRE = "GRAD_VALUE_CLIP_PRE"
    GRAD_VALUE_CLIP_POST = "GRAD_VALUE_CLIP_POST"
    GRAD_NORM_CLIP = "GRAD_NORM_CLIP"

    SAVE_OBJECT = "SAVE_OBJECT"
    SAVE_DICT = "SAVE_DICT"

    def __init__(
        self,
        root,
        model,
        train_dataset,
        n_epochs,
        batch_size,
        lr,
        optimizer_func=torch.optim.Adam,
        device="cuda",
        dtype=torch.float,
        grad_clip_type=None,
        grad_clip_value=None,
        save_type="SAVE_DICT",
        id=None,
        optimizer_kwargs={},
        loader_kwargs={},
    ):
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
        self.save_type = save_type
        self.optimizer_kwargs = optimizer_kwargs

        # Instantiate housekeeping variables
        self.id = str(uuid.uuid4().hex) if id is None else id
        self.log = {"train_loss": [], "duration": []}
        self.train_data_loader = torch.utils.data.DataLoader(
            self.train_dataset, self.batch_size, **loader_kwargs
        )

        self.optimizer = self.optimizer_func(
            self.model.parameters(), self.lr, **optimizer_kwargs
        )

        # Register grad clippings
        if self.grad_clip_type == Trainer.GRAD_VALUE_CLIP_PRE:

            for p in self.model.parameters():
                p.register_hook(
                    lambda grad: torch.clamp(grad, -grad_clip_value, grad_clip_value)
                )

        # Initialise the model
        self.model = self.model.to(device)
        if dtype == torch.float:
            self.model = self.model.float()
        elif dtype == torch.half:
            self.model = self.model.half()
        self.model.train()
        self.date = datetime.today().strftime("%Y-%m-%d-%H:%M:%S")

    @property
    def hyperparams(self):
        hyperparams = {
            "trainer": {
                "date": self.date,
                "n_epochs": self.n_epochs,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "dtype": str(self.dtype),
                "grad_clip_type": self.grad_clip_type,
                "grad_clip_value": self.grad_clip_value,
            },
            "dataset": self.train_dataset.hyperparams,
            "model": self.model.hyperparams,
        }

        return hyperparams

    @property
    def model_path(self):
        return os.path.join(self.root, self.id, "model.pt")

    @property
    def hyperparams_path(self):
        return os.path.join(self.root, self.id, "hyperparams.json")

    @property
    def log_path(self):
        return os.path.join(self.root, self.id, "log.csv")

    def save_model(self):
        if self.save_type == Trainer.SAVE_OBJECT:
            torch.save(self.model, self.model_path)
        elif self.save_type == Trainer.SAVE_DICT:
            torch.save(self.model.state_dict(), self.model_path)

    def save_hyperparams(self):
        with open(self.hyperparams_path, "w", encoding="utf-8") as f:
            json.dump(self.hyperparams, f, ensure_ascii=False, indent=4)

    def save_model_log(self):
        log_df = pd.DataFrame(self.log)
        log_df.to_csv(self.log_path, index=False)

    def loss(self, output, target, model):
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
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_value
                    )

                elif self.grad_clip_type == Trainer.GRAD_VALUE_CLIP_POST:
                    torch.nn.utils.clip_grad_value_(
                        self.model.parameters(), self.grad_clip_value
                    )

            self.optimizer.step()

        return epoch_loss

    def train(self, save=False):
        if save:
            os.mkdir(os.path.join(self.root, self.id))

        self.on_training_start(save)

        for epoch in range(self.n_epochs):
            # Train the model
            start_time = time.time()
            epoch_loss = self.train_for_single_epoch()
            end_time = time.time()
            epoch_duration = end_time - start_time
            logger.info(
                f"Completed epoch {epoch} with loss {epoch_loss} in {epoch_duration:.4f}s"
            )
            self.log["train_loss"].append(epoch_loss)
            self.log["duration"].append(epoch_duration)

            self.on_epoch_complete(save)

        self.on_training_complete(save)


def get_trainer(trainer_class, loss_function, **kwargs):
    model_trainer = trainer_class(**kwargs)
    model_trainer.loss = loss_function

    return model_trainer
