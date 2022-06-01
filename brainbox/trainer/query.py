import os
import json
import glob
import itertools

import torch
import pandas as pd

from brainbox.trainer import Trainer


def get_model_ids(root):
    return os.listdir(root)


def load_model(root, id, model_loader, device="cuda", dtype=torch.float):
    # model_loader: Create model instance from hyperparams
    model_path = os.path.join(root, id, "model.pt")
    model = model_loader(load_hyperparams(root, id))
    model.load_state_dict(torch.load(model_path))

    return model.to(device).type(dtype)


def load_log(root, id):
    log_path = os.path.join(root, id, "log.csv")

    return pd.read_csv(log_path)


def load_hyperparams(root, id):
    hyperparams_path = os.path.join(root, id, "hyperparams.json")
    with open(hyperparams_path) as f:
        hyperparams = json.load(f)

    return hyperparams


def build_models_df(root, model_ids, hyperparams_mapper):
    # hyperparams_mapper: Map hyperparams dict to columns dict
    models_list = []

    for model_id in model_ids:
        hyperparams = load_hyperparams(root, model_id)
        model_columns = hyperparams_mapper(hyperparams)
        row = {"model_id": model_id, **model_columns}
        models_list.append(row)

    return pd.DataFrame(models_list).set_index("model_id")
