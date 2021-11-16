import os
import glob
import itertools

import torch
import pandas as pd

from brainbox.trainer import Trainer


def append_scores(root, model_hyperparams, scoring_dict, dataset, batch_size, device, dtype):

    scores = []
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size)

    def get_score(model, metric):

        score = 0

        for data, target in dataset_loader:
            data = data.to(device).type(dtype)
            target = target.to(device).type(dtype)

            with torch.no_grad():
                output = model(data)
                score += metric(output, target)

        return score

    for i, model_id in enumerate(model_hyperparams.index):
        print('Processing model {0}/{1}...'.format(i, len(model_hyperparams.index)))
        try:
            model = load_model(root, model_id)
            model_id_scores = model_hyperparams.loc[model_id].to_dict()
            model_id_scores['id'] = model_id

            for score_name in scoring_dict:
                metric = scoring_dict[score_name]
                model_id_scores[score_name] = get_score(model, metric).item()
            scores.append(model_id_scores)

            if 'cuda' in device:
                del model
                torch.cuda.empty_cache()
        except:
            print(model_id)

    return pd.DataFrame(scores)


def get_best_models(groups, scores_df, eval_name, criterion=min, attach=[]):

    group_lists = [list(scores_df[group].unique()) for group in groups]

    results = []

    for product in itertools.product(*group_lists):
        query = True
        for i in range(len(product)):
            query &= scores_df[groups[i]] == product[i]

        min_i = scores_df[query][eval_name].idxmin() if criterion == 'min' else scores_df[query][eval_name].idxmax()
        results.append(
            {**{groups[i]: product[i] for i in range(len(product))}, eval_name: scores_df[query][eval_name].loc[min_i],
             **{att: scores_df[query][att].loc[min_i] for att in attach}})

    return pd.DataFrame(results)


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


def load_hyperparams(root, model_id=None, **kwargs):
    if model_id is None:
        model_ids = query_model_ids(root, **kwargs)
        assert len(model_ids) == 1, 'Multiple models match the query criteria'
        model_id = model_ids.index[0]

    model_hyperparams_path = os.path.join(root, '{0}_hyperparams.csv'.format(model_id))

    return pd.read_csv(model_hyperparams_path)


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

