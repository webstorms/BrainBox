import sys
import logging

import torch
import pandas as pd

from brainbox.trainer import load_model

logger = logging.getLogger("validator")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def build_metric_df(
    root,
    model_ids,
    model_loader,
    dataset,
    metric,
    batch_size=128,
    device="cuda",
    dtype=torch.float,
):

    metric_list = []

    for model_id in model_ids:
        logger.info(f"Computing metric for {model_id}...")
        try:
            # model = model_loader(root, model_id, device, dtype)
            model = load_model(root, model_id, model_loader, device, dtype)
            metric_scores = compute_metric(
                model, dataset, metric, batch_size, device, dtype
            )

            for batch_id, metric_score in enumerate(metric_scores):
                row = {
                    "model_id": model_id,
                    "batch_id": batch_id,
                    "metric_score": metric_score,
                }
                metric_list.append(row)

        except Exception as error:
            logger.error(f"Failed computing metric for {model_id}: {error}")

    return pd.DataFrame(metric_list).set_index("model_id")


def compute_metric(
    model, dataset, metric, batch_size=128, device="cuda", dtype=torch.float
):
    metric_list = []

    data_loader = torch.utils.data.DataLoader(dataset, batch_size)

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device).type(dtype)
            target = target.to(device).type(dtype)
            output = model(data)
            metric_value = metric(output, target)
            metric_list.append(metric_value)

    return metric_list
