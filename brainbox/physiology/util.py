import sys
import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger("util")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def run_function_on_batch(
    function,
    dataset,
    batch_size,
    data_device="cuda",
    dtype=torch.float,
    max_batches=None,
    verbose=True,
    store_on_cpu=False,
    **kwargs,
):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size, **kwargs)

    output_list = []

    for i, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            if verbose:
                logger.info(f"Processing batch {i}...")
            data = data.to(data_device).type(dtype)
            target = target.to(data_device).type(dtype)

            output = function(data, target)

            if store_on_cpu:
                output = output.cpu()

            output_list.append(output)

            if i + 1 == max_batches:
                break

    return torch.cat(output_list, dim=0)


def get_mean(tensor, ignore_c=0):
    # b x n x t
    mask = tensor != ignore_c

    return tensor.sum(dim=(2)) / mask.sum(dim=2)


def get_std(tensor, ignore_c=0, mean=None):
    # b x n x t
    nonzero_mask = tensor != ignore_c
    zero_mask = tensor == ignore_c

    if mean is None:
        mean = get_mean(tensor, ignore_c)

    b_dim, n_dim, t_dim = tensor.shape
    mean_repeated = mean.view(b_dim, n_dim, 1).repeat(1, 1, t_dim)
    sqrd_error_with_zero = torch.pow(tensor - mean_repeated, 2).sum(dim=2)
    sqrd_error_from_mask = zero_mask.sum(dim=2) * torch.pow(mean, 2)

    return torch.sqrt(
        (sqrd_error_with_zero - sqrd_error_from_mask) / nonzero_mask.sum(dim=2)
    )


def cross_covariance_matrix(x, y, normalize=True):
    x_min_mean = x - x.mean(0)
    y_min_mean = y - y.mean(0)

    b_dim = x.shape[0]
    cross_covariance_matrix_batch = torch.einsum(
        "bni, bnj -> nij", x_min_mean, y_min_mean
    )
    _cross_covariance_matrix = (
        cross_covariance_matrix_batch / b_dim
    )  # Average across samples / batch dim

    if normalize:
        inv_x_std = torch.nan_to_num(1 / x.std(0, unbiased=False))
        inv_y_std = torch.nan_to_num(1 / y.std(0, unbiased=False))
        return torch.einsum(
            "nij, ni, nj -> nij", _cross_covariance_matrix, inv_x_std, inv_y_std
        )

    return _cross_covariance_matrix
