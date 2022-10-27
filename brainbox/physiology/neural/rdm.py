import torch
import scipy
import numpy as np

# Adapted from Patrick Mineault (https://github.com/patrickmineault/your-head-is-there-to-move-you-around)


def compute_rdm(X):
    assert len(X.shape) == 2
    return 1 - np.corrcoef(X)


def bootstrap_rdm_distance(X, Y, method="spearman", nboot=100):
    rdm_X = compute_rdm(X)
    rdm_Y = compute_rdm(Y)
    mean_deviation = compute_rdm_distance(rdm_X, rdm_Y, method)
    std_deviation = np.std(
        [
            compute_rdm_distance(
                rdm_X,
                compute_rdm(
                    Y[:, np.random.randint(low=0, high=Y.shape[1], size=Y.shape[1])]
                ),
                method,
            )
            for _ in range(nboot)
        ]
    )

    return mean_deviation, std_deviation


def compute_rdm_distance(rdm_X, rdm_Y, method="spearman"):
    assert rdm_X.shape[0] == rdm_X.shape[1]
    assert rdm_Y.shape[0] == rdm_Y.shape[1]
    assert rdm_X.shape == rdm_Y.shape

    ii, jj = np.triu_indices(rdm_X.shape[0], k=1)
    if method == "pearson":
        corr = np.corrcoef(rdm_X[ii, jj], rdm_Y[ii, jj])
        assert corr.shape[0] == 2
        corr = corr[0, 1]
    elif method == "spearman":
        corr = scipy.stats.spearmanr(rdm_X[ii, jj], rdm_Y[ii, jj]).correlation
    elif method == "kendal":
        corr = scipy.stats.kendalltau(rdm_X[ii, jj], rdm_Y[ii, jj]).correlation
    else:
        raise NotImplementedError(f"{method} not implemented")

    return 1 - corr
