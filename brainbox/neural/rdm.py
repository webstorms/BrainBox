import torch
import scipy
import numpy as np

# Adapted from Patrick Mineault (https://github.com/patrickmineault/your-head-is-there-to-move-you-around)


def compute_rdm(x):
    # x: stimuli x neurons
    assert len(x.shape) == 2
    return torch.Tensor(1 - np.corrcoef(x))


def compute_rsm(x):
    # x: stimuli x neurons
    assert len(x.shape) == 2
    return torch.Tensor(np.corrcoef(x))


def similarity(mat1, mat2, method="spearman", deviation=False):
    assert mat1.shape[0] == mat1.shape[1]
    assert mat2.shape[0] == mat2.shape[1]
    assert mat1.shape == mat2.shape

    ii, jj = np.triu_indices(mat1.shape[0], k=1)
    if method == "pearson":
        corr = np.corrcoef(mat1[ii, jj], mat2[ii, jj])
        assert corr.shape[0] == 2
        corr = corr[0, 1]
    elif method == "spearman":
        corr = scipy.stats.spearmanr(mat1[ii, jj], mat2[ii, jj]).correlation
    elif method == "kendal":
        corr = scipy.stats.kendalltau(mat1[ii, jj], mat2[ii, jj]).correlation
    else:
        raise NotImplementedError(f"{method} not implemented")

    if not deviation:
        return corr
    else:
        return 1 - corr


def similarity_bootstrap(
    X, Y, method="spearman", nboot=100, sim_mat=True, deviation=False
):
    sim_func = compute_rsm if sim_mat else compute_rdm
    mat1 = sim_func(X)
    sim_scores = [
        similarity(
            mat1,
            sim_func(Y[:, np.random.randint(low=0, high=Y.shape[1], size=Y.shape[1])]),
            method,
            deviation,
        )
        for _ in range(nboot)
    ]

    return torch.Tensor(sim_scores)
