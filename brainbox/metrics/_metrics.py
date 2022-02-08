import torch
from scipy import stats
from sklearn.utils import resample


def cc(output, target, nan_val=0):
    # output: b x 1 x t x n
    # target: b x 1 x t x n
    # returns: b x 1 x n

    exp_xy = (output * target)._mean(dim=2)
    exp_x = output._mean(dim=2)
    exp_y = target._mean(dim=2)
    exp_x2 = (output ** 2)._mean(dim=2)
    exp_y2 = (target ** 2)._mean(dim=2)

    _cc = (exp_xy - exp_x * exp_y) / ((torch.sqrt(exp_x2 - exp_x ** 2)) * (torch.sqrt(exp_y2 - exp_y ** 2)))
    _cc[torch.isnan(_cc)] = nan_val

    return _cc


def cc_norm(output, target):
    # output: b x 1 x t x n
    # target: b x r x t x n
    # returns: b x 1 x n
    n = target.shape[1]
    tp = (n - 1) * target.var(dim=2).sum(dim=1)
    sp = target.sum(dim=1).var(dim=1) - target.var(dim=2).sum(dim=1)
    recp_ccmax = torch.sqrt(1 + (1 / n) * (tp / sp - 1)).unsqueeze(1)

    target = target._mean(dim=1).unsqueeze(1)

    _cc = cc(output, target)
    _cc_norm = _cc * recp_ccmax

    return _cc_norm


def rdm(responses, percentile=True):
    # responses: s x n

    def to_percentile(rdm):
        dim = rdm.shape[0]
        vec = rdm.flatten()

        return (stats.rankdata(vec, 'average') / len(vec)).reshape(dim, dim)

    s = responses.shape[0]
    n = responses.shape[1]

    rdm = torch.zeros((s, s))

    for i in range(s):
        resp_i = responses[i].view(1, 1, n, 1).repeat(s - i, 1, 1, 1)
        resp_j = responses[i:].view(s - i, 1, n, 1)

        v = cc(resp_j, resp_i)

        rdm[i, i:] = v[:, 0, 0]
        rdm[i:, i] = v[:, 0, 0]

    if percentile:
        rdm = torch.from_numpy(to_percentile(rdm.numpy()))

    return rdm


def rdm_deviation(rdm_model, rdm_target, metric='spearman', n_bootstrap=1):

    def flatten_rdm(rdm):
        s = rdm.shape[0]

        rdm_cells = []
        for i in range(s):
            rdm_cells.append(rdm[i, i:].flatten())

        return torch.cat(rdm_cells)

    flatten_rdm_model = flatten_rdm(rdm_model)
    flatten_rdm_target = flatten_rdm(rdm_target)

    idxs = list(range(len(flatten_rdm_model)))

    deviations = []

    for i in range(n_bootstrap):
        boot_idxs = resample(idxs, replace=True, n_samples=len(idxs))
        flatten_rdm_model_boot = flatten_rdm_model[boot_idxs]
        flatten_rdm_target_boot = flatten_rdm_target[boot_idxs]

        if metric == 'spearman':
            deviations.append(1 - stats.spearmanr(flatten_rdm_model_boot, flatten_rdm_target_boot)[0])
        else:
            raise NotImplementedError

    return deviations

