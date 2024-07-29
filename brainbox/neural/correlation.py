import torch


def R2(output, target):
    # output: T
    # target: T
    assert len(output.shape) == 1
    assert len(target.shape) == 1
    return (
        1
        - torch.pow(target - output, 2).sum()
        / torch.pow((target - target.mean()), 2).sum()
    )


def cc(output, target):
    assert output.shape == target.shape
    # output: nxt
    # target: nxt
    # returns: n

    exp_xy = (output * target).mean(dim=1)
    exp_x = output.mean(dim=1)
    exp_y = target.mean(dim=1)
    exp_x2 = (output**2).mean(dim=1)
    exp_y2 = (target**2).mean(dim=1)

    _cc = (exp_xy - exp_x * exp_y) / (
        (torch.sqrt(exp_x2 - exp_x**2)) * (torch.sqrt(exp_y2 - exp_y**2))
    )

    return _cc


def compute_max_cc(target, eps=1e-3, unbiased=True):
    # target: t x n
    assert len(target.shape) == 2

    n = target.shape[1]
    np = (n - 1) * target.var(dim=0, unbiased=unbiased).sum()
    sp = (
        target.sum(dim=1).var(unbiased=unbiased)
        - target.var(dim=0, unbiased=unbiased).sum()
    )
    max_cc2 = 1 + 1 / n * (np / sp)

    return 1 / torch.sqrt(max_cc2) if max_cc2 > 0 else eps


def compute_max_spe(target, unbiased=True):
    # target: t x n
    assert len(target.shape) == 2

    n = target.shape[1]
    sp = (1 / (n - 1)) * (
        n * target.mean(dim=1).var(unbiased=unbiased)
        - target.var(dim=0, unbiased=unbiased).mean()
    )
    rp = target.mean(1).var(unbiased=unbiased)

    return sp / rp


def compute_max_response_reliability(x):
    # x: clips x time x repeat x neuron
    n_trials = x.shape[2]
    mean_response = x.mean(2).unsqueeze(2).repeat(1, 1, n_trials, 1)

    flatten_mean_response = mean_response.flatten(0, 2).permute(1, 0)
    flatten_trial_response = x.flatten(0, 2).permute(1, 0)

    return cc(flatten_mean_response, flatten_trial_response)


def nc(y):
    # y: neurons x repeat x time
    trial_avg = y.mean(1)
    neuron_avg = trial_avg.mean(1)
    total_diff = y - neuron_avg.view(-1, 1, 1)
    noise_diff = y - trial_avg.unsqueeze(1)
    N, K, T = y.shape
    M_total = (1 / (T * K)) * torch.einsum("ikt, jkt -> ij", total_diff, total_diff)
    M_noise = (1 / (T * K)) * torch.einsum("ikt, jkt -> ij", noise_diff, noise_diff)

    M_total_diag = torch.Tensor([M_total[i, i] for i in range(N)])
    norm_M_noise = M_noise * torch.sqrt(
        torch.einsum("i, j -> ij", M_total_diag, M_total_diag)
    )
    for i in range(N):
        norm_M_noise[i, i] = 0

    return norm_M_noise
