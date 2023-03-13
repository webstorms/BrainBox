import torch


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
    sp = target.sum(dim=1).var(unbiased=unbiased) - target.var(dim=0, unbiased=unbiased).sum()
    max_cc2 = 1 + 1 / n * (np / sp)

    return 1 / torch.sqrt(max_cc2) if max_cc2 > 0 else eps


def compute_max_spe(target, unbiased=True):
    # target: t x n
    assert len(target.shape) == 2

    n = target.shape[1]
    sp = (1 / (n - 1)) * (n * target.mean(dim=1).var(unbiased=unbiased) - target.var(dim=0, unbiased=unbiased).mean())
    rp = target.mean(1).var(unbiased=unbiased)

    return sp / rp
