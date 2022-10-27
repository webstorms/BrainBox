import torch


def cc(output, target):
    # output: t
    # target: t
    # returns: scalar

    exp_xy = (output * target).mean(dim=0)
    exp_x = output.mean(dim=0)
    exp_y = target.mean(dim=0)
    exp_x2 = (output**2).mean(dim=0)
    exp_y2 = (target**2).mean(dim=0)

    _cc = (exp_xy - exp_x * exp_y) / (
        (torch.sqrt(exp_x2 - exp_x**2)) * (torch.sqrt(exp_y2 - exp_y**2))
    )

    return _cc


def compute_max_cc(target):
    # target: t x n
    assert len(target.shape) == 2

    n = target.shape[1]
    np = (n - 1) * target.var(dim=0).sum()
    sp = target.sum(dim=1).var() - target.var(dim=0).sum()
    max_cc2 = 1 + 1 / n * (np / sp)

    return 1 / torch.sqrt(max_cc2) if max_cc2 > 0 else 1
