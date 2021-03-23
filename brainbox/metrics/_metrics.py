import torch


def cc(output, target):
    # output: b x 1 x t x n
    # target: b x 1 x t x n
    # returns: b x 1 x n

    exp_xy = (output * target).mean(dim=2)
    exp_x = output.mean(dim=2)
    exp_y = target.mean(dim=2)
    exp_x2 = (output ** 2).mean(dim=2)
    exp_y2 = (target ** 2).mean(dim=2)

    _cc = (exp_xy - exp_x * exp_y) / ((torch.sqrt(exp_x2 - exp_x ** 2)) * (torch.sqrt(exp_y2 - exp_y ** 2)))

    return _cc


def cc_norm(output, target):
    # output: b x 1 x t x n
    # target: b x r x t x n
    # returns: b x 1 x n
    n = target.shape[1]
    tp = (n - 1) * target.var(dim=2).sum(dim=1)
    sp = target.sum(dim=1).var(dim=1) - target.var(dim=2).sum(dim=1)
    recp_ccmax = torch.sqrt(1 + (1 / n) * (tp / sp - 1)).unsqueeze(1)

    target = target.mean(dim=1).unsqueeze(1)

    _cc = cc(output, target)
    _cc_norm = _cc * recp_ccmax

    return _cc_norm

# TODO rdm

