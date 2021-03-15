import torch


class FastSigmoid(torch.autograd.Function):

    """
    Use the normalized negative part of a fast sigmoid for the surrogate gradient
    as done in Zenke & Ganguli (2018).
    """

    scale = 100.0  # controls steepness of surrogate gradient

    @classmethod
    def get(cls, scale):
        cls.scale = scale

        return cls.apply

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0

        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (FastSigmoid.scale * torch.abs(input) + 1.0) ** 2

        return grad
