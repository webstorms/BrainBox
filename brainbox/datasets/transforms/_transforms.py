import torch


class Compose:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for trans in self.transforms:
            x = trans(x)

        return x

    @property
    def hyperparams(self):
        hyperparams = {}
        for trans in self.transforms:
            hyperparams = {**hyperparams, **trans.hyperparams}

        return hyperparams


class ClipNormalize:

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, clip):

        if not self.inplace:
            clip = clip.clone()

        dtype = clip.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=clip.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=clip.device)
        clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])

        return clip

    @property
    def hyperparams(self):
        return {'clip_normalize': {'mean': self.mean, 'std': self.std}}


class ClipGrayscale:

    def __call__(self, clip):
        r, g, b = clip.unbind(dim=0)
        clip = (0.2989 * r + 0.587 * g + 0.114 * b).unsqueeze(dim=0)

        return clip

    @property
    def hyperparams(self):
        return {'clip_grayscale': {}}


class ClipResize:

    # TODO
    pass

