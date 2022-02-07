import torch
import torch.nn.functional as F
import numpy as np


class BBTransform:

    @property
    def hyperparams(self):
        hyperparams = {'name': self.__class__.__name__}

        return hyperparams


class Compose(BBTransform):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for trans in self.transforms:
            x = trans(x)

        return x

    @property
    def hyperparams(self):
        hyperparams = {**super().hyperparams, 'transforms': [trans.hyperparams for trans in self.transforms]}

        return hyperparams


class ClipNormalize(BBTransform):

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
        hyperparams = {**super().hyperparams, 'mean': self.mean, 'std': self.std}

        return hyperparams


class ClipGrayscale(BBTransform):

    def __call__(self, clip):
        r, g, b = clip.unbind(dim=0)
        clip = (0.2989 * r + 0.587 * g + 0.114 * b).unsqueeze(dim=0)

        return clip


class ClipColorfy(BBTransform):

    def __call__(self, clip):

        return clip.repeat(3, 1, 1, 1)


class GaussianKernel(BBTransform):

    def __init__(self, sigma, width):
        assert width % 2 == 1, 'width needs to be an odd number'
        self.sigma = sigma
        self.width = width

        kernel = [(1 / ((np.sqrt(2 * np.pi) * sigma)) * np.exp(-x ** 2 / (2 * sigma ** 2))) for x in
                  np.arange(-width // 2 + 1, width // 2 + 1, 1)]
        self.kernel = torch.Tensor(kernel).view(1, 1, width, 1)

    def __call__(self, x):
        # x: channel x time x n_neurons
        x = x.unsqueeze(0)  # add batch dimension
        x = F.pad(x, (0, 0, self.width // 2, self.width // 2))
        x = F.conv2d(x, self.kernel)

        return x[0]

    @property
    def hyperparams(self):
        hyperparams = {**super().hyperparams, 'sigma': self.sigma, 'width': self.width}

        return hyperparams


class ClipResize(BBTransform):

    def __init__(self, k, s, p):
        self.k = k
        self.s = s
        self.p = p
        self._filters = torch.ones(k, k).view(1, 1, 1, k, k) / (k**2)

    def __call__(self, clip):
        return F.conv3d(clip.unsqueeze(0), self._filters, stride=(1, self.s, self.s), padding=(0, self.p, self.p))[0]

    @property
    def hyperparams(self):
        hyperparams = {**super().hyperparams, 'k': self.k, 's': self.s, 'p': self.p}

        return hyperparams


class ClipBound(BBTransform):

    def __init__(self, vmin, vmax):
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, clip):
        return torch.clamp(clip, min=self.vmin, max=self.vmax)

    @property
    def hyperparams(self):
        hyperparams = {**super().hyperparams, 'vmin': self.vmin, 'vmax': self.vmax}

        return hyperparams


class ClipCrop(BBTransform):

    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, clip):
        if self.h > 0:
            clip = clip[:, :, self.h:-self.h, :]
        if self.w > 0:
            clip = clip[:, :, :, self.w:-self.w]

        return clip

    @property
    def hyperparams(self):
        hyperparams = {**super().hyperparams, 'h': self.h, 'w': self.w}

        return hyperparams


class ImgToClip(BBTransform):

    def __init__(self, pre_blanks, repeats, post_blanks):
        self.pre_blanks = pre_blanks
        self.repeats = repeats
        self.post_blanks = post_blanks

    def __call__(self, img):
        # img: channel x height x width
        # output: channel x time x height x width
        assert len(img.shape) == 3
        channel = img.shape[0]
        height = img.shape[1]
        width = img.shape[2]

        clip = torch.zeros((channel, self.pre_blanks + self.repeats + self.post_blanks, height, width))
        clip[:, self.pre_blanks:self.pre_blanks + self.repeats] = img

        return clip

    @property
    def hyperparams(self):
        hyperparams = {**super().hyperparams, 'pre_blanks': self.pre_blanks, 'repeats': self.repeats, 'post_blanks': self.post_blanks}

        return hyperparams


class RandomHorizontalFlipClip(BBTransform):

    def __call__(self, clip):
        if torch.rand(1).item() > 0.5:
            return torch.flip(clip, dims=(3,))
        else:
            return clip
