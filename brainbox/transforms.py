import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BBTransform:
    @property
    def hyperparams(self):
        hyperparams = {"name": self.__class__.__name__}

        return hyperparams


class Compose(BBTransform):
    def __init__(self, transforms):
        self._transforms = transforms

    def __call__(self, x):
        for trans in self._transforms:
            x = trans(x)

        return x

    @property
    def hyperparams(self):
        hyperparams = {
            **super().hyperparams,
            "transforms": [trans.hyperparams for trans in self._transforms],
        }

        return hyperparams


class ClipNormalize(BBTransform):
    def __init__(self, mean, std, inplace=False):
        self._mean = mean
        self._std = std
        self._inplace = inplace

    def __call__(self, clip):

        if not self._inplace:
            clip = clip.clone()

        dtype = clip.dtype
        mean = torch.as_tensor(self._mean, dtype=dtype, device=clip.device)
        std = torch.as_tensor(self._std, dtype=dtype, device=clip.device)
        clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])

        return clip

    @property
    def hyperparams(self):
        hyperparams = {**super().hyperparams, "mean": self._mean, "std": self._std}

        return hyperparams


class ClipGrayscale(BBTransform):
    def __call__(self, clip):
        r, g, b = clip.unbind(dim=0)
        clip = (0.2989 * r + 0.587 * g + 0.114 * b).unsqueeze(dim=0)

        return clip


class ClipColorfy(BBTransform):
    def __call__(self, clip):
        return clip.repeat(3, 1, 1, 1)


class ClipResize(BBTransform):
    def __init__(self, kernel, stride, padding):
        self._kernel = kernel
        self._stride = stride
        self._padding = padding

        self._filters = torch.ones(kernel, kernel).view(1, 1, 1, kernel, kernel) / (
            kernel**2
        )

    def __call__(self, clip):
        self._filters = self._filters.to(clip.device)
        return F.conv3d(
            clip.unsqueeze(0),
            self._filters,
            stride=(1, self._stride, self._stride),
            padding=(0, self._padding, self._padding),
        )[0]

    @property
    def hyperparams(self):
        hyperparams = {
            **super().hyperparams,
            "kernel": self._kernel,
            "stride": self._stride,
            "padding": self._padding,
        }

        return hyperparams


class ClipBound(BBTransform):
    def __init__(self, min_value, max_value):
        self._min_value = min_value
        self._max_value = max_value

    def __call__(self, clip):
        return torch.clamp(clip, min=self._min_value, max=self._max_value)

    @property
    def hyperparams(self):
        hyperparams = {
            **super().hyperparams,
            "min_value": self._min_value,
            "max_value": self._max_value,
        }

        return hyperparams


class ClipCrop(BBTransform):
    def __init__(self, h, w):
        self._h = h
        self._w = w

    def __call__(self, clip):
        if self._h > 0:
            clip = clip[:, :, self._h : -self._h, :]
        if self._w > 0:
            clip = clip[:, :, :, self._w : -self._w]

        return clip

    @property
    def hyperparams(self):
        hyperparams = {**super().hyperparams, "h": self._h, "w": self._w}

        return hyperparams


class ClipRandomHorizontalFlip(BBTransform):
    def __call__(self, clip):
        if torch.rand(1).item() > 0.5:
            return torch.flip(clip, dims=(3,))
        else:
            return clip


class ClipExtend(BBTransform):
    def __init__(self, frames):
        self._frames = frames
        self._kernel = torch.ones(1, 1, frames, 1, 1)

    def __call__(self, clip):
        if self._kernel.device != clip.device:
            self._kernel = self._kernel.to(clip.device)
        return F.conv_transpose3d(clip, self._kernel, stride=(self._frames, 1, 1))

    @property
    def hyperparams(self):
        hyperparams = {**super().hyperparams, "frames": self._frames}

        return hyperparams


class ClipShrink(BBTransform):
    def __init__(self, frames):
        self._frames = frames
        self._kernel = torch.ones(1, 1, frames, 1, 1)

    def __call__(self, clip):
        if self._kernel.device != clip.device:
            self._kernel = self._kernel.to(clip.device)
        return F.conv3d(clip, self._kernel, stride=(self._frames, 1, 1))

    @property
    def hyperparams(self):
        hyperparams = {**super().hyperparams, "frames": self._frames}

        return hyperparams


class ImgToClip(BBTransform):
    def __init__(self, pre_blanks, repeats, post_blanks, c=0):
        self._pre_blanks = pre_blanks
        self._repeats = repeats
        self._post_blanks = post_blanks
        self._c = c

    def __call__(self, img):
        if len(img.shape) == 3:
            # img: channel x height x width
            # output: channel x time x height x width
            channel = img.shape[0]
            height = img.shape[1]
            width = img.shape[2]

            clip = self._c * torch.ones(
                (
                    channel,
                    self._pre_blanks + self._repeats + self._post_blanks,
                    height,
                    width,
                )
            )
            clip[:, self._pre_blanks : self._pre_blanks + self._repeats] = img
        elif len(img.shape) == 4:
            # img: batch x channel x height x width
            # output: batch x channel x time x height x width
            batch = img.shape[0]
            channel = img.shape[1]
            height = img.shape[2]
            width = img.shape[3]

            clip = self._c * torch.ones(
                (
                    batch,
                    channel,
                    self._pre_blanks + self._repeats + self._post_blanks,
                    height,
                    width,
                )
            )
            clip[:, :, self._pre_blanks : self._pre_blanks + self._repeats] = (
                img.unsqueeze(2)
            )  # Add fake time dim to img

        return clip

    @property
    def hyperparams(self):
        hyperparams = {
            **super().hyperparams,
            "pre_blanks": self._pre_blanks,
            "repeats": self._repeats,
            "post_blanks": self._post_blanks,
        }

        return hyperparams


class GaussianKernel(BBTransform):
    def __init__(self, sigma, width):
        assert width % 2 == 1, "width needs to be an odd number"
        self._sigma = sigma
        self._width = width

        self._kernel = self._build_kernel(sigma, width)

    def __call__(self, x):
        if len(x.shape) == 3:  # x: channel x time x n_neurons
            x = x.unsqueeze(0)  # add batch dimension
            x = F.pad(x, (0, 0, self._width // 2, self._width // 2))
            x = F.conv2d(x, self._kernel)

            return x[0]
        else:  # x: b x channel x time x n_neurons
            x = F.pad(x, (0, 0, self._width // 2, self._width // 2))
            x = F.conv2d(x, self._kernel)

            return x

    @property
    def hyperparams(self):
        hyperparams = {
            **super().hyperparams,
            "sigma": self._sigma,
            "width": self._width,
        }

        return hyperparams

    def _build_kernel(self, sigma, width):
        kernel = [
            (1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-((x) ** 2) / (2 * sigma**2)))
            for x in np.arange(-width // 2 + 1, width // 2 + 1, 1)
        ]
        return torch.Tensor(kernel).view(1, 1, width, 1)


class HanningKernel(BBTransform):
    def __init__(self, length):
        super().__init__()
        self._length = length
        self._kernel = nn.Parameter(
            torch.from_numpy(np.hanning(length)).view(1, 1, 1, length).float(),
            requires_grad=False,
        )

    def __call__(self, x):
        # x: b x n x t
        self._kernel = self._kernel.to(x.device)

        x = x.unsqueeze(1)  # add channel dimension
        x = F.pad(
            x, (self._length // 2, self._length // 2)
        )  # Bad input to ensure valid conv
        x = F.conv2d(x, self._kernel)
        x = x.squeeze(1)

        return x

    @property
    def hyperparams(self):
        hyperparams = {
            **super().hyperparams,
            "length": self._length,
        }

        return hyperparams
