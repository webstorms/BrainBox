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


class ImgToClip:

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
        return {'img_to_clip': {'pre_blanks': self.pre_blanks, 'repeats': self.repeats, 'post_blanks': self.post_blanks}}

