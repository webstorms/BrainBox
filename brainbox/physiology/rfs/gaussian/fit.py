import sys
import logging
import itertools

import pandas as pd
import numpy as np
import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F

import brainbox.physiology.rfs.rfs as rfs_util

logger = logging.getLogger("gabor")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class GaborFitter:

    GABOR_AMPS = [-1, 1]
    GABOR_SIGMAS = [0.5, 1, 1.5]
    GABOR_PS = [-0.5, 0, 0.5]

    def __init__(self):
        pass

    def fit_spatial(
            self,
            path,
            rfs,
            batch_size,
            n_spectral_iterations=2000,
            n_spatial_iterations=2000,
            spectral_lr=1e-2,
            spatial_lr=1e-2,
            device="cuda",
            **kwargs,
    ):
        batch_params = []

        iterations = rfs.shape[0] // batch_size
        iterations += 1 if rfs.shape[0] > iterations * batch_size else 0

        for b in range(iterations):
            logger.info(f"Fitting gabors for batch {b}...")
            params = self.fit_spatial_single_batch(
                rfs[b * batch_size : (b + 1) * batch_size],
                n_spectral_iterations,
                n_spatial_iterations,
                spectral_lr,
                spatial_lr,
                device,
                **kwargs,
            )
            batch_params.append(torch.stack(params))

        all_params = torch.cat(batch_params, dim=1).cpu()
        all_params_df = pd.DataFrame(
            {
                "amp": all_params[0],
                "x0": all_params[1],
                "y0": all_params[2],
                "sigmax": all_params[3],
                "sigmay": all_params[4],
                "p": all_params[5]
            }
        )
        all_params_df.to_csv(path, index=False)

    def fit_spatial_single_batch(
            self,
            rfs,
            n_spectral_iterations=1000,
            n_spatial_iterations=2000,
            spectral_lr=1e-2,
            spatial_lr=1e-3,
            device="cuda",
            **kwargs,
    ):
        n_rfs = rfs.shape[0]
        rf_size = rfs.shape[-1]
        initial_gabor_params = GaborFitter._build_initial_gabor_params(
            rf_size, n_rfs, **kwargs
        )
        n_params = len(initial_gabor_params[0]) // n_rfs
        normalized_rfs = torch.repeat_interleave(
            rfs_util.normalize_rfs(rfs), n_params, 0
        ).to(device)

        # spectral_fitter = SpectralGaussianFitter(
        #     normalized_rfs,
        #     n_spectral_iterations,
        #     spectral_lr,
        #     device,
        #     rf_size,
        #     *initial_gabor_params,
        # )
        # spectral_fitter.fit()
        spatial_fitter = SpatialGaussianFitter(
            normalized_rfs,
            n_spatial_iterations,
            spatial_lr,
            device,
            rf_size,
            *initial_gabor_params,
        )
        spatial_fitter.fit()

        return GaborFitter._get_best_fits(
            normalized_rfs, spatial_fitter.gaussian_model, n_params, n_rfs
        )

    # def fit_spatiotemporal(
    #     self,
    #     strfs,
    #     gabor_params,
    #     n_spatial_iterations=100,
    #     spatial_lr=1e-3,
    #     device="cuda",
    # ):
    #     rf_size = strfs.shape[-1]
    #     normalized_strfs = rfs_util.normalize_strfs(strfs).flatten(
    #         start_dim=0, end_dim=1
    #     )
    #     spatial_fitter = SpatialFitter(
    #         normalized_strfs,
    #         True,
    #         n_spatial_iterations,
    #         spatial_lr,
    #         device,
    #         rf_size,
    #         *gabor_params,
    #     )
    #     spatial_fitter.fit()
    #
    #     return spatial_fitter.gabor_model.cpu()

    @staticmethod
    def _build_initial_gabor_params(rf_size, repeat, **kwargs):
        amps_init = kwargs.get("sigmax_init", GaborFitter.GABOR_AMPS)
        x0_init, y0_init = [rf_size // 2], [rf_size // 2]
        sigmax_init = kwargs.get("sigmax_init", GaborFitter.GABOR_SIGMAS)
        sigmay_init = kwargs.get("sigmay_init", GaborFitter.GABOR_SIGMAS)
        p_init = kwargs.get("p_init", GaborFitter.GABOR_PS)

        params_product = list(
            itertools.product(
                amps_init,
                x0_init,
                y0_init,
                sigmax_init,
                sigmay_init,
                p_init
            )
        )
        amps_init = torch.Tensor(repeat * [v[0] for v in params_product])
        x0_init = torch.Tensor(repeat * [v[1] for v in params_product])
        y0_init = torch.Tensor(repeat * [v[2] for v in params_product])
        sigmax_init = torch.Tensor(repeat * [v[3] for v in params_product])
        sigmay_init = torch.Tensor(repeat * [v[4] for v in params_product])
        p_init = torch.Tensor(repeat * [v[5] for v in params_product])

        return (
            amps_init,
            x0_init,
            y0_init,
            sigmax_init,
            sigmay_init,
            p_init
        )

    @staticmethod
    def _extract_gabor_params(gabor_model):
        x0_init = gabor_model.x0.data
        y0_init = gabor_model.y0.data
        sigmax_init = gabor_model.sigmax.data
        sigmay_init = gabor_model.sigmay.data
        p_init = gabor_model.p.data

        return (
            x0_init,
            y0_init,
            sigmax_init,
            sigmay_init,
            p_init,
        )

    @staticmethod
    def _get_best_fits(repeated_rfs, gabor_model, n_params, n_rfs):
        predictions = gabor_model()
        losses = (
            F.mse_loss(repeated_rfs, predictions, reduction="none")
            .mean(dim=(1, 2))
            .view(n_rfs, n_params)
        )
        best_idxs = torch.argmin(losses, dim=1) + torch.Tensor(
            [n_params * i for i in range(n_rfs)]
        ).int().to(repeated_rfs.device)

        amp = gabor_model.amplitude.data[best_idxs]
        x0 = gabor_model.x0.data[best_idxs]
        y0 = gabor_model.y0.data[best_idxs]
        sigmax = gabor_model.sigmax.data[best_idxs]
        sigmay = gabor_model.sigmay.data[best_idxs]
        p = gabor_model.p.data[best_idxs]

        return amp, x0, y0, sigmax, sigmay, p


class SpatialGaussianFitter:
    def __init__(
        self,
        rfs,
        n_iterations,
        lr,
        device,
        rf_size,
        amp_init,
        x0_init,
        y0_init,
        sigmax_init,
        sigmay_init,
        p_init
    ):
        self._rfs = rfs
        self._n_iterations = n_iterations

        self._gaussian_model = Gaussian2D(
            rf_size,
            amp_init,
            x0_init,
            y0_init,
            sigmax_init,
            sigmay_init,
            p_init
        ).to(device)
        self._optimizer = torch.optim.Adam(self._gaussian_model.parameters(), lr)

    @property
    def gaussian_model(self):
        return self._gaussian_model

    def prediction(self):
        return self._gaussian_model()

    def fit(self):
        for _ in range(self._n_iterations):
            self._optimizer.zero_grad()
            loss = F.mse_loss(self._rfs, self.prediction())
            loss.backward()
            #print(f"loss={loss.item()}")
            self._optimizer.step()


class SpectralGaussianFitter(SpatialGaussianFitter):
    def __init__(
            self,
            rfs,
            n_iterations,
            lr,
            device,
            rf_size,
            amp_init,
            x0_init,
            y0_init,
            sigmax_init,
            sigmay_init,
            p_init
    ):
        super().__init__(
            rfs,
            n_iterations,
            lr,
            device,
            rf_size,
            amp_init,
            x0_init,
            y0_init,
            sigmax_init,
            sigmay_init,
            p_init
        )
        self._rfs = SpectralGaussianFitter._to_spectral(rfs)

    def prediction(self):
        return SpectralGaussianFitter._to_spectral(self.gaussian_model())

    @staticmethod
    def _to_spectral(rfs, s=[100, 100]):
        return rfs_util.normalize_rfs(fft.fftn(rfs, s=s, dim=(1, 2)).abs())


class Gaussian2D(nn.Module):

    def __init__(self, rf_size=20, amp_init=None, x0_init=None, y0_init=None, sigmax_init=None, sigmay_init=None, p_init=None, eps=1e-4):
        super().__init__()
        self._amplitude = nn.Parameter(amp_init)
        self._x0 = nn.Parameter(x0_init)
        self._y0 = nn.Parameter(y0_init)
        self._sigmax = nn.Parameter(sigmax_init)
        self._sigmay = nn.Parameter(sigmay_init)
        self._p = nn.Parameter(p_init)
        self._eps = eps

        y, x = torch.meshgrid(
            [
                torch.arange(rf_size, dtype=torch.float32),
                torch.arange(rf_size, dtype=torch.float32),
            ]
        )
        self.y = nn.Parameter(y, requires_grad=False)
        self.x = nn.Parameter(x, requires_grad=False)
        self._rf_ghost = nn.Parameter(torch.ones(rf_size, rf_size), requires_grad=False)

    @property
    def amplitude(self):
        return self._amplitude#torch.clamp(self._amplitude, min=-1, max=1)

    @property
    def x0(self):
        return self._x0

    @property
    def y0(self):
        return self._y0

    @property
    def sigmax(self):
        return torch.clamp(self._sigmax, min=self._eps, max=np.inf)

    @property
    def sigmay(self):
        return torch.clamp(self._sigmay, min=self._eps, max=np.inf)

    @property
    def p(self):
        return torch.clamp(self._p, min=-0.99, max=0.99)

    def forward(self):
        amplitude = self._expand_shape(self.amplitude)
        x0 = self._expand_shape(self.x0)
        y0 = self._expand_shape(self.y0)
        sigmax = self._expand_shape(self.sigmax)
        sigmay = self._expand_shape(self.sigmay)
        p = self._expand_shape(self.p)

        x = self.x - x0
        y = self.y - y0
        gaussian = (1 / (2 * np.pi * sigmax * sigmay * torch.sqrt(1 - p**2))) * torch.exp(-(1 / (2 * (1 - p**2))) * ((x / sigmax)**2 -2 * p * (x / sigmax) * (y / sigmay) + (y / sigmay) ** 2))

        return amplitude * gaussian

    def _expand_shape(self, vec):
        return torch.einsum("n, ij -> nij", vec, self._rf_ghost)
