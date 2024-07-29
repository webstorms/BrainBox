import sys
import logging
import itertools

import pandas as pd
import numpy as np
import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F

import brainbox.rfs.rfs as rfs_util

logger = logging.getLogger("gabor")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class GaborFitter:

    GABOR_SIGMAS = [0.5, 1.5]
    GABOR_THETAS = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
    GABOR_PHIS = [0, np.pi / 2, np.pi]
    GABOR_FREQUENCIES = [0.05, 0.1]

    def __init__(self):
        pass

    def fit_spatial(
        self,
        path,
        rfs,
        batch_size,
        n_spectral_iterations=1000,
        n_spatial_iterations=2000,
        spectral_lr=1e-2,
        spatial_lr=1e-3,
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
                "x0": all_params[0],
                "y0": all_params[1],
                "sigmax": all_params[2],
                "sigmay": all_params[3],
                "theta": all_params[4],
                "phi": all_params[5],
                "frequency": all_params[6],
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

        spectral_fitter = SpectralFitter(
            normalized_rfs,
            False,
            n_spectral_iterations,
            spectral_lr,
            device,
            rf_size,
            *initial_gabor_params,
        )
        spectral_fitter.fit()
        spatial_fitter = SpatialFitter(
            normalized_rfs,
            False,
            n_spatial_iterations,
            spatial_lr,
            device,
            rf_size,
            *GaborFitter._extract_gabor_params(spectral_fitter.gabor_model),
        )
        spatial_fitter.fit()

        return GaborFitter._get_best_fits(
            normalized_rfs, spatial_fitter.gabor_model, n_params, n_rfs
        )

    def fit_spatiotemporal(
        self,
        strfs,
        gabor_params,
        n_spatial_iterations=100,
        spatial_lr=1e-3,
        device="cuda",
    ):
        rf_size = strfs.shape[-1]
        normalized_strfs = rfs_util.normalize_strfs(strfs).flatten(
            start_dim=0, end_dim=1
        )
        spatial_fitter = SpatialFitter(
            normalized_strfs,
            True,
            n_spatial_iterations,
            spatial_lr,
            device,
            rf_size,
            *gabor_params,
        )
        spatial_fitter.fit()

        return spatial_fitter.gabor_model.cpu()

    @staticmethod
    def _build_initial_gabor_params(rf_size, repeat, **kwargs):
        x0_init, y0_init = [rf_size // 2], [rf_size // 2]
        sigmax_init = kwargs.get("sigmax_init", GaborFitter.GABOR_SIGMAS)
        sigmay_init = kwargs.get("sigmay_init", GaborFitter.GABOR_SIGMAS)
        theta_init = kwargs.get("theta_init", GaborFitter.GABOR_THETAS)
        phi_init = kwargs.get("phi_init", GaborFitter.GABOR_PHIS)
        frequency_init = kwargs.get("frequency_init", GaborFitter.GABOR_FREQUENCIES)

        params_product = list(
            itertools.product(
                x0_init,
                y0_init,
                sigmax_init,
                sigmay_init,
                theta_init,
                phi_init,
                frequency_init,
            )
        )
        x0_init = torch.Tensor(repeat * [v[0] for v in params_product])
        y0_init = torch.Tensor(repeat * [v[1] for v in params_product])
        sigmax_init = torch.Tensor(repeat * [v[2] for v in params_product])
        sigmay_init = torch.Tensor(repeat * [v[3] for v in params_product])
        theta_init = torch.Tensor(repeat * [v[4] for v in params_product])
        phi_init = torch.Tensor(repeat * [v[5] for v in params_product])
        frequency_init = torch.Tensor(repeat * [v[6] for v in params_product])

        return (
            x0_init,
            y0_init,
            sigmax_init,
            sigmay_init,
            theta_init,
            phi_init,
            frequency_init,
        )

    @staticmethod
    def _extract_gabor_params(gabor_model):
        x0_init = gabor_model.x0.data
        y0_init = gabor_model.y0.data
        sigmax_init = gabor_model.sigmax.data
        sigmay_init = gabor_model.sigmay.data
        theta_init = gabor_model.theta.data
        phi_init = gabor_model.phi.data
        frequency_init = gabor_model.frequency.data

        return (
            x0_init,
            y0_init,
            sigmax_init,
            sigmay_init,
            theta_init,
            phi_init,
            frequency_init,
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

        x0 = gabor_model.x0.data[best_idxs]
        y0 = gabor_model.y0.data[best_idxs]
        sigmax = gabor_model.sigmax.data[best_idxs]
        sigmay = gabor_model.sigmay.data[best_idxs]
        theta = gabor_model.theta.data[best_idxs]
        phi = gabor_model.phi.data[best_idxs]
        frequency = gabor_model.frequency.data[best_idxs]

        return x0, y0, sigmax, sigmay, theta, phi, frequency


class SpatialFitter:
    def __init__(
        self,
        rfs,
        freeze,
        n_iterations,
        lr,
        device,
        rf_size,
        x0_init,
        y0_init,
        sigmax_init,
        sigmay_init,
        theta_init,
        phi_init,
        frequency_init,
    ):
        self._rfs = rfs
        self._n_iterations = n_iterations

        self._gabor_model = Gabor(
            freeze,
            rf_size,
            x0_init,
            y0_init,
            sigmax_init,
            sigmay_init,
            theta_init,
            phi_init,
            frequency_init,
        ).to(device)
        self._optimizer = torch.optim.Adam(self._gabor_model.parameters(), lr)

    @property
    def gabor_model(self):
        return self._gabor_model

    def prediction(self):
        return self._gabor_model()

    def fit(self):
        for _ in range(self._n_iterations):
            self._optimizer.zero_grad()
            loss = F.mse_loss(self._rfs, self.prediction())
            loss.backward()
            self._optimizer.step()


class SpectralFitter(SpatialFitter):
    def __init__(
        self,
        rfs,
        freeze,
        n_iterations,
        lr,
        device,
        rf_size,
        x0_init,
        y0_init,
        sigmax_init,
        sigmay_init,
        theta_init,
        phi_init,
        frequency_init,
    ):
        super().__init__(
            rfs,
            freeze,
            n_iterations,
            lr,
            device,
            rf_size,
            x0_init,
            y0_init,
            sigmax_init,
            sigmay_init,
            theta_init,
            phi_init,
            frequency_init,
        )
        self._rfs = SpectralFitter._to_spectral(rfs)

    def prediction(self):
        return SpectralFitter._to_spectral(self._gabor_model())

    @staticmethod
    def _to_spectral(rfs, s=[100, 100]):
        return rfs_util.normalize_rfs(fft.fftn(rfs, s=s, dim=(1, 2)).abs())


class Gabor(nn.Module):
    def __init__(
        self,
        freeze=False,
        rf_size=20,
        x0_init=None,
        y0_init=None,
        sigmax_init=None,
        sigmay_init=None,
        theta_init=None,
        phi_init=None,
        frequency_init=None,
        eps=1e-3,
    ):
        super().__init__()
        self._amplitude = nn.Parameter(torch.ones(len(x0_init)), requires_grad=True)
        self._x0 = nn.Parameter(x0_init, requires_grad=not freeze)
        self._y0 = nn.Parameter(y0_init, requires_grad=not freeze)
        self._sigmax = nn.Parameter(sigmax_init, requires_grad=not freeze)
        self._sigmay = nn.Parameter(sigmay_init, requires_grad=not freeze)
        self._theta = nn.Parameter(theta_init, requires_grad=not freeze)
        self._phi = nn.Parameter(phi_init, requires_grad=not freeze)
        self._frequency = nn.Parameter(frequency_init, requires_grad=not freeze)
        self._eps = eps

        y, x = torch.meshgrid(
            [
                torch.arange(rf_size, dtype=torch.float32),
                torch.arange(rf_size, dtype=torch.float32),
            ],
            indexing="ij",
        )
        self.y = nn.Parameter(y, requires_grad=False)
        self.x = nn.Parameter(x, requires_grad=False)

        self._rf_ghost = nn.Parameter(torch.ones(rf_size, rf_size), requires_grad=False)

    @property
    def amplitude(self):
        return torch.clamp(self._amplitude, min=-1, max=1)

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
    def theta(self):
        return torch.clamp(self._theta, min=0, max=np.pi)

    @property
    def phi(self):
        return torch.clamp(self._phi, min=0, max=np.pi)

    @property
    def frequency(self):
        return torch.clamp(self._frequency, min=0, max=0.5)

    def forward(self):
        amplitude = self._expand_shape(self.amplitude)
        x0 = self._expand_shape(self.x0)
        y0 = self._expand_shape(self.y0)
        theta = self._expand_shape(self.theta)
        sigmax = self._expand_shape(self.sigmax)
        sigmay = self._expand_shape(self.sigmay)
        phi = self._expand_shape(self.phi)
        frequency = self._expand_shape(self.frequency)

        rotx = (self.x - x0) * torch.cos(theta) + (self.y - y0) * torch.sin(theta)
        roty = -(self.x - x0) * torch.sin(theta) + (self.y - y0) * torch.cos(theta)
        gaussian = torch.exp(-0.5 * ((rotx / sigmax) ** 2 + (roty / sigmay) ** 2))
        wave = torch.cos(2 * np.pi * frequency * rotx + phi)

        return amplitude * gaussian * wave

    def _expand_shape(self, vec):
        return torch.einsum("n, ij -> nij", vec, self._rf_ghost)
