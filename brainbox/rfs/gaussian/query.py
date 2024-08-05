import sys
import logging
import torch
import numpy as np
import pandas as pd

from kornia import geometry

from brainbox.rfs import rfs
from brainbox.rfs.gaussian.fit import Gaussian2D

logger = logging.getLogger("gaussian")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class GaussianQuery:

    SPACE_TIME_SEPARABLE = "separable"
    SPACE_TIME_INSEPARABLE = "inseparable"

    def __init__(self, fits_path, strfs):
        self._params_df = pd.read_csv(fits_path)
        self._strfs = strfs.cpu()

        self._rf_size = strfs.shape[-1]
        self._spatial_rfs = rfs.get_all_highest_power_spatial_rf(strfs)
        self._gaussians = self._build_gaussians()
        self._params_df["cc"] = self._get_correlation()

    def validate(
        self,
        min_cc,
        min_env=0.5,
        separability=None,
        inseperable_thresh=0.5,
        verbose=True,
    ):
        query = self._query_cc(min_cc)
        query &= self._query_location()
        query &= self._query_envelope(min_env)
        query &= self._query_separability(separability, inseperable_thresh)

        if verbose:
            logger.info(
                f"CC criteria exclusion {(self._query_cc(min_cc)==False).sum()}"
            )
            logger.info(
                f"Location criteria exclusion {(self._query_location()==False).sum()}"
            )
            logger.info(
                f"Envelope criteria exclusion {(self._query_envelope(min_env)==False).sum()}"
            )

        params_df = self._params_df[query]
        spatial_rfs = self._spatial_rfs[query]
        gaussians = self._gaussians[query]
        strfs = self._strfs[query]

        return params_df, spatial_rfs, gaussians, strfs

    def _query_cc(self, min_cc):
        return self._params_df["cc"] > min_cc

    def _query_location(self):
        query = (0 <= self._params_df["x0"]) & (self._params_df["x0"] < self._rf_size)
        query &= (0 <= self._params_df["y0"]) & (self._params_df["y0"] < self._rf_size)

        return query

    def _query_envelope(self, min_env):
        query = self._params_df["sigmax"] > min_env
        query &= self._params_df["sigmay"] > min_env

        return query

    def _query_separability(self, separability, inseperable_thresh):
        if separability is None:
            return True

        spatiotemporal_rfs = self._strfs.flatten(start_dim=2, end_dim=3)
        spatiotemporal_rfs = spatiotemporal_rfs.permute(0, 2, 1)
        u, s, v = torch.svd(spatiotemporal_rfs)
        inseperable_units = (s[:, 1] / s[:, 0] > inseperable_thresh).numpy()
        query = (
            inseperable_units
            if separability == GaussianQuery.SPACE_TIME_INSEPARABLE
            else inseperable_units == False
        )

        return query

    def _get_correlation(self):
        def correlation(gaussian, rf):
            return np.corrcoef(
                gaussian.flatten().detach().numpy(), rf.flatten().detach().numpy()
            )[0, 1]

        return [
            correlation(gaussian, rf)
            for gaussian, rf in zip(self._gaussians, self._spatial_rfs)
        ]

    def _build_gaussians(self):
        (amp, x0, y0, sigmax, sigmay, p) = GaussianQuery._df_params_to_gaussian_params(
            self._params_df
        )

        return Gaussian2D(
            rf_size=self._rf_size,
            amp_init=amp,
            x0_init=x0,
            y0_init=y0,
            sigmax_init=sigmax,
            sigmay_init=sigmay,
            p_init=p,
        )()

    @staticmethod
    def _df_params_to_gaussian_params(params_df):
        amp = torch.Tensor(params_df["amp"].to_numpy())
        x0 = torch.Tensor(params_df["x0"].to_numpy())
        y0 = torch.Tensor(params_df["y0"].to_numpy())
        sigmax = torch.Tensor(params_df["sigmax"].to_numpy())
        sigmay = torch.Tensor(params_df["sigmay"].to_numpy())
        p = torch.Tensor(params_df["p"].to_numpy())

        return amp, x0, y0, sigmax, sigmay, p
