import os

import numpy as np
import pandas as pd
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from kornia import geometry

from brainbox.physiology.rfs import *


class GaborFitter:

    GABOR_SIGMAS = [0.5, 1.5]
    GABOR_THETAS = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
    GABOR_PHIS = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    GABOR_FREQUENCIES = [0.05, 0.1]

    def __init__(self, root, name, rfs, spatial_locations=None, sigmas=None, thetas=None, phis=None, frequencies=None, n_spectral_itr=800, n_spatial_itr=1200, spectral_lr=0.01, spatial_lr=0.002):
        self.root = root
        self.name = name
        self.rfs = rfs.cpu()
        rf_size = rfs[0].shape[0]

        self.spatial_locations = [(rf_size // 2, rf_size // 2)] if spatial_locations is None else spatial_locations
        self.sigmas = GaborFitter.GABOR_SIGMAS if sigmas is None else sigmas
        self.thetas = GaborFitter.GABOR_THETAS if thetas is None else thetas
        self.phis = GaborFitter.GABOR_PHIS if phis is None else phis
        self.frequencies = GaborFitter.GABOR_FREQUENCIES if frequencies is None else frequencies
        self.n_spectral_itr = n_spectral_itr
        self.n_spatial_itr = n_spatial_itr
        self.spectral_lr = spectral_lr
        self.spatial_lr = spatial_lr

    @staticmethod
    def get_fit(i, rf, spatial_locations, sigmas, thetas, phis, frequencies, n_spectral_itr, n_spatial_itr, spectral_lr, spatial_lr):
        torch.set_num_threads(1)
        print('Started fit {0}...'.format(i))
        single_gabor_fitter = SingleGaborFitter(rf, spatial_locations, sigmas, thetas, phis, frequencies,
                                                n_spectral_itr, n_spatial_itr, spectral_lr, spatial_lr)
        single_gabor_fitter.fit()
        print('Completed fit {0}...'.format(i))

        return single_gabor_fitter.get_best_fit()

    @property
    def model_path_name(self):
        return os.path.join(self.root, self.name)

    def fit(self):
        pool = mp.Pool()
        results = [pool.apply_async(GaborFitter.get_fit, args=(i, self.rfs[i], self.spatial_locations, self.sigmas,
                                                               self.thetas, self.phis, self.frequencies,
                                                               self.n_spectral_itr,
                                                               self.n_spatial_itr, self.spectral_lr, self.spatial_lr,)) for i in
                   range(self.rfs.shape[0])]
        params = [p.get() for p in results]

        params = pd.DataFrame(params, columns=['x0', 'y0', 'sigmax', 'sigmay', 'theta', 'phi', 'frequency'])
        params.to_csv(self.model_path_name)


class SingleGaborFitter(nn.Module):

    def __init__(self, rf, spatial_locations, sigmas, thetas, phis, frequencies, n_spectral_itr, n_spatial_itr, spectral_lr, spatial_lr):
        super().__init__()
        assert rf.shape[0] == rf.shape[1], 'Please provide square RF shape'
        self.n_spectral_itr = n_spectral_itr
        self.n_spatial_itr = n_spatial_itr
        self.spectral_lr = spectral_lr
        self.spatial_lr = spatial_lr

        self.rf_size = rf.shape[0]
        self.x0_inits, self.y0_inits, self.sigmax_inits, self.sigmay_inits, self.theta_inits, self.phi_inits, self.frequency_inits \
            = SingleGaborFitter.build_initial_params(spatial_locations, sigmas, thetas, phis, frequencies)
        self.rf = rf.cpu().unsqueeze(0).repeat(self.n_units, 1, 1)

        self.spectral_gabor_model = None
        self.spatial_gabor_model = None

    @property
    def n_units(self):
        return len(self.x0_inits)

    @staticmethod
    def build_initial_params(spatial_locations, sigmas, thetas, phis, frequencies):
        x0_inits = []
        y0_inits = []
        sigmax_inits = []
        sigmay_inits = []
        theta_inits = []
        phi_inits = []
        frequency_inits = []

        for location in spatial_locations:
            x0 = location[0]
            y0 = location[1]
            for sigx in sigmas:
                for sigy in sigmas:
                    for theta in thetas:
                        for phi in phis:
                            for frequency in frequencies:
                                x0_inits.append(x0)
                                y0_inits.append(y0)
                                sigmax_inits.append(sigx)
                                sigmay_inits.append(sigy)
                                theta_inits.append(theta)
                                phi_inits.append(phi)
                                frequency_inits.append(frequency)

        return x0_inits, y0_inits, sigmax_inits, sigmay_inits, theta_inits, phi_inits, frequency_inits

    @staticmethod
    def to_spectral(rf, s=[100, 100]):
        return fft.fftn(rf, s=s).abs()

    @staticmethod
    def to_normalised(rf):
        return rf / torch.clamp(rf.abs().amax((1, 2)).view(-1, 1, 1), min=1e-5, max=np.inf)

    def get_best_fit(self):
        predicted_rfs = self.spatial_gabor_model()
        best_index = 0
        best_loss = np.inf

        for i in range(self.n_units):
            pred_spectral_rf = SingleGaborFitter.to_normalised(predicted_rfs[i].unsqueeze(0))
            target_spectral_rf = SingleGaborFitter.to_normalised(self.rf[0].unsqueeze(0))

            loss = F.mse_loss(pred_spectral_rf, target_spectral_rf)
            if loss < best_loss:
                best_loss = loss
                best_index = i

        x0 = self.spatial_gabor_model.x0[best_index].item()
        y0 = self.spatial_gabor_model.y0[best_index].item()
        sigmax = self.spatial_gabor_model.sigmax[best_index].item()
        sigmay = self.spatial_gabor_model.sigmay[best_index].item()
        theta = self.spatial_gabor_model.theta[best_index].item()
        phi = self.spatial_gabor_model.phi[best_index].item()
        frequency = self.spatial_gabor_model.frequency[best_index].item()

        return x0, y0, sigmax, sigmay, theta, phi, frequency

    def fit(self):
        self.fit_spectral()
        self.fit_spatial()

    def fit_spectral(self):
        self.spectral_gabor_model = Gabor(self.n_units, self.rf_size, self.x0_inits, self.y0_inits,
                                                self.sigmax_inits, self.sigmay_inits, self.theta_inits, self.phi_inits,
                                                self.frequency_inits)
        self._fit_gabor_model(self.spectral_gabor_model, self.n_spectral_itr, self.spectral_lr, True)

    def fit_spatial(self):
        x0_inits = self.spectral_gabor_model.x0
        y0_inits = self.spectral_gabor_model.y0
        sigx_inits = self.spectral_gabor_model.sigmax
        sigy_inits = self.spectral_gabor_model.sigmay
        theta_inits = self.spectral_gabor_model.theta
        phi_inits = self.spectral_gabor_model.phi
        frequency_inits = self.spectral_gabor_model.frequency

        self.spatial_gabor_model = Gabor(self.n_units, self.rf_size, x0_inits, y0_inits, sigx_inits, sigy_inits,
                                               theta_inits, phi_inits, frequency_inits)
        self._fit_gabor_model(self.spatial_gabor_model, self.n_spatial_itr, self.spatial_lr, False)

    def _fit_gabor_model(self, gabor_model, n_iterations, lr, spectral):
        target = SingleGaborFitter.to_normalised(self.rf)
        target = SingleGaborFitter.to_normalised(SingleGaborFitter.to_spectral(target)) if spectral else target

        for iteration in range(n_iterations):
            prediction = gabor_model()
            prediction = SingleGaborFitter.to_normalised(prediction)
            prediction = SingleGaborFitter.to_normalised(SingleGaborFitter.to_spectral(prediction)) if spectral else prediction

            optimizer = torch.optim.Adam(gabor_model.parameters(), lr)

            optimizer.zero_grad()
            loss = F.mse_loss(prediction, target)
            if iteration % 100 == 0 and iteration > 0:
                print('it={0} loss={1}'.format(iteration, loss))
            loss.backward()
            optimizer.step()


class Gabor(nn.Module):

    def __init__(self, n_units, rf_size=20, x0_init=None, y0_init=None, sigmax_init=None,
                 sigmay_init=None, theta_init=None, phi_init=None, frequency_init=None, eps=1e-3):
        super().__init__()
        self.n_units = n_units
        x0_init = torch.Tensor(x0_init)
        y0_init = torch.Tensor(y0_init)
        sigmax_init = torch.Tensor(sigmax_init)
        sigmay_init = torch.Tensor(sigmay_init)
        theta_init = torch.Tensor(theta_init)
        phi_init = torch.Tensor(phi_init)
        frequency_init = torch.Tensor(frequency_init)
        self.eps = eps

        self._x0 = nn.Parameter(x0_init, requires_grad=True)
        self._y0 = nn.Parameter(y0_init, requires_grad=True)
        self._sigmax = nn.Parameter(sigmax_init, requires_grad=True)
        self._sigmay = nn.Parameter(sigmay_init, requires_grad=True)
        self._theta = nn.Parameter(theta_init, requires_grad=True)
        self._phi = nn.Parameter(phi_init, requires_grad=True)
        self._frequency = nn.Parameter(frequency_init, requires_grad=True)

        self.y, self.x = torch.meshgrid([
            torch.arange(rf_size, dtype=torch.float32),
            torch.arange(rf_size, dtype=torch.float32)
        ])
        self.y = nn.Parameter(self.y, requires_grad=False)
        self.x = nn.Parameter(self.x, requires_grad=False)

    @property
    def x0(self):
        return self._x0

    @property
    def y0(self):
        return self._y0

    @property
    def sigmax(self):
        return torch.clamp(self._sigmax, min=self.eps, max=np.inf)

    @property
    def sigmay(self):
        return torch.clamp(self._sigmay, min=self.eps, max=np.inf)

    @property
    def theta(self):
        return torch.clamp(self._theta, min=0, max=np.pi)

    @property
    def phi(self):
        return torch.clamp(self._phi, min=0, max=2*np.pi)

    @property
    def frequency(self):
        return torch.clamp(self._frequency, min=0, max=0.5)

    def forward(self):
        gabors = []

        for i in range(self.n_units):
            sigmax = self.sigmax[i].expand_as(self.y)
            sigmay = self.sigmay[i].expand_as(self.y)
            theta = self.theta[i].expand_as(self.y)
            phi = self.phi[i].expand_as(self.y)
            freq = self.frequency[i].expand_as(self.y)

            rotx = (self.x - self.x0[i]) * torch.cos(theta) + (self.y - self.y0[i]) * torch.sin(theta)
            roty = -(self.x - self.x0[i]) * torch.sin(theta) + (self.y - self.y0[i]) * torch.cos(theta)

            g = torch.exp(-0.5 * ((rotx / sigmax) ** 2 + (roty / sigmay) ** 2))
            g = g * torch.cos(2 * np.pi * freq * rotx + phi)
            gabors.append(g)

        return torch.stack(gabors, dim=0)


class GaborValidator:

    SPACE_TIME_SEPARABLE = 'separable'
    SPACE_TIME_INSEPARABLE = 'inseparable'

    def __init__(self, params_df, spatiotemporal_rfs, min_cc=0.7, min_env=0.5, separability=None, inseperable_thresh=0.5):
        self.params_df = params_df
        self.spatiotemporal_rfs = spatiotemporal_rfs
        self.min_cc = min_cc
        self.min_env = min_env
        self.separability = separability
        self.inseperable_thresh = inseperable_thresh

        self.spatial_rfs = get_all_highest_power_spatial_rf(spatiotemporal_rfs)
        self.rf_size = spatiotemporal_rfs.shape[2]
        self.gabors = self.build_gabors()
        self.params_df['cc'] = self.get_ccs()

    def __str__(self):
        total_units = len(self.spatial_rfs)
        n_units_included = len(self.validate())
        n_units_excluded = total_units - n_units_included

        output = 'cc query: excluding {0} units\n'.format((self._query_cc() == False).sum())
        output += 'location query: excluding {0} units\n'.format((self._query_location() == False).sum())
        output += 'envolope query: excluding {0} units\n'.format((self._query_envelope() == False).sum())
        output += 'seperability query={0}: excluding {1} units\n'.format(self.separability, (self._query_separability() == False).sum())
        output += '--> Total units after validation {0} (excluded {1}/{2})\n'.format(n_units_included, n_units_excluded, total_units)

        return output

    def _query_cc(self):
        return self.params_df['cc'] > self.min_cc

    def _query_location(self):
        query = (0 < self.params_df['x0']) & (self.params_df['x0'] < self.rf_size)
        query &= (0 < self.params_df['y0']) & (self.params_df['y0'] < self.rf_size)

        return query

    def _query_envelope(self):
        query = self.params_df['sigmax'] > self.min_env
        query &= self.params_df['sigmay'] > self.min_env

        return query

    def _query_separability(self):
        spatiotemporal_rfs = self.spatiotemporal_rfs.permute(0, 2, 3, 1)
        n_units = spatiotemporal_rfs.shape[0]
        n_timesteps = spatiotemporal_rfs.shape[3]
        spatiotemporal_rfs = spatiotemporal_rfs.view(n_units, -1, n_timesteps)

        u, s, v = torch.svd(spatiotemporal_rfs)

        inseperable_units = (s[:, 1] / s[:, 0] > self.inseperable_thresh).numpy()
        query = inseperable_units

        if self.separability == GaborValidator.SPACE_TIME_SEPARABLE:
            query = inseperable_units == False

        return query

    def validate(self):
        query = self._query_cc()
        query &= self._query_location()
        query &= self._query_envelope()

        if self.separability is not None:
            query &= self._query_separability()

        return self.params_df[query]

    def get_ccs(self):
        ccs = []

        for i in range(self.gabors.shape[0]):
            gabor = self.gabors[i]
            rf = self.spatial_rfs[i]
            cc = GaborValidator.get_cc(gabor, rf)
            ccs.append(cc)

        return ccs

    def build_gabors(self):
        x0_init = self.params_df['x0'].values
        y0_init = self.params_df['y0'].values
        sigmax_init = self.params_df['sigmax'].values
        sigmay_init = self.params_df['sigmay'].values
        theta_init = self.params_df['theta'].values
        phi_init = self.params_df['phi'].values
        frequency_init = self.params_df['frequency'].values
        gabor_model = Gabor(len(x0_init), self.rf_size, x0_init, y0_init, sigmax_init,
                                  sigmay_init, theta_init, phi_init, frequency_init)

        return gabor_model()

    @staticmethod
    def get_cc(gabor, rf):
        return np.corrcoef(gabor.flatten().detach().numpy(), rf.flatten().detach().numpy())[0, 1]


def fit_gabors(root, name, spatiotemporal_rfs, spatial_locations=None, sigmas=None, thetas=None, phis=None, frequencies=None, n_spectral_itr=800, n_spatial_itr=1200, spectral_lr=0.01, spatial_lr=0.002):

    """
    Fit gabor parameters to model receptive fields. Various initial conditions are used for fitting each gabor, for which the best fit (for every rf)
    is eventually saved as a pandas dataframe. Initial condition values (spatial_locations, sigmas, thetas, phis, frequencies) can be set to None for which
    default parameters will be used. Fitting is performed as outlined in Singer et al., 2018 (https://elifesciences.org/articles/31557#bib73):
    Step 1. Gabors are fitted in the spectral domain as to obtain gabor parameter estimates that avoid local minima. Step 2. Parameter estimates
    are then used to fit gabors in the spatial domain.

    :param root: Root directory containing the fitted gabors
    :param name: The name of the pandas dataframe containing the fitted gabors parameters
    :param spatiotemporal_rfs: A tensor (n_neurons x rf_len x width x height) of rfs to which the gabors are fitted
    :param spatial_locations: A list of spatial locations (defining the gabor centre) initial conditions
    :param sigmas:
    :param thetas: A list of spatial orientation initial conditions
    :param phis: A list of phase initial conditions
    :param frequencies: A list of spatial frequency initial conditions
    :param n_spectral_itr: Number of iterations to use for fitting in the spectral domain
    :param n_spatial_itr: Number of iterations to use for fitting in the spatial domain
    :param spectral_lr: Learning rate for optimisation in spectral domain
    :param spatial_lr: Learing rate for optimisation in spatial domain
    """

    rfs = get_all_highest_power_spatial_rf(spatiotemporal_rfs)

    gabor_fitter = GaborFitter(root, name, rfs, spatial_locations, sigmas, thetas, phis, frequencies, n_spectral_itr, n_spatial_itr, spectral_lr, spatial_lr)
    gabor_fitter.fit()


def get_rf_shape_distribution(root, name, spatiotemporal_rfs, min_cc=0.7, min_env=0.5, separability=None, inseperable_thresh=0.5):

    """
    Compute the distribution of receptive field shapes as defined in Ringach, 2002 (https://journals.physiology.org/doi/full/10.1152/jn.2002.88.1.455)
    after applying filtering criterion. Every shape is represented by two values (nx, ny) which are a measure of RF span parallel and orthogonal to
    orientation tuning as a proportion to the spatial oscillation period. In other terms, ny gives a measure of the length of the bars in the RF and nx
    gives a measure of the number of oscillations of its sinusoidal component. Filtering criterion is applied to ensure
    sufficient fits as done in Singer et al., 2018 (https://elifesciences.org/articles/31557#bib73).

    :param root: Root directory containing the fitted gabors
    :param name: The name of the file containing the fitted gabors parameters
    :param spatiotemporal_rfs: A tensor (n_neurons x rf_len x width x height) of spatiotemporal rfs to which the gabors are fitted
    :param min_cc: Ignore poorly fitted gabors that obtain a correlation coefficient less than min_cc
    :param min_env: Ignore fitted gabors where estimated standard deviation of the Gaussian envelope in either x or y is less than min_env
    :param separability: Only include rfs meeting separability criterion ['separable', 'inseparable', None]
    :param inseparable_thresh: RFs are deemed inseparable if their 2nd/1st singular values are above inseparable_thresh (see Singer et al., 2018)
    :return: Two numpy arrays representing nxs and nys
    """

    params_df = pd.read_csv(os.path.join(root, name))
    gabor_validator = GaborValidator(params_df, spatiotemporal_rfs, min_cc, min_env, separability, inseperable_thresh)
    validated_params_df = gabor_validator.validate()

    nx = (validated_params_df['sigmax'] * validated_params_df['frequency']).values
    ny = (validated_params_df['sigmay'] * validated_params_df['frequency']).values

    return nx, ny


def get_gabors(root, name, spatiotemporal_rfs, min_cc=0.7, min_env=0.5, separability=None, inseperable_thresh=0.5):

    """
    Obtain a pandas dataframe of fitted gabor parameters; a tensor containing the gabor fits; a tensor containing the
    spatial RFs with largest power and a tensor containing the spatiotemporal RFs. Filtering criterion is applied to
    ensure sufficient fits as done in Singer et al., 2018 (https://elifesciences.org/articles/31557#bib73).

    :param root: Root directory containing the fitted gabors
    :param name: The name of the file containing the fitted gabors parameters
    :param spatiotemporal_rfs: A tensor (n_neurons x rf_len x width x height) of spatiotemporal rfs to which the gabors are fitted
    :param min_cc: Ignore poorly fitted gabors that obtain a correlation coefficient less than min_cc
    :param min_env: Ignore fitted gabors where estimated standard deviation of the Gaussian envelope in either x or y is less than min_env
    :param separability: Only include rfs meeting separability criterion ['separable', 'inseparable', None]
    :param inseparable_thresh: RFs are deemed inseparable if their 2nd/1st singular values are above inseparable_thresh (see Singer et al., 2018)
    :return: Four elements: Pandas dataframe, tensor, tensor, tensor
    """

    params_df = pd.read_csv(os.path.join(root, name))
    gabor_validator = GaborValidator(params_df, spatiotemporal_rfs, min_cc, min_env, separability, inseperable_thresh)
    validated_params_df = gabor_validator.validate()

    validated_index = validated_params_df.index.values
    
    return validated_params_df, gabor_validator.build_gabors()[validated_index].detach(), gabor_validator.spatial_rfs[validated_index].detach(), spatiotemporal_rfs[validated_index].detach()


def get_gabor_validator(root, name, spatiotemporal_rfs, min_cc=0.7, min_env=0.5, separability=None, inseperable_thresh=0.5):

    """
    Obtain a GaborValidator instance.

    :param root: Root directory containing the fitted gabors
    :param name: The name of the file containing the fitted gabors parameters
    :param spatiotemporal_rfs: A tensor (n_neurons x rf_len x width x height) of spatiotemporal rfs to which the gabors are fitted
    :param min_cc: Ignore poorly fitted gabors that obtain a correlation coefficient less than min_cc
    :param min_env: Ignore fitted gabors where estimated standard deviation of the Gaussian envelope in either x or y is less than min_env
    :param separability: Only include rfs meeting separability criterion ['separable', 'inseparable', None]
    :param inseparable_thresh: RFs are deemed inseparable if their 2nd/1st singular values are above inseparable_thresh (see Singer et al., 2018)
    :return: A GaborValidator instance
    """

    params_df = pd.read_csv(os.path.join(root, name))
    gabor_validator = GaborValidator(params_df, spatiotemporal_rfs, min_cc, min_env, separability, inseperable_thresh)

    return gabor_validator


def get_2D_spatiotemporal_rfs(params, spatiotemporal_rfs):

    """
    Obtain space-time RFs for a given set of spatiotemporal RFs as done in DeAngelis et al., 1993 (https://journals.physiology.org/doi/pdf/10.1152/jn.1993.69.4.1091).
    These RFs are useful to better view the temporal characteristics of spatiotemporal RFs.

    :param params: The pandas datafrane containing the gabor fit parameters
    :param spatiotemporal_rfs: The tensor (n_neurons x rf_len x width x height) of spatiotemporal RFs
    :return: A tensor (n_neurons x width x rf_len) of 2d spatiotemporal RFs
    """

    def translate(rf, dx, dy):
        return geometry.translate(rf.unsqueeze(0), torch.Tensor([[dx, dy]]))[0]

    def rotate(rf, rot_deg):
        return geometry.rotate(rf.unsqueeze(0), torch.Tensor([rot_deg]))[0]

    def get_2d_spatiotemporal_rf(spatiotemporal_rf, x0, y0, theta):
        # spatiotemporal_rf: rf_len x w x h
        n_timesteps = spatiotemporal_rf.shape[0]
        rf_size = spatiotemporal_rf.shape[1] // 2
        dx = rf_size - x0
        dy = rf_size - y0
        rot_deg = np.rad2deg(theta)
        transformed_spatiotemporal_rf = []

        for t in range(n_timesteps):
            transformed_rf = translate(spatiotemporal_rf[t], dx, dy)
            transformed_rf = rotate(transformed_rf, rot_deg)
            transformed_spatiotemporal_rf.append(transformed_rf)

        transformed_spatiotemporal_rf = torch.stack(transformed_spatiotemporal_rf)

        return transformed_spatiotemporal_rf.sum(dim=1).T

    assert len(params) == len(spatiotemporal_rfs), 'Provided params and spatiotemporal_rfs are of different lengths.'
    spatiotemporal_2d_rfs = []

    for i in range(len(params)):
        x0 = params.iloc[i].x0
        y0 = params.iloc[i].y0
        theta = params.iloc[i].theta

        spatiotemporal_2d_rf = get_2d_spatiotemporal_rf(spatiotemporal_rfs[i], x0, y0, theta)
        spatiotemporal_2d_rfs.append(spatiotemporal_2d_rf)

    return torch.stack(spatiotemporal_2d_rfs)

