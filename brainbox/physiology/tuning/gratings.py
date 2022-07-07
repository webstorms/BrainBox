import math
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np


class GratingsProber:

    GRATING_THETAS = np.linspace(0, np.pi*2, 10)# np.linspace(0, np.pi*2, 32)
    GRATINGS_SPATIAL_FREQS = np.linspace(0.05, 0.2, 10)# np.linspace(0, 0.4, 20)
    GRATINGS_TEMPORAL_FREQS = np.linspace(1, 4, 10)# np.linspace(1, 2)

    SINFT_AMPLITUDES_INIT = [0, 1, 2, 6]
    SINFIT_OFFSETS_INIT = [0, 1, 4]
    SINFIT_N_ITERATIONS = 500
    SINFIT_LOSS_THRESH = 1e-2

    GAUSSFIT_N_ITERATIONS = 500
    GAUSSFIT_LOSS_THRESH = 1e-2

    def __init__(self, model, amplitude, rf_w, rf_h, n_timesteps, dt, thetas=None, spatial_freqs=None, temporal_freqs=None,
                 amplitudes_init=None, offsets_init=None, n_iterations=None, loss_threshold=None):
        self.model = model
        self.amplitude = amplitude
        self.rf_w = rf_w
        self.rf_h = rf_h
        self.n_timesteps = n_timesteps
        self.dt = dt
        self.thetas = np.array(GratingsProber.GRATING_THETAS if thetas is None else thetas)
        self.spatial_freqs = np.array(GratingsProber.GRATINGS_SPATIAL_FREQS if spatial_freqs is None else spatial_freqs)
        self.temporal_freqs = np.array(GratingsProber.GRATINGS_TEMPORAL_FREQS if temporal_freqs is None else temporal_freqs)

        self.amplitudes_init = GratingsProber.SINFT_AMPLITUDES_INIT if amplitudes_init is None else amplitudes_init
        self.offsets_init = GratingsProber.SINFIT_OFFSETS_INIT if offsets_init is None else offsets_init
        self.n_iterations = GratingsProber.SINFIT_N_ITERATIONS if n_iterations is None else n_iterations
        self.loss_threshold = GratingsProber.SINFIT_LOSS_THRESH if loss_threshold is None else loss_threshold

        self.grating_results = None

    def check_model_has_been_probed(func):

        def wrapper(*args, **kwargs):
            assert args[0].grating_results is not None, "The model needs to be probed before calling this function."
            return func(*args, **kwargs)

        return wrapper

    @property
    @check_model_has_been_probed
    def n_units(self):
        return len(self.grating_results.columns) - 3

    def probe(self, batch_size=1024):

        def populate_mean_model_responses(batch_gratings, model_output_list):
            model_outputs = self.model(torch.stack(batch_gratings))
            model_outputs = model_outputs.mean(dim=2)
            n_units = model_outputs.shape[1]

            # Initialize model output list if not already done
            # We initialize after obtaining model responses as the number
            # of model units are unknown before passing grating
            if model_output_list is None:
                model_output_list = {str(i): [] for i in range(n_units)}

            # Store mean response of every unit for every sample
            for i in range(n_units):
                model_output_list[str(i)].extend([model_outputs[b, i].item() for b in range(len(batch_gratings))])

            return model_output_list

        batch_gratings = []
        thetas_list = []
        spatial_freq_list = []
        temporal_freq_list = []
        model_output_list = None

        for theta in self.thetas:
            for spatial_freq in self.spatial_freqs:
                for temporal_freq in self.temporal_freqs:
                    gratings = self._generate_grating(self.amplitude, self.rf_w, self.rf_h, theta, spatial_freq,
                                                      temporal_freq, self.n_timesteps, self.dt)
                    batch_gratings.append(gratings)

                    if len(batch_gratings) == batch_size:
                        model_output_list = populate_mean_model_responses(batch_gratings, model_output_list)

                        batch_gratings = []

                    thetas_list.append(theta)
                    spatial_freq_list.append(spatial_freq)
                    temporal_freq_list.append(temporal_freq)

        if len(batch_gratings) > 0:
            model_output_list = populate_mean_model_responses(batch_gratings, model_output_list)

        self.grating_results = pd.DataFrame({'theta': thetas_list,
                                             'spatial_frequency': spatial_freq_list,
                                             'temporal_frequency': temporal_freq_list,
                                             **model_output_list})

    @check_model_has_been_probed
    def preferred_direction(self, unit_id):
        max_output_index = self.grating_results[str(unit_id)].argmax()

        return self.grating_results.iloc[max_output_index]["theta"]

    @check_model_has_been_probed
    def preferred_spatial_frequency(self, unit_id):
        max_output_index = self.grating_results[str(unit_id)].argmax()

        return self.grating_results.iloc[max_output_index]["spatial_frequency"]

    @check_model_has_been_probed
    def preferred_temporal_frequency(self, unit_id):
        max_output_index = self.grating_results[str(unit_id)].argmax()

        return self.grating_results.iloc[max_output_index]['temporal_frequency']

    @check_model_has_been_probed
    def preferred_model_output(self, unit_id):
        return self.grating_results[str(unit_id)].max()

    @check_model_has_been_probed
    def orientation_tuning_curve(self, unit_id):
        """
        Orientation tuning curve (response as a function of grating orientation) for the preferred
        spatial frequency and temporal frequency
        Returns
        -------
        theta : ndarray
            Tuning curve orientations in radians
        tuning_curve : ndarray
            Model response at each orientation in `theta`
        """

        preferred_spatial_and_temporal_frequency_query = (self.grating_results["spatial_frequency"] == self.preferred_spatial_frequency(unit_id)) & (self.grating_results["temporal_frequency"] == self.preferred_temporal_frequency(unit_id))
        response_per_theta = self.grating_results[preferred_spatial_and_temporal_frequency_query]
        response_per_theta = response_per_theta.sort_values(by=["theta"])
        theta = response_per_theta[["theta"]].to_numpy().flatten()
        response = response_per_theta[[str(unit_id)]].to_numpy().flatten()

        return theta, response

    @check_model_has_been_probed
    def orientation_selectivity_index(self, unit_id):
        """
        Orientation selectivity index, computed as
        $OSI=\frac{{R}_{pref}^{Or}-{R}_{orth}^{Or}}{{R}_{pref}^{Or}+{R}_{orth}^{Or}}$ (Marques, Nguyen, Fioreze & Petreanu, 2018)
        Returns
        -------
        osi : numpy.float64
            Orientation selectivity index
        """

        # Absolute orthogonal to preferred
        orthogonal_orientation = (self.preferred_direction(unit_id) + np.pi/2) % (2 * np.pi)
        # Nearest angle to that actually probed
        nearest_orthogonal_orientation_idx = np.abs(self.thetas - orthogonal_orientation).argmin()
        orthogonal_orientation = self.thetas[nearest_orthogonal_orientation_idx]

        orthogonal_orientation_response_query = (self.grating_results["theta"] == orthogonal_orientation) & (self.grating_results["temporal_frequency"] == self.preferred_temporal_frequency(unit_id)) & (self.grating_results["spatial_frequency"] == self.preferred_spatial_frequency(unit_id))
        orthogonal_orientation_response = self.grating_results[orthogonal_orientation_response_query][str(unit_id)].values[0]

        return (self.preferred_model_output(unit_id) - orthogonal_orientation_response) / (self.preferred_model_output(unit_id) + orthogonal_orientation_response)

    # @property
    # def direction_selectivity_index(self):
    #     """
    #     Direction selectivity index, computed as
    #     $DSI=\frac{{R}_{pref}^{Dir}-{R}_{opp}^{Dir}}{{R}_{pref}^{Dir}+{R}_{opp}^{Dir}}$ (Marques, Nguyen, Fioreze & Petreanu, 2018)
    #     Returns
    #     -------
    #     dsi : numpy.float64
    #         Direction selectivity index
    #     """
    #
    #     assert self.grating_results is not None, 'The model needs to be probed before calling this function.'
    #
    #     # Absolute opposite to preferred
    #     opposite_direction = (self.preferred_direction + np.pi) % 2*np.pi
    #     # Nearest angle to that actually probed
    #     nearest_opposite_orientation_idx = np.abs(self.thetas - opposite_direction).argmin()
    #     nearest_opposite_orientation = self.thetas[nearest_opposite_orientation_idx]
    #
    #     opposite_direction_response = self.grating_results[
    #         (self.grating_results["theta"] == nearest_opposite_orientation) &
    #         (self.grating_results["temporal_frequency"] == self.preferred_temporal_frequency) &
    #         (self.grating_results["spatial_frequency"] == self.preferred_spatial_frequency)
    #         ]["model_output"].values[0]
    #
    #     return ((self.preferred_model_output - opposite_direction_response) /
    #             (self.preferred_model_output + opposite_direction_response))
    #
    # @property
    # def circular_variance(self):
    #     """
    #     Circular variance, computed as
    #     $CV= 1 - \frac{\sqrt{ (\sum_{q}^{Q}r_q\sin{2\theta_q})^2 + (\sum_{q}^{Q}r_q\cos{2\theta_q})^2}}{\sum_{q}^{Q}r_q}$ (Singer, Willmore, King & Harper, 2019)
    #     Returns
    #     -------
    #     cv : numpy.float64
    #         Circular variance
    #     """
    #
    #     assert self.grating_results is not None, 'The model needs to be probed before calling this function.'
    #
    #     theta, tuning_curve = self.orientation_tuning_curve
    #
    #     sum_sin = 0
    #     sum_cos = 0
    #     sum_r = 0
    #
    #     for i in range(len(tuning_curve)):
    #         r = tuning_curve[i]
    #         th = theta[i]
    #
    #         sum_sin += r*math.sin(2*th)
    #         sum_cos += r*math.cos(2*th)
    #         sum_r += r
    #
    #     return 1 - math.sqrt(sum_sin**2 + sum_cos**2)/sum_r
    #
    # @property
    # def orientation_bandwidth(self):
    #     """
    #     Orientation bandwith, described using the halfwidth at the halfmaximum of
    #     the orientation tuning curve's fitted Gaussian.
    #     $BW=\sqrt{2\ln{2}\sigma}$ (Jeon, Swain, Good Chase & Kuhlman, 2018)
    #     Returns
    #     -------
    #     bw : numpy.float64
    #         Orientation bandwidth
    #     """
    #
    #     assert self.grating_results is not None, 'The model needs to be probed before calling this function.'
    #
    #     theta, tuning_curve = self.orientation_tuning_curve
    #
    #     # Average over 0-180 and 180-360 degrees
    #     orient_n = len(theta)//2
    #     orient_curve = (tuning_curve[:orient_n] + tuning_curve[:orient_n])/2
    #     orient_theta = theta[:orient_n]
    #
    #     # Determine shift to bring peak into 'centre' for fitting
    #     # and rotate tuning curve by that shift
    #     shift = math.ceil(len(orient_curve)/2) - (np.argmax(orient_curve)+1)
    #     orient_curve_deque = collections.deque(orient_curve)
    #     orient_curve_deque.rotate(shift)
    #     orient_curve_rot = np.array(orient_curve_deque)
    #
    #     loss, amplitude, mean, variance = GaussianFitter.fit_gaussian(
    #         xdata=orient_theta,
    #         ydata=orient_curve_rot,
    #         amplitude_init=max(orient_curve_rot),
    #         mean_init=orient_theta[np.argmax(orient_curve_rot)],
    #         variance_init=np.var(orient_curve_rot),
    #         n_iterations=self.GAUSSFIT_N_ITERATIONS
    #     )
    #
    #     assert loss < self.GAUSSFIT_LOSS_THRESH, \
    #         'Could not determine orientation bandwidth. Gaussian fit min loss={0} did not meet threshold criteria {1}.'.format(loss, self.GAUSSFIT_LOSS_THRESH)
    #
    #     return math.sqrt(2*math.log(2)) * math.sqrt(variance)

    @check_model_has_been_probed
    def modulation_ratio(self, unit_id):
        model_output = self.get_response_to_preferred_grating(unit_id, self.n_timesteps).cpu()
        phases_init = [0]
        frequency_init = [(2 * np.pi) * self.preferred_temporal_frequency / (1000 // self.dt)]
        F1 = self.get_sinusoid_fit_to_response(model_output.flatten().detach(), self.amplitudes_init, phases_init, self.offsets_init, frequency_init, self.n_iterations, self.loss_threshold)[0]
        F0 = model_output.mean().item()

        return F1 / F0

    def get_sinusoid_fit_to_response(self, model_output, amplitudes_init, phases_init, offsets_init, frequencies_init, n_iterations, loss_threshold):
        sinusoid_fitter = SinusoidFitter(model_output, amplitudes_init, phases_init, offsets_init, frequencies_init, n_iterations, loss_threshold)
        sinusoid_fitter.fit()

        return sinusoid_fitter.best_fit

    def get_response_to_preferred_grating(self, n_timesteps, unit_id=None, batch_size=10):
        gratings = self.generate_preferred_grating(n_timesteps, unit_id)
        
        if unit_id is not None:
            return self.model(gratings.unsqueeze(0))[0, int(unit_id)]
        else:
            batch_response_to_preferred_grating = []

            for i in range(self.n_units // batch_size + self.n_units % batch_size):
                batch_gratings = gratings[i*batch_size: (i+1)*batch_size]
                batch_response_to_preferred_grating.append(self.model(batch_gratings))

            response_to_preferred_grating = torch.cat(batch_response_to_preferred_grating)
            response_to_preferred_grating = torch.stack([response_to_preferred_grating[i, i] for i in range(self.n_units)])

            return response_to_preferred_grating

    def generate_preferred_grating(self, n_timesteps, unit_id=None):
        if unit_id is not None:
            return self._generate_grating(self.amplitude, self.rf_w, self.rf_h, self.preferred_direction(unit_id),
                                          self.preferred_spatial_frequency(unit_id), self.preferred_temporal_frequency(unit_id), n_timesteps, self.dt)
        else:
            return torch.stack([self.generate_preferred_grating(n_timesteps, str(idx)) for idx in range(self.n_units)])

    def _generate_grating(self, amplitude, rf_w, rf_h, theta, spatial_freq, temporal_freq, n_timesteps, dt):
        n_frames = 1000 / dt
        y, x = torch.meshgrid([
            torch.arange(rf_h, dtype=torch.float32),
            torch.arange(rf_w, dtype=torch.float32)
        ])
        theta = torch.Tensor([theta]).expand_as(y)
        spatial_freq = torch.Tensor([spatial_freq]).expand_as(y)
        time = torch.arange(n_timesteps).view(n_timesteps, 1, 1).repeat(1, rf_h, rf_w)

        rotx = x * torch.cos(theta) - y * torch.sin(theta)
        grating = amplitude * torch.sin(2 * np.pi * (spatial_freq * rotx - temporal_freq * time / n_frames))

        return grating
