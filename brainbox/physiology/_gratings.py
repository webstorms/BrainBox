import math, collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np


class GratingsProber:

    GRATING_THETAS = np.linspace(0, np.pi*2, 32)
    GRATINGS_SPATIAL_FREQS = np.linspace(0, 0.4, 20)
    GRATINGS_TEMPORAL_FREQS = np.linspace(1, 2)

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
        self.thetas = GratingsProber.GRATING_THETAS if thetas is None else thetas
        self.spatial_freqs = GratingsProber.GRATINGS_SPATIAL_FREQS if spatial_freqs is None else spatial_freqs
        self.temporal_freqs = GratingsProber.GRATINGS_TEMPORAL_FREQS if temporal_freqs is None else temporal_freqs
        self.amplitudes_init = GratingsProber.SINFT_AMPLITUDES_INIT if amplitudes_init is None else amplitudes_init
        self.offsets_init = GratingsProber.SINFIT_OFFSETS_INIT if offsets_init is None else offsets_init
        self.n_iterations = GratingsProber.SINFIT_N_ITERATIONS if n_iterations is None else n_iterations
        self.loss_threshold = GratingsProber.SINFIT_LOSS_THRESH if loss_threshold is None else loss_threshold

        self.grating_results = None

    def probe(self, batch_size):
        batch_gratings = []
        batch_counter = 0

        thetas_list = []
        spatial_freq_list = []
        temporal_freq_list = []
        model_output_list = []

        for theta in self.thetas:
            for spatial_freq in self.spatial_freqs:
                for temporal_freq in self.temporal_freqs:
                    gratings = self.generate_gratings(self.amplitude, self.rf_w, self.rf_h, theta, spatial_freq,
                                                      temporal_freq, self.n_timesteps, self.dt)
                    batch_gratings.append(gratings)
                    batch_counter += 1

                    if batch_counter == batch_size:
                        model_outputs = self.model(torch.stack(batch_gratings))
                        assert len(model_outputs.shape) == 2, 'Model output should be two-dimensional'
                        for i in range(len(model_outputs)):
                            model_output_list.append(model_outputs[i].mean().item())
                        batch_counter = 0
                        batch_gratings = []

                    thetas_list.append(theta)
                    spatial_freq_list.append(spatial_freq)
                    temporal_freq_list.append(temporal_freq)

        if len(batch_gratings) > 0:
            model_outputs = self.model(batch_gratings)
            for i in range(len(model_outputs)):
                model_output_list.append(model_outputs[i].mean().item())

        self.grating_results = pd.DataFrame({'theta': thetas_list,
                                             'spatial_frequency': spatial_freq_list,
                                             'temporal_frequency': temporal_freq_list,
                                             'model_output': model_output_list})

    def generate_gratings(self, amplitude, rf_w, rf_h, theta, spatial_freq, temporal_freq, n_timesteps, dt):
        n_frames = 1000 // dt
        y, x = torch.meshgrid([
            torch.arange(rf_h, dtype=torch.float32),
            torch.arange(rf_w, dtype=torch.float32)
        ])
        theta = torch.Tensor([theta]).expand_as(y)
        spatial_freq = torch.Tensor([spatial_freq]).expand_as(y)

        time = torch.arange(n_timesteps).view(n_timesteps, 1, 1).repeat(1, rf_h, rf_w)

        rotx = x * torch.cos(theta) - y * torch.sin(theta)
        gratings = amplitude * torch.sin(2 * np.pi * (spatial_freq * rotx - temporal_freq * time / n_frames))

        return gratings

    @property
    def preferred_direction(self):
        assert self.grating_results is not None, 'The model needs to be probed before calling this function.'
        max_output_index = self.grating_results['model_output'].argmax()

        return self.grating_results.iloc[max_output_index]['theta']

    @property
    def preferred_orientation(self):
        return self.preferred_direction % np.pi

    @property
    def preferred_spatial_freq(self):
        assert self.grating_results is not None, 'The model needs to be probed before calling this function.'
        max_output_index = self.grating_results['model_output'].argmax()

        return self.grating_results.iloc[max_output_index]['spatial_frequency']

    @property
    def preferred_temporal_freq(self):
        assert self.grating_results is not None, 'The model needs to be probed before calling this function.'
        max_output_index = self.grating_results['model_output'].argmax()

        return self.grating_results.iloc[max_output_index]['temporal_frequency']

    @property
    def preferred_model_output(self):
        """
        Maximum model output over all probed grating parameters 

        Returns
        -------
        output : numpy.float64
            Preferred model output
        """

        assert self.grating_results is not None, 'The model needs to be probed before calling this function.'
        return self.grating_results['model_output'].max()

    @property
    def orientation_tuning_curve(self):
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

        assert self.grating_results is not None, 'The model needs to be probed before calling this function.'

        max_response_by_theta = self.grating_results[
            (self.grating_results["spatial_frequency"] == self.preferred_spatial_freq) &
            (self.grating_results["temporal_frequency"] == self.preferred_temporal_freq)
        ]
        max_response_by_theta = max_response_by_theta.sort_values(by=['theta'])
        
        theta = max_response_by_theta[['theta']].to_numpy().flatten()
        tuning_curve = max_response_by_theta[['model_output']].to_numpy().flatten()

        return theta, tuning_curve

    @property
    def orientation_selectivity_index(self):
        """
        Orientation selectivity index, computed as 
        $OSI=\frac{{R}_{pref}^{Or}-{R}_{orth}^{Or}}{{R}_{pref}^{Or}+{R}_{orth}^{Or}}$ (Marques, Nguyen, Fioreze & Petreanu, 2018)

        Returns
        -------
        osi : numpy.float64
            Orientation selectivity index
        """

        assert self.grating_results is not None, 'The model needs to be probed before calling this function.'

        # Absolute orthogonal to preferred
        orthogonal_orientation = (self.preferred_orientation + np.pi/2) % 2*np.pi
        # Nearest angle to that actually probed
        nearest_orthogonal_orientation_idx = np.abs(self.thetas - orthogonal_orientation).argmin()
        nearest_orthogonal_orientation = self.thetas[nearest_orthogonal_orientation_idx]

        orthogonal_orientation_response = self.grating_results[
            (self.grating_results["theta"] == nearest_orthogonal_orientation) &
            (self.grating_results["temporal_frequency"] == self.preferred_temporal_freq) &
            (self.grating_results["spatial_frequency"] == self.preferred_spatial_freq)
        ]["model_output"].values[0]

        return ((self.preferred_model_output - orthogonal_orientation_response) /
                (self.preferred_model_output + orthogonal_orientation_response))

    @property
    def direction_selectivity_index(self):
        """
        Direction selectivity index, computed as 
        $DSI=\frac{{R}_{pref}^{Dir}-{R}_{opp}^{Dir}}{{R}_{pref}^{Dir}+{R}_{opp}^{Dir}}$ (Marques, Nguyen, Fioreze & Petreanu, 2018)

        Returns
        -------
        dsi : numpy.float64
            Direction selectivity index
        """

        assert self.grating_results is not None, 'The model needs to be probed before calling this function.'
        
        # Absolute opposite to preferred
        opposite_direction = (self.preferred_direction + np.pi) % 2*np.pi
        # Nearest angle to that actually probed
        nearest_opposite_orientation_idx = np.abs(self.thetas - opposite_direction).argmin()
        nearest_opposite_orientation = self.thetas[nearest_opposite_orientation_idx]

        opposite_direction_response = self.grating_results[
            (self.grating_results["theta"] == nearest_opposite_orientation) &
            (self.grating_results["temporal_frequency"] == self.preferred_temporal_freq) &
            (self.grating_results["spatial_frequency"] == self.preferred_spatial_freq)
        ]["model_output"].values[0]

        return ((self.preferred_model_output - opposite_direction_response) /
                (self.preferred_model_output + opposite_direction_response))

    @property
    def circular_variance(self):
        """
        Circular variance, computed as 
        $CV= 1 - \frac{\sqrt{ (\sum_{q}^{Q}r_q\sin{2\theta_q})^2 + (\sum_{q}^{Q}r_q\cos{2\theta_q})^2}}{\sum_{q}^{Q}r_q}$ (Singer, Willmore, King & Harper, 2019)

        Returns
        -------
        cv : numpy.float64
            Circular variance
        """

        assert self.grating_results is not None, 'The model needs to be probed before calling this function.'

        theta, tuning_curve = self.orientation_tuning_curve
        
        sum_sin = 0
        sum_cos = 0
        sum_r = 0
        
        for i in range(len(tuning_curve)):
            r = tuning_curve[i]
            th = theta[i]
            
            sum_sin += r*math.sin(2*th)
            sum_cos += r*math.cos(2*th)
            sum_r += r
            
        return 1 - math.sqrt(sum_sin**2 + sum_cos**2)/sum_r

    @property
    def orientation_bandwidth(self):
        """
        Orientation bandwith, described using the halfwidth at the halfmaximum of
        the orientation tuning curve's fitted Gaussian.

        $BW=\sqrt{2\ln{2}\sigma}$ (Jeon, Swain, Good Chase & Kuhlman, 2018)

        Returns
        -------
        bw : numpy.float64
            Orientation bandwidth
        """

        assert self.grating_results is not None, 'The model needs to be probed before calling this function.'

        theta, tuning_curve = self.orientation_tuning_curve

        # Average over 0-180 and 180-360 degrees
        orient_n = len(theta)//2
        orient_curve = (tuning_curve[:orient_n] + tuning_curve[:orient_n])/2
        orient_theta = theta[:orient_n]

        # Determine shift to bring peak into 'centre' for fitting
        # and rotate tuning curve by that shift
        shift = math.ceil(len(orient_curve)/2) - (np.argmax(orient_curve)+1)
        orient_curve_deque = collections.deque(orient_curve)
        orient_curve_deque.rotate(shift)
        orient_curve_rot = np.array(orient_curve_deque)

        loss, amplitude, mean, variance = GaussianFitter.fit_gaussian(
            xdata=orient_theta,
            ydata=orient_curve_rot,
            amplitude_init=max(orient_curve_rot),
            mean_init=orient_theta[np.argmax(orient_curve_rot)],
            variance_init=np.var(orient_curve_rot),
            n_iterations=self.GAUSSFIT_N_ITERATIONS
        )

        assert loss < self.GAUSSFIT_LOSS_THRESH, \
               'Could not determine orientation bandwidth. Gaussian fit min loss={0} did not meet threshold criteria {1}.'.format(loss, self.GAUSSFIT_LOSS_THRESH)

        return math.sqrt(2*math.log(2)) * math.sqrt(variance)

    @property
    def modulation_ratio(self):
        model_output = self.get_response_to_preferred_grating(self.n_timesteps).cpu()
        phases_init = [0]
        frequency_init = [(2 * np.pi) * self.preferred_temporal_freq / (1000 // self.dt)]
        F1 = self.get_sinusoid_fit_to_response(model_output.flatten().detach(), self.amplitudes_init, phases_init, self.offsets_init, frequency_init, self.n_iterations, self.loss_threshold)[0]
        F0 = model_output.mean().item()

        return F1 / F0

    def get_sinusoid_fit_to_response(self, model_output, amplitudes_init, phases_init, offsets_init, frequencies_init, n_iterations, loss_threshold):
        sinusoid_fitter = SinusoidFitter(model_output, amplitudes_init, phases_init, offsets_init, frequencies_init, n_iterations, loss_threshold)
        sinusoid_fitter.fit()

        return sinusoid_fitter.best_fit

    def get_response_to_preferred_grating(self, n_timesteps):
        gratings = self.generate_preferred_grating(n_timesteps)

        return self.model(gratings.unsqueeze(0))

    def generate_preferred_grating(self, n_timesteps):
        return self.generate_gratings(self.amplitude, self.rf_w, self.rf_h, self.preferred_direction,
                                      self.preferred_spatial_freq, self.preferred_temporal_freq, n_timesteps, self.dt)

#     def best_fix_temporal_freq(self):
#         raise NotImplementedError

#     def best_fix_spatial_freq(self):
#         raise NotImplementedError

class GaussianFitter:
    def __init__(self):
        pass

    @staticmethod
    def fit_gaussian(xdata, ydata, amplitude_init, mean_init, variance_init, n_iterations):
        x, y = torch.Tensor(xdata), torch.Tensor(ydata)

        amplitude = nn.Parameter(data=torch.Tensor([amplitude_init]), requires_grad=True)
        mean = nn.Parameter(data=torch.Tensor([mean_init]), requires_grad=True)
        variance = nn.Parameter(data=torch.Tensor([variance_init]), requires_grad=True)

        optimizer = torch.optim.Adam([amplitude, mean, variance], 10 ** -2)

        def predicted_gaussian():
            return amplitude * torch.exp(-(x-mean)**2 / (2*variance))

        for _ in range(n_iterations):
            optimizer.zero_grad()
            loss = F.mse_loss(predicted_gaussian(), y)
            loss.backward()
            optimizer.step()

        return loss.detach().item(), amplitude.detach().item(), mean.detach().item(), variance.detach().item()

class SinusoidFitter:

    def __init__(self, model_output, amplitudes_init, phases_init, offsets_init, frequencies_init, n_iterations, loss_threshold):
        self.model_output = model_output
        self.amplitudes = amplitudes_init
        self.phases = phases_init
        self.offsets = offsets_init
        self.frequencies = frequencies_init
        self.n_iterations = n_iterations
        self.loss_threshold = loss_threshold

        self._best_params = None

    @property
    def best_fit(self):
        return self._best_params

    def fit(self):
        min_loss = np.inf
        best_params = None

        for amplitude in self.amplitudes:
            for phase in self.phases:
                for offset in self.offsets:
                    for frequency in self.frequencies:
                        loss, *params = SinusoidFitter.fit_sinusoid(self.model_output, amplitude, phase, offset,
                                                                    frequency, self.n_iterations)
                        if loss < min_loss:
                            min_loss = loss
                            best_params = params

        assert min_loss < self.loss_threshold, 'Fit min loss={0} did not meet threshold criteria {1}.'.format(min_loss, self.loss_threshold)

        self._best_params = best_params

    @staticmethod
    def fit_sinusoid(model_output, amplitude_init, phase_init, offset_init, frequency_init, n_iterations):
        n_timesteps = model_output.shape[0]

        amplitude = nn.Parameter(data=torch.Tensor([amplitude_init]), requires_grad=True)
        phase = nn.Parameter(data=torch.Tensor([phase_init]), requires_grad=True)
        offset = nn.Parameter(data=torch.Tensor([offset_init]), requires_grad=True)
        frequency = nn.Parameter(data=torch.Tensor([frequency_init]), requires_grad=True)
        t = torch.arange(n_timesteps)

        optimizer = torch.optim.Adam([amplitude, phase, offset, frequency], 10 ** -2)

        def predicted_sinusoid():
            return amplitude * torch.sin(frequency * t + phase) + offset

        for _ in range(n_iterations):
            optimizer.zero_grad()
            loss = F.mse_loss(predicted_sinusoid(), model_output)
            loss.backward()
            optimizer.step()

        return loss.detach().item(), amplitude.detach().item(), phase.detach().item(), offset.detach().item(), frequency.detach().item()
