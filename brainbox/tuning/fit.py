import os
import sys
import logging
import math
import collections

import torch
import pandas as pd
import numpy as np

from brainbox.tuning.query import TuningQuery

logger = logging.getLogger("tuning")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class GratingsProber:

    GRATING_THETAS = np.linspace(0, np.pi * 2, 10)
    GRATINGS_SPATIAL_FREQS = np.linspace(0.05, 0.2, 10)
    GRATINGS_TEMPORAL_FREQS = np.linspace(1, 10, 10)

    def __init__(
        self,
        model,
        amplitude,
        rf_w,
        rf_h,
        duration,
        dt,
        thetas=None,
        spatial_freqs=None,
        temporal_freqs=None,
    ):
        self.model = model
        self.amplitude = amplitude
        self.rf_w = rf_w
        self.rf_h = rf_h
        self.duration = duration
        self.dt = dt
        self.thetas = np.array(
            GratingsProber.GRATING_THETAS if thetas is None else thetas
        )
        self.spatial_freqs = np.array(
            GratingsProber.GRATINGS_SPATIAL_FREQS
            if spatial_freqs is None
            else spatial_freqs
        )
        self.temporal_freqs = np.array(
            GratingsProber.GRATINGS_TEMPORAL_FREQS
            if temporal_freqs is None
            else temporal_freqs
        )

        self._tuning_query = None

    def probe_and_fit(self, path, probe_batch=1024, response_batch=32, device="cuda"):
        probe_path = os.path.join(path, "probe.csv")
        spectral_path = os.path.join(path, "spectral.csv")

        logging.info("Probing the model...")
        self.probe(probe_path, probe_batch, device)
        self._tuning_query = TuningQuery(path)

        logging.info("Obtaining all unit responses to optimal gratings...")
        gratings = self.generate_preferred_grating(
            self.amplitude, self.rf_w, self.rf_h, self.duration, self.dt
        )
        unit_responses = self.get_response_to_preferred_grating(
            self.amplitude,
            self.rf_w,
            self.rf_h,
            self.duration,
            self.dt,
            batch_size=response_batch,
            device=device,
        )
        torch.save(gratings.cpu(), os.path.join(path, "gratings.pt"))
        torch.save(unit_responses.cpu(), os.path.join(path, "unit_responses.pt"))
        self.compute_and_save_spectral_analysis(spectral_path, unit_responses)

    def probe(self, path, batch_size=1024, device="cuda"):
        def populate_mean_model_responses(batch_gratings, model_output_list):
            batch_gratings = torch.stack(batch_gratings).to(device)
            model_outputs = self.model(batch_gratings)
            model_outputs = model_outputs.mean(dim=2)
            n_units = model_outputs.shape[1]

            # Initialize model output list if not already done
            # We initialize after obtaining model responses as the number
            # of model units are unknown before passing grating
            if model_output_list is None:
                model_output_list = {str(i): [] for i in range(n_units)}

            # Store mean response of every unit for every sample
            for i in range(n_units):
                model_output_list[str(i)].extend(
                    [
                        model_outputs[b, i].cpu().item()
                        for b in range(len(batch_gratings))
                    ]
                )

            return model_output_list

        batch_gratings = []
        thetas_list = []
        spatial_freq_list = []
        temporal_freq_list = []
        model_output_list = None

        for theta in self.thetas:
            for spatial_freq in self.spatial_freqs:
                for temporal_freq in self.temporal_freqs:
                    gratings = GratingsProber.generate_grating(
                        self.amplitude,
                        self.rf_w,
                        self.rf_h,
                        theta,
                        spatial_freq,
                        temporal_freq,
                        self.duration,
                        self.dt,
                    )
                    batch_gratings.append(gratings)

                    if len(batch_gratings) == batch_size:
                        model_output_list = populate_mean_model_responses(
                            batch_gratings, model_output_list
                        )

                        batch_gratings = []

                    thetas_list.append(theta)
                    spatial_freq_list.append(spatial_freq)
                    temporal_freq_list.append(temporal_freq)

        if len(batch_gratings) > 0:
            model_output_list = populate_mean_model_responses(
                batch_gratings, model_output_list
            )

        grating_results = pd.DataFrame(
            {
                "theta": thetas_list,
                "spatial_frequency": spatial_freq_list,
                "temporal_frequency": temporal_freq_list,
                **model_output_list,
            }
        )
        grating_results.to_csv(path, index=False)

    @staticmethod
    def generate_grating(
        amplitude, rf_w, rf_h, theta, spatial_freq, temporal_freq, duration, dt
    ):
        y, x = torch.meshgrid(
            [
                torch.arange(rf_h, dtype=torch.float32),
                torch.arange(rf_w, dtype=torch.float32),
            ],
            indexing="ij",
        )
        theta = torch.Tensor([theta]).expand_as(y)
        spatial_freq = torch.Tensor([spatial_freq]).expand_as(y)

        fps = int(1000 / dt)
        n_timesteps = int(duration / dt)
        timesteps = (
            torch.arange(n_timesteps).view(n_timesteps, 1, 1).repeat(1, rf_h, rf_w)
        )

        rotx = x * torch.cos(theta) - y * torch.sin(theta)
        grating = amplitude * torch.sin(
            2 * np.pi * (spatial_freq * rotx - temporal_freq * timesteps / fps)
        )

        return grating

    def get_response_to_preferred_grating(
        self,
        amplitude,
        rf_w,
        rf_h,
        duration,
        dt,
        unit_id=None,
        batch_size=10,
        device="cuda",
    ):
        gratings = self.generate_preferred_grating(
            amplitude, rf_w, rf_h, duration, dt, unit_id
        )
        gratings = gratings.to(device)

        if unit_id is not None:
            return self.model(gratings.unsqueeze(0))[0, int(unit_id)]
        else:
            batch_response_to_preferred_grating = []

            for i in range(
                self._tuning_query.n_units // batch_size + 1
                if self._tuning_query.n_units % batch_size > 0
                else 0
            ):
                batch_gratings = gratings[i * batch_size : (i + 1) * batch_size]
                batch_response_to_preferred_grating.append(self.model(batch_gratings))

            response_to_preferred_grating = torch.cat(
                batch_response_to_preferred_grating
            )
            response_to_preferred_grating = torch.stack(
                [
                    response_to_preferred_grating[i, i]
                    for i in range(self._tuning_query.n_units)
                ]
            )

            return response_to_preferred_grating

    def generate_preferred_grating(
        self, amplitude, rf_w, rf_h, duration, dt, unit_id=None
    ):
        if unit_id is not None:
            return GratingsProber.generate_grating(
                amplitude,
                rf_w,
                rf_h,
                self._tuning_query.preferred_direction(unit_id),
                self._tuning_query.preferred_spatial_frequency(unit_id),
                self._tuning_query.preferred_temporal_frequency(unit_id),
                duration,
                dt,
            )
        else:
            return torch.stack(
                [
                    self.generate_preferred_grating(
                        amplitude, rf_w, rf_h, duration, dt, str(idx)
                    )
                    for idx in range(self._tuning_query.n_units)
                ]
            )

    def compute_and_save_spectral_analysis(self, spectral_path, unit_responses):
        spectral_results_dict = {
            str(i): SpectralAnalysis(unit_responses[i].cpu()).get_dict()
            for i in range(unit_responses.shape[0])
        }
        spectral_results_df = pd.DataFrame(spectral_results_dict).T
        spectral_results_df.to_csv(spectral_path, index=False)


class SpectralAnalysis:
    def __init__(self, signal):
        self._n_samples = len(signal)
        self._spectral = torch.fft.rfft(signal.cpu())

    @property
    def n_samples(self):
        return self._n_samples

    @property
    def amplitudes(self):
        return self._spectral.abs() / self._n_samples

    @property
    def DC(self):
        return self.amplitudes[0]

    @property
    def first_harmonic_amplitude(self):
        return 2 * self.amplitudes[1:].max()

    @property
    def first_harmonic_frequency(self):
        return (self.amplitudes[1:].argmax() + 1) / self._n_samples

    @property
    def first_harmonic_phase(self):
        harmonic = self._spectral[self.amplitudes[1:].argmax() + 1]
        real_harmonic = harmonic.real
        imaginary_harmonic = harmonic.imag
        return torch.atan2(imaginary_harmonic, real_harmonic) + np.pi / 2

    @property
    def modulation_ratio(self):
        return self.first_harmonic_amplitude / self.DC

    def get_dict(self):
        return {
            "DC": self.DC.item(),
            "first_harmonic_amplitude": self.first_harmonic_amplitude.item(),
            "first_harmonic_frequency": self.first_harmonic_frequency.item(),
            "first_harmonic_phase": self.first_harmonic_phase.item(),
            "modulation_ratio": self.modulation_ratio.item(),
        }
