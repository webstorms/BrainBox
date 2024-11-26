import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)


class TuningQuery:
    def __init__(self, path):
        probe_path = os.path.join(path, "probe.csv")
        spectral_path = (
            os.path.join(path, "spectral.csv")
            if os.path.exists(os.path.join(path, "spectral.csv"))
            else None
        )
        self._probes_df = pd.read_csv(probe_path)
        self._spectral_df = (
            pd.read_csv(spectral_path) if spectral_path is not None else None
        )
        try:
            self._gratings = torch.load(os.path.join(path, "gratings.pt"))
            self._responses = torch.load(os.path.join(path, "unit_responses.pt"))
        except:
            pass

    @property
    def n_units(self):
        return len(self._probes_df.columns) - 3

    def build_sinusoid_from_spectral(self, unit_id):
        spectral_data = self._spectral_df.iloc[unit_id]
        a = spectral_data["first_harmonic_amplitude"]
        f = spectral_data["first_harmonic_frequency"]
        phase = spectral_data["first_harmonic_phase"]
        offset = spectral_data["DC"]
        n_samples = self._responses.shape[1]
        return TuningQuery.build_sinusoid(a, f, phase, offset, n_samples)

    @staticmethod
    def build_sinusoid(a, f, phase, offset, n_samples):
        return a * torch.sin(f * 2 * np.pi * torch.arange(n_samples) + phase) + offset

    def validate(self, response_threshold=0.01, fit_threshold=0.5, additional_data={}):
        def get_unit_statistics(i):
            unit_response = self._responses[i].numpy()
            first_harmonic = self.build_sinusoid_from_spectral(i).numpy()
            fit_cc = np.corrcoef(unit_response, first_harmonic)[0, 1]

            return {
                "mean_response": unit_mean_responses[i].item(),
                "fit_cc": fit_cc,
                "theta": self.preferred_direction(i),
                "sf": self.preferred_spatial_frequency(i),
                "tf": self.preferred_temporal_frequency(i),
                "OSI": self.orientation_selectivity_index(i),
                "DSI": self.direction_selectivity_index(i),
                "F1F0": self._spectral_df.iloc[i]["modulation_ratio"],
            }

        unit_mean_responses = self._responses.mean(dim=1)
        unit_statistics_dict = {i: get_unit_statistics(i) for i in range(self.n_units)}
        unit_statistics_df = pd.DataFrame(unit_statistics_dict).T

        for key, column in additional_data.items():
            unit_statistics_df[key] = column

        mean_responses = unit_mean_responses.mean().item()
        query = (
            unit_statistics_df["mean_response"] > mean_responses * response_threshold
        ) & (unit_statistics_df["fit_cc"] > fit_threshold)

        return unit_statistics_df[query]

    def preferred_direction(self, unit_id=0):
        max_output_index = self._probes_df[str(unit_id)].argmax()

        return self._probes_df.iloc[max_output_index]["theta"]

    def preferred_spatial_frequency(self, unit_id):
        max_output_index = self._probes_df[str(unit_id)].argmax()

        return self._probes_df.iloc[max_output_index]["spatial_frequency"]

    def preferred_temporal_frequency(self, unit_id):
        max_output_index = self._probes_df[str(unit_id)].argmax()

        return self._probes_df.iloc[max_output_index]["temporal_frequency"]

    def preferred_model_output(self, unit_id):
        return self._probes_df[str(unit_id)].max()

    def orientation_tuning_curve(self, unit_id):
        preferred_spatial_and_temporal_frequency_query = (
            self._probes_df["spatial_frequency"]
            == self.preferred_spatial_frequency(unit_id)
        ) & (
            self._probes_df["temporal_frequency"]
            == self.preferred_temporal_frequency(unit_id)
        )
        response_per_theta = self._probes_df[
            preferred_spatial_and_temporal_frequency_query
        ]
        response_per_theta = response_per_theta.sort_values(by=["theta"])
        theta = response_per_theta[["theta"]].to_numpy().flatten()
        response = response_per_theta[[str(unit_id)]].to_numpy().flatten()

        return theta, response

    def orientation_sf_tuning_curve(self, unit_id):
        preferred_temporal_frequency_query = self._probes_df[
            "temporal_frequency"
        ] == self.preferred_temporal_frequency(unit_id)
        response_per_theta = self._probes_df[preferred_temporal_frequency_query]
        # return response_per_theta
        response_per_theta = response_per_theta.sort_values(
            by=["theta", "spatial_frequency"]
        )

        theta = response_per_theta[["theta"]].to_numpy().flatten()
        sf = response_per_theta[["spatial_frequency"]].to_numpy().flatten()
        response = response_per_theta[[str(unit_id)]].to_numpy().flatten()

        return theta, sf, response

    def orientation_tf_tuning_curve(self, unit_id):
        preferred_temporal_frequency_query = self._probes_df[
            "spatial_frequency"
        ] == self.preferred_spatial_frequency(unit_id)
        response_per_theta = self._probes_df[preferred_temporal_frequency_query]
        # return response_per_theta
        response_per_theta = response_per_theta.sort_values(
            by=["theta", "temporal_frequency"]
        )

        theta = response_per_theta[["theta"]].to_numpy().flatten()
        tf = response_per_theta[["temporal_frequency"]].to_numpy().flatten()
        response = response_per_theta[[str(unit_id)]].to_numpy().flatten()

        return theta, tf, response

    def orientation_selectivity_index(self, unit_id):
        thetas = self._probes_df["theta"].unique()

        # Absolute orthogonal to preferred
        orthogonal_orientation = (self.preferred_direction(unit_id) + np.pi / 2) % (
            2 * np.pi
        )
        # Nearest angle to that actually probed
        nearest_orthogonal_orientation_idx = np.abs(
            thetas - orthogonal_orientation
        ).argmin()
        orthogonal_orientation = thetas[nearest_orthogonal_orientation_idx]

        orthogonal_orientation_response = self._probes_df[
            (self._probes_df["theta"] == orthogonal_orientation)
            & (
                self._probes_df["temporal_frequency"]
                == self.preferred_temporal_frequency(unit_id)
            )
            & (
                self._probes_df["spatial_frequency"]
                == self.preferred_spatial_frequency(unit_id)
            )
        ][str(unit_id)].values[0]

        return (
            self.preferred_model_output(unit_id) - orthogonal_orientation_response
        ) / (self.preferred_model_output(unit_id) + orthogonal_orientation_response)

    def direction_selectivity_index(self, unit_id):
        thetas = self._probes_df["theta"].unique()

        # Absolute opposite to preferred
        opposite_direction = (self.preferred_direction(unit_id) + np.pi) % (2 * np.pi)
        # Nearest angle to that actually probed
        nearest_opposite_orientation_idx = np.abs(thetas - opposite_direction).argmin()
        nearest_opposite_orientation = thetas[nearest_opposite_orientation_idx]

        opposite_direction_response = self._probes_df[
            (self._probes_df["theta"] == nearest_opposite_orientation)
            & (
                self._probes_df["temporal_frequency"]
                == self.preferred_temporal_frequency(unit_id)
            )
            & (
                self._probes_df["spatial_frequency"]
                == self.preferred_spatial_frequency(unit_id)
            )
        ][str(unit_id)].values[0]

        return (self.preferred_model_output(unit_id) - opposite_direction_response) / (
            self.preferred_model_output(unit_id) + opposite_direction_response
        )
