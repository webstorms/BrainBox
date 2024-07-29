import numpy as np

from brainbox import spiking
from . import test_util


def test_isis():
    b_dim, n_dim, t_dim = 4, 4, 1000
    spike_trains = test_util.generate_spike_tensor(b_dim, n_dim, t_dim)
    isis_tensor = spiking.isi.compute_isis_tensor(spike_trains)

    def extract_isis(isis_tensor):
        return [isi.item() for isi in isis_tensor if isi > 0]

    for b in range(b_dim):
        for n in range(n_dim):
            isis = extract_isis(isis_tensor[b, n])
            target_isis = _compute_isis(spike_trains[b, n])

            assert isis == target_isis


def test_isi_cvs():
    n_spikes_thresh = 5
    b_dim, n_dim, t_dim = 4, 10, 100
    spike_trains = test_util.generate_spike_tensor(b_dim, n_dim, t_dim, f=0.1)
    isis_tensor = spiking.isi.compute_isis_tensor(spike_trains)
    isi_cvs = spiking.isi.compute_isi_cvs(isis_tensor, n_spikes_thresh)

    def get_cv(isis):
        if len(isis) < n_spikes_thresh:
            return 0
        return np.std(isis) / np.mean(isis)

    for b in range(b_dim):
        for n in range(n_dim):
            isi_cv = isi_cvs[b, n]
            target_isi_cv = get_cv(_compute_isis(spike_trains[b, n]))
            assert abs(isi_cv - target_isi_cv) < 1e5


def _compute_isis(spike_train):
    last_spike = None
    isis = []
    for i in range(len(spike_train)):
        if spike_train[i] == 1:
            if last_spike is None:
                last_spike = i
            else:
                isis.append(i - last_spike)
                last_spike = i
    return isis
