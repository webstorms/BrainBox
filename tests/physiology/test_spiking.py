from brainbox.physiology import spiking
from tests.physiology import test_util


def test_spiking():
    b_dim, n_dim, t_dim = 4, 4, 1000
    spike_trains = test_util.generate_spike_tensor(b_dim, n_dim, t_dim)
    isis_tensor = spiking.isi.compute_isis_tensor(spike_trains)

    def compute_isis(spike_train):
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

    def extract_isis(isis_tensor):
        return [isi.item() for isi in isis_tensor if isi > 0]

    for b in range(b_dim):
        for n in range(n_dim):
            isis = extract_isis(isis_tensor[b, n])
            target_isis = compute_isis(isis_tensor[b, n])

            assert isis == target_isis