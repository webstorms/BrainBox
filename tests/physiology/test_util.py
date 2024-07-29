import torch

from brainbox.spiking import util


def generate_spike_tensor(b=2, n=10, t=100, f=0.1):
    foo = torch.rand(b, n, t)
    foo[foo >= f] = 0
    foo[foo != 0] = 1

    return foo


def test_mean():
    patchy_tensor = _generate_patchy_tensor(b=2, n=4)

    def compute_mean(time_series, ignore_c):
        return torch.Tensor([e for e in time_series if e != ignore_c]).mean()

    mean_tensor = util.get_mean(patchy_tensor, ignore_c=0)
    _assert_measure(patchy_tensor, compute_mean, mean_tensor, ignore_c=0)


def test_std():
    patchy_tensor = _generate_patchy_tensor(b=2, n=4)

    def compute_std(time_series, ignore_c):
        return torch.Tensor([e for e in time_series if e != ignore_c]).std(
            unbiased=False
        )

    std_tensor = util.get_std(patchy_tensor, ignore_c=0)
    _assert_measure(patchy_tensor, compute_std, std_tensor, ignore_c=0)


def _assert_measure(input_tensor, measure, target_tensor, ignore_c):
    b_dim, n_dim, _ = input_tensor.shape
    for b_idx in range(b_dim):
        for n_idx in range(n_dim):
            expected_measure = measure(input_tensor[b_idx, n_idx], ignore_c)
            assert _compare_float(target_tensor[b_idx, n_idx], expected_measure)


def _generate_patchy_tensor(b=2, n=10, t=100, patchy_f=0.1):
    foo = torch.rand(b, n, t)
    foo[foo < patchy_f] = 0

    return foo


def _compare_float(v, exp_v, thresh=1e-5):
    return abs(v - exp_v) < thresh
