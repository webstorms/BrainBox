import torch
import numpy as np


def get_isi_distributions(model, bin_dt, dataset, batch_size, max_batch=None):

    """
    Generate a list of inter-spike interval arrays.

    :param model: A function which maps input to output (batch x n_neurons x n_timesteps)
    :param bin_dt: The time difference between the spike bins (in milliseconds)
    :param dataset: The dataset to run throught the model
    :param batch_size: The number of samples per batch
    :param max_batch: Stop computing isis after specified number of batches (used for computational speedup)
    :return: A list of numpy arrays
    """

    def process_batch(model_output):
        # model_output: n_neurons x n_timesteps
        n_neurons = model_output.shape[0]
        n_timesteps = model_output.shape[1]
        indices = np.arange(n_neurons)
        last_spike_index = np.zeros(n_neurons)
        first_spike = np.ones(n_neurons)

        for t in range(n_timesteps):

            spikes_at_t = model_output[:, t].cpu()
            c = spikes_at_t > 0
            isi = (t - last_spike_index[c]) * bin_dt
            last_spike_index[c] = t

            for i, j in enumerate(indices[c]):

                # Only add isi if it is for spikes occurring after the first
                if first_spike[i] == 0:
                    results[j].append(isi[i])
            first_spike[c] = 0

    results = None
    data_loader = torch.utils.data.DataLoader(dataset, batch_size)

    with torch.no_grad():
        for batch_id, (data, _) in enumerate(data_loader):
            model_output = model(data)
            assert len(model_output.shape) == 3, 'Model output should be of dimensions n_batch x n_neurons x n_timesteps'

            n_batch = model_output.shape[0]
            n_neurons = model_output.shape[1]

            if results is None:
                results = [[] for _ in range(n_neurons)]

            for sample_id in range(n_batch):
                print('Processing batch... {0}.{1}/{2}.{3}'.format(batch_id, sample_id, len(data_loader), n_batch))
                process_batch(model_output[sample_id, :, :])

            # Stop sampling isis if we have reached the target number batches to compute over
            if max_batch is not None and max_batch == (batch_id + 1):
                break

    # Convert isis to numpy arrays
    n_neurons = len(results)
    for i in range(n_neurons):
        results[i] = np.array(results[i])

    return results


def get_isi_cvs(isi_distributions, thresh_n):

    """
    Obtain a list of Coefficient of Variation (CV) values for a given list of isi distributions. The CV is a measure of
    irregularity of the spike trains produced by a neuron. It is computed as the standard deviation divided by the mean
    of the isi distribution of a neuron. See chapter 2.7 in Neuronal Dynamics (https://neuronaldynamics.epfl.ch/online/Ch7.S3.html for more details).

    :param isi_distributions: A list of numpy arrays containing the isis (where every array is associated with a distinct neuron)
    :param thresh_n: Only compute CV when at least thresh_n number of isi values have been provided. Otherwise the CV is set to 0
    :return: A numpy array of CV values (one for each neuron)
    """

    n_neurons = len(isi_distributions)
    isi_cvs = []

    for i in range(n_neurons):
        neuron_isis = isi_distributions[i]
        if len(neuron_isis) >= thresh_n:
            isi_cv = neuron_isis.std() / neuron_isis.mean()
            isi_cvs.append(isi_cv)
        else:
            isi_cvs.append(0)

    return np.array(isi_cvs)


def get_firing_rate_distributions(model, bin_dt, dataset, batch_size, max_batch=None):

    # TODO: Documentation

    def process_batch(model_output):
        # model_output: n_neurons x n_timesteps
        firing_rates = model_output.mean(dim=-1)

        for i in range(len(firing_rates)):
            results[i].append(firing_rates[i].item() * (1000 / bin_dt))

    results = None
    data_loader = torch.utils.data.DataLoader(dataset, batch_size)

    with torch.no_grad():
        for batch_id, (data, _) in enumerate(data_loader):
            model_output = model(data)
            assert len(
                model_output.shape) == 3, 'Model output should be of dimensions n_batch x n_neurons x n_timesteps'

            n_batch = model_output.shape[0]
            n_neurons = model_output.shape[1]

            if results is None:
                results = [[] for _ in range(n_neurons)]

            for sample_id in range(n_batch):
                print('Processing batch... {0}.{1}/{2}.{3}'.format(batch_id, sample_id, len(data_loader), n_batch))
                process_batch(model_output[sample_id, :, :])

            # Stop sampling isis if we have reached the target number batches to compute over
            if max_batch is not None and max_batch == (batch_id + 1):
                break

    # Convert isis to numpy arrays
    n_neurons = len(results)
    for i in range(n_neurons):
        results[i] = np.array(results[i])

    return results


def get_mean_firing_rates(firing_rate_distributions):

    n_neurons = len(firing_rate_distributions)
    mean_firing_rates = []

    for i in range(n_neurons):
        neuron_firing_rates = firing_rate_distributions[i]

        mean_firing_rates.append(neuron_firing_rates.mean())

    return np.array(mean_firing_rates)


# TODO: Get firing rate distribution
# TODO: Get isi distribution
# TODO: Add other neural spiking measures
