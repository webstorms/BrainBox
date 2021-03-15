import os
import argparse
from pathlib import Path

import torch
from torchvision.transforms import transforms
import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat
from scipy.cluster.vq import kmeans


class PVC1Config:

    MAX_MOVIE = 4
    MAX_SEG = 30
    MAX_FRAMES = 900

    MAX_DATASET = 7
    MAX_CHANNEL = 33

    SINGLE_TRIAL_REP = 1
    MULTI_TRIAL_REP = 10


class MovieBuilder:

    def __init__(self, src_path, scale):
        self.src_path = src_path
        self.scale = scale

    def load_image(self, movie_id, seg_id, image_id):
        movie_folder = 'movie{0:03d}_{1:03d}.images'.format(movie_id, seg_id)
        image_name = 'movie{0:03d}_{1:03d}_{2:03d}.jpeg'.format(movie_id, seg_id, image_id)
        image_path = os.path.join(self.src_path, 'movie_frames', movie_folder, image_name)

        return Image.open(image_path)

    def get_segment(self, movie_id, seg_id):
        to_tensor = transforms.ToTensor()

        images = []

        for image_id in range(PVC1Config.MAX_FRAMES):
            # Get the image and rescale
            image = self.load_image(movie_id, seg_id, image_id)
            width, height = image.size
            rescale_width, rescale_height = int(self.scale * width), int(self.scale * height)
            image = image.resize((rescale_width, rescale_height), Image.ANTIALIAS)

            # Image to Tensor
            image = (255 * to_tensor(image))
            images.append(image)

        segment = torch.stack(images)

        return segment

    def build(self):
        movies = []

        for movie_id in range(PVC1Config.MAX_MOVIE):
            for seg_id in range(PVC1Config.MAX_SEG):
                print('Generating segment {0} of movie {1}'.format(seg_id, movie_id))

                images = self.get_segment(movie_id, seg_id)
                movies.append(images)

        movies = torch.stack(movies)

        return movies


class SpikeFrameBuilder:

    def __init__(self, src_path):
        self.src_path = src_path

    @staticmethod
    def get_spikes(dataset_id, channel_id, src_path):

        src_path = Path(src_path)
        # Adapted from https://github.com/ben-willmore/ringach-pvc1

        def unpack(array):
            while True:
                if isinstance(array, np.ndarray):
                    if any([s > 1 for s in array.shape]):
                        return array.squeeze()
                    array = array[0]
                else:
                    return array

        def get_dataset_paths(src_path):
            data_files = []
            for subdir in sorted(src_path.iterdir()):
                for file in sorted(subdir.iterdir()):
                    data_files.append(file)

            # Sort by filename to ensure equivalent dataset construction
            # on different machines using different filepaths
            data_files = sorted(data_files, key=lambda path: str(path).split('/')[-1])

            print('Loaded the ringach data files...')
            print(data_files)

            return data_files

        def convert_dataset(pepANA, channel_id):
            lor = unpack(unpack(pepANA)['listOfResults'])
            n_conditions = lor.shape[0]
            conditions = []

            for condition_idx in range(n_conditions):
                res = lor[condition_idx]
                condition_params = unpack(res['condition'])
                n_reps = unpack(res['noRepeats'])
                spike_times = []
                spike_shapes = []
                rep_numbers = []
                reps = unpack(res['repeat'])

                sym = [unpack(s) for s in unpack(res['symbols'])]

                if 'movie_id' not in sym:
                    continue

                val = [unpack(s) for s in unpack(res['values'])]

                for rep_idx in range(n_reps):
                    if n_reps == 1:
                        rep = unpack(reps)
                    else:
                        rep = unpack(reps[rep_idx])
                    dat = unpack(unpack(rep['data'])[channel_id])
                    spike_times.append(unpack(dat[0]))
                    spike_shapes.append(unpack(dat[1]))
                    rep_numbers.append(np.array([rep_idx] * spike_times[-1].shape[0]))

                conditions.append({'condition_params': condition_params,
                                   'symbols': sym,
                                   'values': val,
                                   'n_reps': n_reps,
                                   'spike_times': np.concatenate(spike_times),
                                   'spike_shapes': np.concatenate(spike_shapes, axis=1),
                                   'rep_numbers': np.concatenate(rep_numbers)})

            return {'n_conditions': n_conditions, 'conditions': conditions}

        def get_spike_frames(dataset):
            all_spike_shapes = np.concatenate([c['spike_shapes'] for c in dataset['conditions']], axis=1)
            n_spikes = all_spike_shapes.shape[1]
            codebook, _ = kmeans(all_spike_shapes.astype(np.double).transpose(), 2)
            norm = np.sum(codebook ** 2, axis=1)

            # 0th cluster is noise, 1st cluster is spikes
            if norm[1] < norm[0]:
                codebook = codebook[::-1, :]

            for i, c in enumerate(dataset['conditions']):
                spike_times = []
                spike_shapes = []
                rep_numbers = []
                for idx in range(c['spike_times'].shape[0]):
                    time = c['spike_times'][idx]
                    shape = c['spike_shapes'][:, idx]
                    rep = c['rep_numbers'][idx]
                    d0 = np.sum((shape - codebook[0]) ** 2)
                    d1 = np.sum((shape - codebook[1]) ** 2)
                    if d1 < d0:
                        spike_times.append(time)
                        spike_shapes.append(shape)
                        rep_numbers.append(rep)
                if len(spike_times) == 0:
                    dataset['conditions'][i]['spike_times'] = np.array([])
                    dataset['conditions'][i]['spike_shapes'] = np.array([])
                    dataset['conditions'][i]['rep_numbers'] = np.array([])
                dataset['conditions'][i]['spike_times'] = np.array(spike_times)
                if dataset['conditions'][i]['spike_times'].shape[0] == 0:
                    dataset['conditions'][i]['spike_shapes'] = \
                        np.zeros((all_spike_shapes.shape[0], 0))
                else:
                    dataset['conditions'][i]['spike_shapes'] = np.stack(spike_shapes, axis=1)
                dataset['conditions'][i]['rep_numbers'] = np.array(rep_numbers)

            assignment = np.zeros(n_spikes)
            for i in range(n_spikes):
                d0 = np.sum((all_spike_shapes[:, i] - codebook[0]) ** 2)
                d1 = np.sum((all_spike_shapes[:, i] - codebook[1]) ** 2)
                assignment[i] = 0 if d0 < d1 else 1

            spike_frames = []
            for condition in dataset['conditions']:
                if condition['symbols'] != ['movie_id', 'segment_id']:
                    raise ValueError('Unexpected parameters, not movie_id and segment_id')
                frame_idxes = (np.floor(condition['spike_times'] / 3 * 90)).astype(np.int)
                spike_frames.append((condition['values'][0], condition['values'][1], condition['rep_numbers'], frame_idxes))

            return spike_frames

        dataset_path = get_dataset_paths(src_path)[dataset_id]
        pepANA = loadmat(dataset_path)['pepANA']
        dataset = convert_dataset(pepANA, channel_id)
        spikes = get_spike_frames(dataset)

        return spikes

    def build(self):

        spike_frames = {}

        for dataset_id in range(PVC1Config.MAX_DATASET):
            for channel_id in range(PVC1Config.MAX_CHANNEL):

                print('Retrieving spike frames dataset={0} and channel={1}'.format(dataset_id, channel_id))

                try:
                    spikes = SpikeFrameBuilder.get_spikes(dataset_id, channel_id, self.src_path)

                    for spike_frame in spikes:
                        movie_id = spike_frame[0]
                        segment_id = spike_frame[1]
                        key = (movie_id, segment_id, channel_id)

                        repetitions = spike_frame[2]
                        spike_times = spike_frame[3]

                        if spike_frames.get(key) is None:
                            spike_frames[key] = {}

                        spike_frames[key][dataset_id] = [repetitions, spike_times]

                except Exception as e:
                    print(e)

        return spike_frames


class NeuralResponseBuilder:

    def __init__(self, spike_frames):
        self.spike_frames = spike_frames

    def build(self, rep_query):
        movie_ids, neuron_ids = self.get_movie_and_neuron_ids(rep_query)

        neural_responses = {}
        trial_neuron_ids = set()

        for movie_id in movie_ids:

            orig_movie_id = movie_id[0]
            seg_id = movie_id[1]

            for neuron_id in neuron_ids:

                dataset_id = neuron_id[0]
                channel_id = neuron_id[1]

                neuron_spike_frames = self.spike_frames[(orig_movie_id, seg_id, channel_id)][dataset_id]
                rep_codes = neuron_spike_frames[0]
                spike_frames_all_reps = neuron_spike_frames[1]

                # Get the list of spikes frame per repetition
                spike_frames_per_rep = NeuralResponseBuilder.get_spike_frames_per_rep(rep_codes, spike_frames_all_reps)

                # Convert the spike frames per repetition into spike counts per repetition
                spike_counts_per_rep = []

                for spike_frames in spike_frames_per_rep:
                    spike_counts = NeuralResponseBuilder.get_spike_counts(spike_frames, PVC1Config.MAX_FRAMES)
                    spike_counts_per_rep.append(spike_counts)

                spike_counts_per_rep = np.array(spike_counts_per_rep)
                if spike_counts_per_rep.shape[0] > 1:
                    trial_neuron_ids.add(neuron_id)

                if neural_responses.get(movie_id) is None:
                    neural_responses[movie_id] = {}

                neural_responses[movie_id][neuron_id] = spike_counts_per_rep

        return NeuralResponseBuilder.tensorise_neural_responses(neural_responses)

    def get_movie_and_neuron_ids(self, rep_query):
        # First we build a table of which neurons (denoted by neuron_id) have a recording
        # for each movie (denoted by movie_id)
        recordings_info = []

        for movie_id in range(PVC1Config.MAX_MOVIE):
            for seg_id in range(PVC1Config.MAX_SEG):
                for channel_id in range(PVC1Config.MAX_CHANNEL):

                    neuron_spike_frames = self.spike_frames[(movie_id, seg_id, channel_id)]

                    for dataset_id in neuron_spike_frames:

                        # Ignore the empty arrays
                        if len(neuron_spike_frames[dataset_id][0]) == 0:
                            continue

                        n_rep = neuron_spike_frames[dataset_id][0].max() + 1

                        recordings_info.append({
                            'movie_id': (movie_id, seg_id),
                            'neuron_id': (dataset_id, channel_id),
                            'n_rep': n_rep
                        })

        recordings_info = pd.DataFrame(recordings_info)

        neuron_ids = recordings_info[recordings_info['n_rep'] == rep_query]['neuron_id'].unique()
        movie_ids = None

        for neuron_id in neuron_ids:

            neuron_id_query = (recordings_info['neuron_id'] == neuron_id) & (recordings_info['n_rep'] == rep_query)

            if movie_ids is None:
                movie_ids = pd.Index(recordings_info[neuron_id_query]['movie_id'])
            else:
                movie_ids = movie_ids.intersection(recordings_info[neuron_id_query]['movie_id'])

        return movie_ids, neuron_ids

    @staticmethod
    def tensorise_neural_responses(neural_responses):
        movie_ids = list(neural_responses.keys())
        neuron_ids = list(neural_responses[movie_ids[0]].keys())
        all_movie_responses = []

        for movie_id in movie_ids:

            movie_responses = []

            for neuron_id in neuron_ids:
                neuron_responses = torch.from_numpy(neural_responses[movie_id][neuron_id])
                movie_responses.append(neuron_responses)

            all_movie_responses.append(torch.stack(movie_responses))

        return torch.stack(all_movie_responses)

    @staticmethod
    def get_spike_frames_per_rep(reps, spikes):
        n_reps = reps.max() + 1

        spike_frames_per_rep = [[] for _ in range(n_reps)]

        for rep_id, spike_frame_id in zip(reps, spikes):
            spike_frames_per_rep[rep_id].append(spike_frame_id)

        return spike_frames_per_rep

    @staticmethod
    def get_spike_counts(spike_times, max_frame):
        # Convert spike frames into spike counts
        spike_counts = np.zeros(max_frame)

        for spike_time, spike_count in zip(*np.unique(spike_times, return_counts=True)):
            if spike_time >= max_frame:
                continue
            spike_counts[spike_time] = spike_count

        return spike_counts


def main():

    # Building settings
    parser = argparse.ArgumentParser(description='Build crcns-build_ringach-data responses')
    parser.add_argument('--path', type=str, default='.', metavar='N', help='source path(default: .)')
    parser.add_argument('--scale', type=float, default=0.4, metavar='N', help='rescale the crcns-build_ringach-data images by scaling factor (default: 0.5)')
    args = parser.parse_args()

    # 1. Build movies
    movies_src_path = os.path.join(args.path, 'crcns-ringach-data')
    movies_dst_path = os.path.join(args.path, 'processed', 'movies.pt')
    movie_builder = MovieBuilder(movies_src_path, args.scale)
    movies = movie_builder.build()
    torch.save(movies, movies_dst_path)

    # # 2. Build neural responses
    # # 2.1. Convert matlab spike frames to python
    frames_src_path = os.path.join(args.path, 'crcns-ringach-data', 'neurodata')
    spike_frames_builder = SpikeFrameBuilder(frames_src_path)
    spike_frames = spike_frames_builder.build()

    # # 2.2. Convert spike frames to formatted tensors
    dst_single_path = os.path.join(args.path, 'processed', 'single_trial_responses.pt')
    dst_multi_path = os.path.join(args.path, 'processed', 'multi_trial_responses.pt')
    neural_response_builder = NeuralResponseBuilder(spike_frames)
    single_responses = neural_response_builder.build(PVC1Config.SINGLE_TRIAL_REP)
    multi_responses = neural_response_builder.build(PVC1Config.MULTI_TRIAL_REP)
    torch.save(single_responses, dst_single_path)
    torch.save(multi_responses, dst_multi_path)


if __name__ == '__main__':
    main()
