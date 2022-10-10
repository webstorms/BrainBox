import torch
import numpy as np
import scipy.stats as stats
from sklearn.decomposition import PCA

class RSA:
    def __init__(self, model):
        self.model = model

        self.static_stimuli = []
        self.moving_stimuli = []
        self.neural_responses = []

        self.time_size = 10
        self.warmup_offset = 10

    def get_RSM (self, responses, log=True):
        M_clips = responses.shape[0]
        RSM = np.zeros((M_clips, M_clips))
        
        for idx_a, clip_a in enumerate(responses):
            if idx_a % 100 == 0 and log:
                print('\tStarting loop', idx_a, '/', len(responses))
            for idx_b, clip_b in enumerate(responses):
                RSM[idx_a, idx_b]  = stats.pearsonr(clip_a, clip_b)[0]
                
        return RSM

    def get_RSM_ceiling (self, responses, iterations=100):
        noise_ceiling = []
        
        for i in range(iterations):
            if i % 10 == 0:
                print('Iteration', i)
            
            n_units = responses.shape[1]
            shuffle_arr = np.random.permutation(n_units)
            shuffle_a, shuffle_b = shuffle_arr[:n_units//2], shuffle_arr[n_units//2:]
            
            RSM_a = self.get_RSM(responses[:, shuffle_a], log=False)
            RSM_b = self.get_RSM(responses[:, shuffle_b], log=False)
            
            noise_ceiling.append(self.get_tau(RSM_a, RSM_b))
            
        return np.mean(noise_ceiling), noise_ceiling

    def get_tau (self, RSM_a, RSM_b):
        RSM_a_flat = RSM_a[np.triu_indices_from(RSM_a, k = 1)]
        RSM_b_flat = RSM_b[np.triu_indices_from(RSM_b, k = 1)]
        
        return stats.kendalltau(RSM_a_flat, RSM_b_flat)[0]

    def get_RSM_similarity (self, pca, responses, neural_RSM):    
        dim_reduc = pca.transform(responses)
        
        RSM = self.get_RSM(responses)
        RSM_similarity = self.get_tau(RSM, neural_RSM)
        
        return RSM_similarity

    def compare_model (self):
        neural_RSM = get_RSM(self.neural_responses)

        static_responses = self.model(self.static_stimuli)
        moving_responses = self.model(self.static_stimuli)

        pca = PCA(n_components=100)
        pca.fit(static_responses)

        return self.get_RSM_similarity(pca, moving_responses, neural_RSM)