import numpy as np
import scipy.stats as stats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class RSA:
    def __init__(
        self, model, dimension_reduction_stimuli, stimuli, neural_responses,
        logging=False, stimuli_correlation_method='pearson', RSM_correlation_method='tau_b'
    ):
        """
        model                          callable    Takes stimuli as input and returns
                                                       output as stimuli x unit matrix
        dimension_reduction_stimuli    any         Stimuli used for dimension reduction of model outputs
        stimuli                        any         Stimuli used to produce RSMs across models and neural data
        neural_response                ndarray     Response matrix of form stimuli x neuron
        logging                        bool        Should functions log their output
        stimuli_correlation_method     str         Correlation method between each stimulus vector (pearson,
                                                       spearman, tau_b, tau_c)  
        RSM_correlation_method         str         Correlation method between each flattened RSM (pearson,
                                                       spearman, tau_b, tau_c)
        """
        
        self.model = model
        self.dimension_reduction_stimuli = dimension_reduction_stimuli
        self.stimuli = stimuli
        self.neural_responses = neural_responses
        
        self.logging = logging
        self.stimuli_correlation_method = stimuli_correlation_method
        self.RSM_correlation_method = RSM_correlation_method
        
        self.noise_ceiling = None
    
    # Returns correlation between two vector using given method
    @staticmethod
    def get_correlation (a, b, method):
        """
        a         ndarray    
        b         ndarray
        method    str        pearson, spearman, tau_b, tau_c
        """

        if method == 'pearson':
            return stats.pearsonr(a, b)[0]
        elif method == 'spearman':
            return stats.spearmanr(a, b)[0]
        elif method == 'tau_b':
            return stats.kendalltau(a, b, variant='b')[0]
        elif method == 'tau_c':
            return stats.kendalltau(a, b, variant='c')[0]
        else:
            raise Exception('Correlation method not implemented.')
           
    # Plots the lower triangle of a correlation matrix
    @staticmethod
    def plot_correlation_matrix (correlation_matrix, labels):
        xy = correlation_matrix.shape[0]
        mask =  np.tri(xy, k=0).astype(bool)
        lower_triangle = np.ma.array(correlation_matrix, mask=np.invert(mask)) 

        fig, ax = plt.subplots(dpi=100)
        cmap = cm.get_cmap('viridis', 1000)
        cmap.set_bad('w')

        im = plt.imshow(lower_triangle, cmap=cmap)
        plt.xticks(np.arange(xy), labels)
        plt.yticks(np.arange(xy), labels)
        plt.colorbar(im, label='Response similarity')
        plt.show()

    # Returns the RSM for a set of responses or RSA object with optional dimensional reduction
    def get_RSM (self, responses, dimension_reduction=None):
        """
        responses    RSA|ndarray    RSA object | stimulus x unit response matrix
        """
        
        if type(responses) == RSA:
            rsa = responses
            dimension_reduction_responses = rsa.model(rsa.dimension_reduction_stimuli)
            responses = rsa.model(rsa.stimuli)

            if dimension_reduction:
                n_components = 100 if type(dimension_reduction)==bool else dimension_reduction
                pca = PCA(n_components=n_components)
                pca.fit(dimension_reduction_responses)
                responses = pca.transform(responses)
            
        M_clips = responses.shape[0]
        RSM = np.zeros((M_clips, M_clips))
        
        for idx_a, clip_a in enumerate(responses):
            if idx_a % 100 == 0 and self.logging:
                print('\tStarting RSM loop', idx_a, '/', len(responses))
            for idx_b, clip_b in enumerate(responses):
                RSM[idx_a, idx_b]  = self.get_correlation(clip_a, clip_b, self.stimuli_correlation_method)
                
        return RSM

    # Returns the RSM ceiling for an RSA object's neural_responses
    def get_RSM_ceiling (self, iterations):
        """
        iterations    int    Number of shuffles over which to estimate noise ceiling
        """
        
        if self.noise_ceiling is not None:
            return self.noise_ceiling
        
        noise_ceiling = []
        
        for i in range(iterations):
            if i % 10 == 0 and self.logging:
                print('RSM ceiling iteration', i)
            
            n_units = self.neural_responses.shape[1]
            shuffle_arr = np.random.permutation(n_units)
            shuffle_a, shuffle_b = shuffle_arr[:n_units//2], shuffle_arr[n_units//2:]
            
            RSM_a = self.get_RSM(self.neural_responses[:, shuffle_a])
            RSM_b = self.get_RSM(self.neural_responses[:, shuffle_b])
            
            noise_ceiling.append(self.get_RSM_correlation(RSM_a, RSM_b))
            
        self.noise_ceiling = noise_ceiling
            
        return np.mean(self.noise_ceiling), self.noise_ceiling

    # Returns the correlation between two RSMs
    def get_RSM_correlation (self, RSM_a, RSM_b):
        """
        RSM_a    ndarray    stimulus x stimulus correlation matrix
        RSM_b    ndarray    stimulus x stimulus correlation matrix
        """
        
        RSM_a_flat = RSM_a[np.triu_indices_from(RSM_a, k = 1)]
        RSM_b_flat = RSM_b[np.triu_indices_from(RSM_b, k = 1)]
        
        return self.get_correlation(RSM_a_flat, RSM_b_flat, self.RSM_correlation_method)

    # Returns the response similarity between the model and neural responses (as a scalar)
    #     or a response correlation matrix between each pair of models
    def compare_model (self, models=None, dimension_reduction=False, noise_corrected=False):
        """
        models                 [RSA]       List of RSA model instances or neural responses matrix
        dimension_reduction    bool|int    Should use dimension reduction on model outputs |
                                                Number of dimensions to use
        noise_corrected        bool        Should noise correct similarity measure | Number
                                               of iterations for noise ceiling estimation
        """

        if models is None:
            models = [self.neural_responses]
        models.insert(0, self)        
        model_responses = np.zeros((len(models), len(models)))
        
        for model_a_idx, model_a in enumerate(models):
            RSM_a = self.get_RSM(model_a, dimension_reduction)
                
            for model_b_idx, model_b in enumerate(models):
                RSM_b = self.get_RSM(model_b, dimension_reduction)
                
                model_responses[model_a_idx, model_b_idx] = self.get_RSM_correlation(RSM_a, RSM_b)
                
        if noise_corrected:
            if self.noise_ceiling is None:
                iterations = 100 if type(noise_corrected)==bool else noise_corrected
                self.get_RSM_ceiling(iterations)

            model_responses = model_responses/np.mean(self.noise_ceiling)
    
        if len(models) == 2:
            return model_responses[0, 1]
        else:
            return model_responses