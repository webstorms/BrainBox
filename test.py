import brainbox.physiology as phys

import sys
sys.path.append('../rnn')
from lib.network import RecurrentTemporalPrediction

import math
from matplotlib import pyplot as plt
import numpy as np
import torch
from scipy import ndimage
import pandas as pd


model_, hyperparameters, loss_history = RecurrentTemporalPrediction.load(
    model_path='../rnn/models/L1-6.5_beta-0.1_Nov14-16-27/1250-epochs_model.pt',
    device='cpu',
    plot_loss_history=False,
    plot_loglog=False
)

def model (x_):
	x = x_.reshape((1, 10, -1))
	_, y = model_(x)

	return y[:, :, 200]

gratingsProber = phys.GratingsProber(model, 0.5, 20, 20, 10, 1)
#gratingsProber.probe(1)
gratingsProber.grating_results = pd.read_pickle('./gratings.pickle')
#gratingsProber.grating_results.to_pickle('./gratings.pickle')

#print('orientation bw', gratingsProber.orientation_bandwidth)
print('osi', gratingsProber.orientation_selectivity_index)
print('dsi', gratingsProber.direction_selectivity_index)
print('cv', gratingsProber.circular_variance)