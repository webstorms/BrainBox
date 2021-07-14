import torch
import torch.nn as nn
import torch.nn.functional as F

from ._spike_functions import FastSigmoid


class LIFNeurons(nn.Module):

    RESET_TYPE_HARD = 'HARD'
    RESET_TYPE_SOFT = 'SOFT'

    def __init__(self, n_in, n_out, beta_syn_init=[0.9], beta_mem_init=[0.9], beta_syn_range=[0, 1], beta_mem_range=[0, 1], beta_syn_diff=False, beta_mem_diff=False, reset_type='SOFT', spike_func=FastSigmoid.get(100), bias=True):
        super(LIFNeurons, self).__init__()

        def validate_beta_values(beta_vals, beta_range):
            assert type(beta_vals) == list, 'Provided decay constants need to be of type list'
            for v in beta_vals:
                assert beta_range[0] <= v <= beta_range[1], 'Ensure that decay constants are in range.'

        def validate_beta_len(vals):
            assert len(vals) in [1, n_out], 'Ensure that the list of provided decay constants are either of length 1 or n_out'

        def validate_beta_range(vals):
            assert len(vals) == 2, 'Ensure to provide two values for the decay constant range'

        validate_beta_range(beta_syn_range)
        validate_beta_range(beta_mem_range)
        validate_beta_values(beta_syn_init, beta_syn_range)
        validate_beta_values(beta_mem_init, beta_mem_range)
        validate_beta_len(beta_syn_init)
        validate_beta_len(beta_mem_init)

        self.n_in = n_in
        self.n_out = n_out
        self.beta_syn_init = beta_syn_init
        self.beta_mem_init = beta_mem_init
        self.beta_syn_range = beta_syn_range
        self.beta_mem_range = beta_mem_range
        self.beta_syn_diff = beta_syn_diff
        self.beta_mem_diff = beta_mem_diff
        self.reset_type = reset_type
        self.spike_func = spike_func
        self.bias = bias

        self._beta_syn = nn.Parameter(data=torch.Tensor(n_out * beta_syn_init) if len(beta_syn_init) == 1 else torch.Tensor(beta_syn_init), requires_grad=beta_syn_diff)
        self._beta_mem = nn.Parameter(data=torch.Tensor(n_out * beta_mem_init) if len(beta_mem_init) == 1 else torch.Tensor(beta_mem_init), requires_grad=beta_mem_diff)

    @property
    def beta_syn(self):
        return torch.clamp(self._beta_syn, min=self.beta_syn_range[0], max=self.beta_syn_range[1])

    @property
    def beta_mem(self):
        return torch.clamp(self._beta_mem, min=self.beta_mem_range[0], max=self.beta_mem_range[1])

    def compute_input_current(self, pre_spikes, inject_current, post_spikes):
        raise NotImplementedError

    def forward(self, pre_spikes=None, inject_current=None, return_all=False):
        # pre_spikes: n_batch x n_in x n_timesteps x ...

        # Initialise placeholders for the synaptic current and membrane voltage
        with torch.no_grad():
            t_len = pre_spikes.shape[2] if pre_spikes is not None else inject_current.shape[2]
            device = pre_spikes.device if pre_spikes is not None else inject_current.device
            dtype = pre_spikes.dtype if pre_spikes is not None else inject_current.dtype

            pre_spikes_at_t0 = pre_spikes[:, :, 0] if pre_spikes is not None else None
            inject_current_at_t0 = inject_current[:, :, 0] if inject_current is not None else None
            input_current_shape = self.compute_input_current(pre_spikes_at_t0, inject_current_at_t0, []).shape

            # Initial current and membrane potential
            syn = torch.zeros(input_current_shape, device=device, dtype=dtype)
            mem = torch.zeros(input_current_shape, device=device, dtype=dtype)

        # Keeping track of the neurons states
        mem_list = []
        post_spikes_list = []

        for t in range(t_len):

            pre_spikes_at_t = pre_spikes[:, :, t] if pre_spikes is not None else None
            inject_current_at_t = inject_current[:, :, t] if inject_current is not None else None
            input_current = self.compute_input_current(pre_spikes_at_t, inject_current_at_t, post_spikes_list)

            # 1. Update synaptic conductance
            syn = torch.einsum('bn...,n->bn...', syn, self.beta_syn) + input_current

            # 2. Update membrane potential
            mem = (torch.einsum('bn...,n->bn...', mem, self.beta_mem) + torch.einsum('bn...,n->bn...', syn, 1 - self.beta_mem))

            # 3. To spike or not to spike
            post_spike = self.spike_func(mem - 1)
            mem_list.append(mem.clone())
            post_spikes_list.append(post_spike)

            # 4. Reset membrane potential for spiked neurons
            with torch.no_grad():
                if self.reset_type == 'HARD':
                    mem *= (1 - post_spike)
                elif self.reset_type == 'SOFT':
                    mem -= post_spike

        if return_all:
            # Return spikes and membrane potential
            return torch.stack(post_spikes_list, dim=2), torch.stack(mem_list, dim=2)
        else:
            # batch x n_out x t_len x ...
            return torch.stack(post_spikes_list, dim=2)


class ConvLIFNeurons(LIFNeurons):

    def __init__(self, n_in, n_out, kh, kw, stride, beta_syn_init=[0.9], beta_mem_init=[0.9], beta_syn_range=[0, 1], beta_mem_range=[0, 1], beta_syn_diff=False, beta_mem_diff=False, reset_type='SOFT', spike_func=FastSigmoid.get(100), bias=True):
        super().__init__(n_in, n_out, beta_syn_init, beta_mem_init, beta_syn_range, beta_mem_range, beta_syn_diff, beta_mem_diff, reset_type, spike_func, bias)
        self.kh = kh
        self.kw = kw
        self.stride = stride

        self.spikes_to_current = nn.Conv2d(n_in, n_out, (kh, kw), stride, bias=bias)

    def compute_input_current(self, pre_spikes, inject_current, post_spikes):
        # pre_spikes: batch x n_in x h x w
        if pre_spikes is not None:
            return self.spikes_to_current(pre_spikes)
        else:
            return inject_current


class RecConvLIFNeurons(LIFNeurons):

    def __init__(self, n_in, n_out, kh, kw, rf_stride, rec_spatial_size, beta_syn_init=[0.9], beta_mem_init=[0.9], beta_syn_range=[0, 1], beta_mem_range=[0, 1], beta_syn_diff=False, beta_mem_diff=False, reset_type='SOFT', spike_func=FastSigmoid.get(100), bias=True):
        super().__init__(n_in, n_out, beta_syn_init, beta_mem_init, beta_syn_range, beta_mem_range, beta_syn_diff, beta_mem_diff, reset_type, spike_func, bias)
        self.kh = kh
        self.kw = kw
        self.rf_stride = rf_stride
        self.rec_spatial_size = rec_spatial_size

        self.pre_spikes_to_current = nn.Conv2d(n_in, n_out, (kh, kw), rf_stride, bias=bias)
        self.post_spikes_to_current = nn.Conv2d(n_out, n_out, rec_spatial_size, padding=rec_spatial_size//2, bias=False)

    def compute_input_current(self, pre_spikes, inject_current, post_spikes):
        # pre_spikes: batch x n_in x h x w
        if pre_spikes is not None:
            current = self.pre_spikes_to_current(pre_spikes)
        else:
            current = inject_current

        if len(post_spikes) > 0:
            current += self.post_spikes_to_current(post_spikes[-1])

        return current


class DaleRecConvLIFNeurons(LIFNeurons):

    def __init__(self, n_in, n_out, kh, kw, rf_stride, rec_spatial_size, frac_inhibitory=0.1, beta_syn_init=[0.9], beta_mem_init=[0.9], beta_syn_range=[0, 1], beta_mem_range=[0, 1], beta_syn_diff=False, beta_mem_diff=False, reset_type='SOFT', spike_func=FastSigmoid.get(100), bias=True):
        super().__init__(n_in, n_out, beta_syn_init, beta_mem_init, beta_syn_range, beta_mem_range, beta_syn_diff, beta_mem_diff, reset_type, spike_func, bias)
        assert 0 <= frac_inhibitory <= 1, 'Fraction of inhibitory neuron needs to be in range [0,1].'
        self.kh = kh
        self.kw = kw
        self.rf_stride = rf_stride
        self.rec_spatial_size = rec_spatial_size
        self.frac_inhibitory = frac_inhibitory

        self.pre_spikes_to_current = nn.Conv2d(n_in, n_out, (kh, kw), rf_stride, bias=bias)
        self._recurrent_connectivity = nn.Parameter(nn.Conv2d(n_out, n_out, rec_spatial_size).weight, requires_grad=True)

        # Set all units as excitatory except the first frac_inhibitory * n_out
        self._excitatory_inhibitory_mask = nn.Parameter(torch.ones(self._recurrent_connectivity.shape), requires_grad=False)
        n_inhibitory_neurons = int(frac_inhibitory * n_out)
        self._excitatory_inhibitory_mask[:, :n_inhibitory_neurons] = -1

    @property
    def recurrent_connectivity(self):
        return self._excitatory_inhibitory_mask * torch.abs(self._recurrent_connectivity)

    def compute_input_current(self, pre_spikes, inject_current, post_spikes):
        # pre_spikes: batch x n_in x t_len x h x w
        if pre_spikes is not None:
            current = self.pre_spikes_to_current(pre_spikes)
        else:
            current = inject_current

        if len(post_spikes) > 0:
            current += F.conv2d(post_spikes[-1], self.recurrent_connectivity, padding=self.rec_spatial_size//2)

        return current

