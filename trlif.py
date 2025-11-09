import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from spikingjelly.activation_based import neuron, surrogate

@torch.jit.script
def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

@torch.jit.script
def firing_function(thr, v):
    x = v / thr - (1 / 2) # relative difference argument
    x = torch.clamp(x, 0, 1.0)
    x = round_pass(x)
    x = x * thr # threshold multiplication
    return x     

# TrSG adopted Leaky-Fire-Integrate Neuron
class trLIFNode(neuron.BaseNode):
    def __init__(self, init_tau: float = 2.00, decay_input: bool = True, init_thr: float = 2.0, mode="pqn", *args, **kwargs):

        assert isinstance(init_tau, float) and init_tau > 1.
        super().__init__(*args, **kwargs)
        
        self.decay_input = decay_input
        self.count = 0
        
        init_w = - math.log(init_tau - 1.)
        init_thr = math.log(math.exp(init_thr)-1.0)
        
        self.w = nn.Parameter(torch.as_tensor(init_w))
        self.thr = nn.Parameter(torch.as_tensor(init_thr))
        
        # Running mean parameter for each channel
        self.running_mean = torch.tensor([0.0])

        self.mode=mode

    @property
    def supported_backends(self):
        return ('torch',)

    def extra_repr(self):
        with torch.no_grad():
            tau = 1. / self.w.sigmoid()
            thr = F.softplus(self.thr).item()
            running_mean = self.running_mean.detach().cpu()
        return (f'v_reset={self.v_reset}, decay_input={self.decay_input}, detach_reset={self.detach_reset}, step_mode={self.step_mode}, '
                f'mode={self.mode}, backend={self.backend}, tau={tau}, v_threshold={thr}, '
                f'running_mean={running_mean}')
    
    @staticmethod
    @torch.jit.script
    def jit_soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: torch.Tensor):
        v = v - spike # soft reset
        return v
    
    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_soft_reset_decay_input(x: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                            tau: float):
        v = v + (x - v) / tau
        spike = ((v / v_threshold).floor() * v_threshold).clamp(0, v_threshold)
        v = v - spike
        return spike, v

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_soft_reset_no_decay_input(x: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                               tau: float):
        v = v * (1. - 1. / tau) + x
        spike = ((v / v_threshold).floor() * v_threshold).clamp(0, v_threshold)
        v = v - spike
        return spike, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_decay_input(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                           tau: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] - v) / tau
            spike = ((v / v_threshold).floor() * v_threshold).clamp(0, v_threshold)
            v = v - spike
            spike_seq[t] = spike
        return spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_decay_input_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor,
                                                                      v_threshold: float, tau: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] - v) / tau
            spike = ((v / v_threshold).floor() * v_threshold).clamp(0, v_threshold)
            v = v - spike
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_no_decay_input(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                              tau: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v * (1. - 1. / tau) + x_seq[t]
            spike = ((v / v_threshold).floor() * v_threshold).clamp(0, v_threshold)
            v = v - spike
            spike_seq[t] = spike
        return spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_no_decay_input_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor,
                                                                         v_threshold: float,
                                                                         tau: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v * (1. - 1. / tau) + x_seq[t]
            spike = ((v / v_threshold).floor() * v_threshold).clamp(0, v_threshold)
            v = v - spike
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq

    def neuronal_charge(self, x: torch.Tensor):
        thr = F.softplus(self.thr)

        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v + (x - self.v) * self.w.sigmoid()
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) * self.w.sigmoid()
        else:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v * (1. - self.w.sigmoid()) + x
            else:
                self.v = self.v - (self.v - self.v_reset) * self.w.sigmoid() + x
    
    def neuronal_fire(self):
        thr = F.softplus(self.thr)
        return firing_function(thr, self.v)
    
    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.v_reset is None:
            # soft reset
            self.v = self.jit_soft_reset(self.v, spike_d, F.softplus(self.thr))
        else:
            raise NotImplementedError(self.v_reset)
    
    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.running_mean = self.running_mean.to(x.get_device())
        self.v = self.v.to(x.get_device())
        
        thr = F.softplus(self.thr)

        if self.training:
            if self.count == 0:
                self.v = self.running_mean
            
            result = super().single_step_forward(x)
            
            # To exclude silent neurons, mask only active neurons during a simulation
            if self.count == 0:
                self.mask = (result.clone().detach() > 0)
            else:
                self.mask = torch.logical_or(self.mask, result.clone().detach() > 0)

            self.count += 1
            return result
        else:
            if self.count == 0:       
                self.v = self.running_mean
            self.count += 1
            
            if self.v_reset is None:            
                if self.decay_input:
                    spike, self.v = self.jit_eval_single_step_forward_soft_reset_decay_input(x, self.v, thr, 1. / self.w.sigmoid())
                else:
                    spike, self.v = self.jit_eval_single_step_forward_soft_reset_no_decay_input(x, self.v, thr, 1. / self.w.sigmoid())
            else:
                raise NotImplementedError(self.v_reset)
            return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        self.v_float_to_tensor(x_seq[0])
        self.running_mean = self.running_mean.to(x_seq.get_device())
        self.v = self.v.to(x_seq.get_device())

        thr = F.softplus(self.thr)
        
        if self.training:
            if self.backend == 'torch':
                return super().multi_step_forward(x_seq)
        else:
            if self.v_reset is None:
                self.v = self.running_mean
                if self.decay_input:
                    if self.store_v_seq:
                        spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_soft_reset_decay_input_with_v_seq(
                            x_seq, self.v, thr, 1. / self.w.sigmoid())
                    else:
                        spike_seq, self.v = self.jit_eval_multi_step_forward_soft_reset_decay_input(x_seq, self.v,
                                                                                                    thr,
                                                                                                    1. / self.w.sigmoid())
                else:
                    if self.store_v_seq:
                        spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_soft_reset_no_decay_input_with_v_seq(
                            x_seq, self.v, thr, 1. / self.w.sigmoid())
                    else:
                        spike_seq, self.v = self.jit_eval_multi_step_forward_soft_reset_no_decay_input(x_seq, self.v,
                                                                                                       thr,
                                                                                                       1. / self.w.sigmoid())
            else:
                raise NotImplementedError(self.v_reset)
            return spike_seq

    def update_running_stats(self):
        # Update running mean and variance
        with torch.no_grad():
            # Mask values to include only active neurons
            masked_v = self.v * self.mask.float()
            
            # To avoid dividing by zero in case all values are masked out
            count_non_zero = self.mask.float().sum()
            count_non_zero = torch.clamp(count_non_zero, min=1.0)  # Ensure no division by zero
            
            # Calculate mean of non-negative values
            mean = masked_v.sum() / count_non_zero
            
            # Update running mean
            self.running_mean = self.running_mean * 0.9 + mean * 0.1


    def reset(self):
        # update running mean for every simulation (minibatch) during training
        if self.training and self.count > 0:
            self.update_running_stats()

        super().reset()
        self.count = 0