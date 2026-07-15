"""TrLIF neuron: LIF with jointly-learnable threshold + tau, TrSG firing, MP-Init.

Reference (torch-backend) implementation prepared for upstreaming into
``spikingjelly.activation_based.neuron``. This is the pure-PyTorch path that a
CUDA/Triton backend would have to match numerically; see the package README for
the FlexSN discussion and the two-PR split.

Combines three components from Kook et al. (WACV 2026, arXiv:2511.08708):

* **Learnable tau** ``tau = 1 / sigmoid(w)`` (as in :class:`ParametricLIFNode`).
* **Learnable threshold** ``V_thr = softplus(thr_param)`` (positivity-constrained).
* **TrSG firing** via :class:`spikingjelly_contrib.trsg.TrSG` (threshold-robust
  surrogate gradient); the neuron passes ``(v, V_thr)`` so the firing output is
  the scaled spike ``V_thr * S``.
* **MP-Init** (Membrane Potential Initialization): each layer's membrane
  potential is initialized at ``t = 0`` from an EMA running mean of the
  last-timestep membrane potential over *active* neurons, snapping the state
  Markov chain to its stationary mean and suppressing Temporal Covariate Shift.

Only soft reset (``v_reset=None``) is supported, matching the paper. Because the
firing output is already scaled by ``V_thr``, the soft reset subtracts the
scaled spike directly (``v = v - spike``), not ``v - V_thr * spike``.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based.neuron import BaseNode

from .trsg import TrSG


class TrLIFNode(BaseNode):
    r"""LIF neuron with joint learnable threshold/tau, TrSG, and MP-Init.

    :param init_tau: initial membrane time constant (``> 1``); stored as
        ``w = -log(init_tau - 1)`` so ``tau = 1 / sigmoid(w)``.
    :type init_tau: float
    :param init_v_threshold: initial firing threshold (``> 0``); stored as
        ``thr_param = log(exp(init_v_threshold) - 1)`` so
        ``V_thr = softplus(thr_param)``.
    :type init_v_threshold: float
    :param decay_input: whether the input current is scaled by ``1/tau`` when
        charging the membrane potential.
    :type decay_input: bool
    :param learnable_v_threshold: whether the threshold ``V_thr`` is trained. If
        ``False`` it is a fixed buffer (still used for the relative-scale
        argument and the forward rescale, but receives no gradient). With
        ``learnable_v_threshold=False`` and ``init_v_threshold=1.0`` TrSG
        reduces exactly to a standard fixed-threshold surrogate.
    :type learnable_v_threshold: bool
    :param learnable_tau: whether ``tau`` is trained. Independent of
        ``learnable_v_threshold`` -- all four train/fix combinations are
        supported (joint threshold+tau training is the paper's default).
    :type learnable_tau: bool
    :param mp_init: whether to enable Membrane Potential Initialization.
    :type mp_init: bool
    :param mp_init_momentum: EMA update weight for the new batch mean (retention
        is ``1 - mp_init_momentum``); paper default ``0.1``.
    :type mp_init_momentum: float
    :param surrogate_function: base surrogate shape wrapped by TrSG; defaults to
        :class:`spikingjelly.activation_based.surrogate.Rect` (alpha=1),
        reproducing the paper's rectangular round/clamp STE exactly.
    """

    def __init__(
        self,
        init_tau: float = 2.0,
        init_v_threshold: float = 1.0,
        decay_input: bool = True,
        learnable_v_threshold: bool = True,
        learnable_tau: bool = True,
        mp_init: bool = True,
        mp_init_momentum: float = 0.1,
        surrogate_function=None,
        detach_reset: bool = False,
        step_mode: str = "s",
        backend: str = "torch",
        store_v_seq: bool = False,
    ):
        assert isinstance(init_tau, float) and init_tau > 1.0
        assert isinstance(init_v_threshold, float) and init_v_threshold > 0.0
        assert 0.0 < mp_init_momentum <= 1.0

        trsg = TrSG(surrogate_function)
        # Pass a plain float threshold up to BaseNode (it asserts float); the
        # real, learnable threshold lives in self.thr_param. v_reset=None ->
        # soft reset.
        super().__init__(
            v_threshold=float(init_v_threshold),
            v_reset=None,
            surrogate_function=trsg,
            detach_reset=detach_reset,
            step_mode=step_mode,
            backend=backend,
            store_v_seq=store_v_seq,
        )

        self.decay_input = decay_input
        self.mp_init = mp_init
        self.mp_init_momentum = mp_init_momentum

        # Learnable tau: w with tau = 1 / sigmoid(w) (PLIF convention). A
        # non-trainable Parameter (requires_grad=False) is skipped by the
        # optimizer and excluded from the DDP reducer.
        init_w = -math.log(init_tau - 1.0)
        self.w = nn.Parameter(torch.as_tensor(init_w), requires_grad=learnable_tau)

        # Learnable threshold: V_thr = softplus(thr_param). Parameter when
        # trained, buffer when fixed (so V_thr still folds/broadcasts and is
        # saved in state_dict, but accrues no gradient).
        init_thr = math.log(math.exp(init_v_threshold) - 1.0)  # softplus^{-1}
        if learnable_v_threshold:
            self.thr_param = nn.Parameter(torch.as_tensor(init_thr))
        else:
            self.register_buffer("thr_param", torch.as_tensor(init_thr))

        # Persistent MP-Init running mean (per-layer scalar). A buffer so it is
        # saved in state_dict and follows .to()/.cuda(); DDP broadcasts it like
        # a BN running stat.
        self.register_buffer("running_mean", torch.zeros(1))

        # Per-simulation bookkeeping (plain attributes; untouched by reset()).
        self.count = 0
        self.mask = None

    @property
    def supported_backends(self):
        return ("torch",)

    def v_threshold_value(self) -> torch.Tensor:
        return F.softplus(self.thr_param)

    def tau_value(self) -> torch.Tensor:
        return 1.0 / self.w.sigmoid()

    def extra_repr(self):
        with torch.no_grad():
            tau = self.tau_value().item()
            thr = self.v_threshold_value().item()
            rm = self.running_mean.detach().flatten()[0].item()
        return (
            f"decay_input={self.decay_input}, detach_reset={self.detach_reset}, "
            f"step_mode={self.step_mode}, backend={self.backend}, "
            f"learnable_v_threshold={isinstance(self.thr_param, nn.Parameter)}, "
            f"learnable_tau={self.w.requires_grad}, mp_init={self.mp_init}, "
            f"tau={tau:.4f}, v_threshold={thr:.4f}, running_mean={rm:.4f}"
        )

    def neuronal_charge(self, x: torch.Tensor):
        decay = self.w.sigmoid()  # = 1 / tau
        if self.decay_input:
            self.v = self.v + (x - self.v) * decay
        else:
            self.v = self.v * (1.0 - decay) + x

    def neuronal_fire(self):
        # TrSG surrogate is threshold-aware: returns the scaled spike V_thr * S.
        return self.surrogate_function(self.v, self.v_threshold_value())

    def neuronal_reset(self, spike):
        # Soft reset. `spike` is already scaled by V_thr, so subtract it
        # directly (equivalent to v - V_thr * S).
        spike_d = spike.detach() if self.detach_reset else spike
        self.v = self.v - spike_d

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)

        if self.mp_init and self.count == 0:
            # Seed the membrane potential from the persistent running mean.
            self.v = self.v + self.running_mean.to(x)

        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)

        # Accumulate the active-neuron mask (fired at least once this run).
        active = spike.detach() > 0
        if self.count == 0:
            self.mask = active
        else:
            self.mask = self.mask | active
        self.count += 1
        return spike

    @torch.no_grad()
    def update_running_stats(self):
        if self.mask is None or not isinstance(self.v, torch.Tensor):
            return
        masked_v = self.v * self.mask.to(self.v.dtype)
        count_active = self.mask.to(self.v.dtype).sum().clamp_min(1.0)
        mean = masked_v.sum() / count_active
        m = self.mp_init_momentum
        self.running_mean = self.running_mean * (1.0 - m) + mean * m

    def reset(self):
        # Update MP-Init running mean from this simulation before wiping state.
        if self.training and self.mp_init and self.count > 0:
            self.update_running_stats()
        super().reset()
        self.count = 0
        self.mask = None
