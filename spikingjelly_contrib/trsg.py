"""Threshold-robust Surrogate Gradient (TrSG).

Reference implementation prepared for upstreaming into
``spikingjelly.activation_based.surrogate``.

TrSG (Kook et al., WACV 2026, arXiv:2511.08708) makes surrogate gradients
robust to the *scale* of a (possibly learnable) firing threshold ``V_thr``.
It is a thin wrapper that turns **any** existing SpikingJelly surrogate shape
(``Rect``, ``ATan``, ``Sigmoid``, ``PiecewiseQuadratic``, ...) into a
threshold-robust one, so it deliberately reuses their ``primitive_function``
(the relaxed spike probability ``Phi``) and ``backward`` (the window
``phi = Phi'``) instead of re-deriving per-shape gradients.

Given the membrane potential ``M`` and threshold ``V_thr``, TrSG uses:

* relative-scale surrogate argument   ``x = M / V_thr - 1``
* forward output                      ``O = V_thr * H(M - V_thr)``          (Eq. 10)

so that (Eq. 11)::

    dO/dM = V_thr * (1 / V_thr) * phi(x) = phi(x)

The ``V_thr`` on the forward path cancels the ``1 / V_thr`` factor that a
relative-scale argument introduces in the backward pass, making the gradient
*magnitude* threshold-invariant while the active window still scales with
``V_thr``. Baselines fail on one side or the other:

* AS-SG (absolute scale, ``x = M - V_thr``): gradient flood (small ``V_thr``)
  / starvation (large ``V_thr``) -- window width does not track ``V_thr``.
* RS-SG (relative scale, no output rescale): gradient explosion (small
  ``V_thr``) / vanishing (large ``V_thr``) -- the ``1 / V_thr`` magnitude
  factor is left uncancelled.

At inference the ``V_thr`` factor is a fixed per-layer scalar and folds into
the next layer's weights, so deployment spikes stay binary and unchanged.
"""

import torch
import torch.nn as nn

from spikingjelly.activation_based import surrogate


def heaviside(x: torch.Tensor) -> torch.Tensor:
    return (x >= 0.0).to(x)


class trsg_function(torch.autograd.Function):
    """Autograd bridge for TrSG.

    Forward emits the threshold-scaled hard spike ``O = V_thr * S``, with the
    hard spike ``S = H(M - V_thr)``. Backward reuses the wrapped surrogate's
    ``phi`` (``backward``) evaluated at the relative argument
    ``x = M / V_thr - 1``:

        grad_M     = grad_output * phi(x)                    (threshold-invariant)
        grad_Vthr  = grad_output * S - grad_M * (x + 1)

    ``x + 1 = M / V_thr``. The ``grad_Vthr`` expression is the exact autograd of
    the straight-through realization ``O = V_thr * S_ste`` where
    ``S_ste = (S - Phi(x)).detach() + Phi(x)``: the explicit ``V_thr``
    multiplier contributes the hard spike value ``S`` (product rule), while
    ``V_thr``'s appearance inside ``x`` contributes ``-grad_M * (x + 1)``. This
    matches the released ``firing_function`` (round/clamp STE) exactly for the
    rectangular shape. Nothing on the threshold path is detached, so a genuine
    learning signal reaches ``V_thr``.
    """

    @staticmethod
    def forward(m, v_threshold, primitive_function, backward_function, sg_params):
        return v_threshold * heaviside(m - v_threshold)

    @staticmethod
    def setup_context(ctx, inputs, output):
        m, v_threshold, primitive_function, backward_function, sg_params = inputs
        ctx.save_for_backward(m, v_threshold)
        ctx.primitive_function = primitive_function
        ctx.backward_function = backward_function
        ctx.sg_params = sg_params

    @staticmethod
    def backward(ctx, grad_output):
        m, v_threshold = ctx.saved_tensors
        x = m / v_threshold - 1.0
        # grad_M = grad_output * phi(x); reuse the wrapped surrogate's backward,
        # which already returns grad_output * phi(x).
        grad_m = ctx.backward_function(grad_output, x, **ctx.sg_params)
        grad_v_threshold = None
        if v_threshold.requires_grad:
            spike = heaviside(m - v_threshold)  # S, the hard forward spike
            grad_vth = grad_output * spike - grad_m * (x + 1.0)
            # Reduce to v_threshold's shape (typically a 0-dim learnable scalar).
            grad_v_threshold = _sum_to(grad_vth, v_threshold.shape)
        return grad_m, grad_v_threshold, None, None, None


def _sum_to(grad: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Sum ``grad`` down to ``shape`` (inverse of broadcasting)."""
    if shape == grad.shape:
        return grad
    if len(shape) == 0:
        return grad.sum()
    # Sum extra leading dims, then any broadcast (size-1) dims.
    while grad.dim() > len(shape):
        grad = grad.sum(0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(i, keepdim=True)
    return grad.reshape(shape)


class TrSG(nn.Module):
    r"""Threshold-robust Surrogate Gradient wrapper.

    Wraps an existing SpikingJelly surrogate shape and makes it robust to the
    scale of ``V_thr``. Unlike ordinary surrogates (which the neuron calls as
    ``surrogate_function(v - v_threshold)`` and which return a ``{0, 1}``
    spike), a ``TrSG`` instance is threshold-aware and is called as
    ``trsg(v, v_threshold)``, returning the scaled spike ``V_thr * S``.

    ``v_threshold`` may be a learnable tensor or a fixed value: the backward
    only produces a threshold gradient when ``v_threshold.requires_grad`` is
    ``True``, so the same surrogate serves trained and fixed thresholds without
    changes.

    :param surrogate_function: the base surrogate shape supplying ``Phi``
        (``primitive_function``) and ``phi`` (``backward``). Defaults to
        :class:`spikingjelly.activation_based.surrogate.Rect` with
        ``alpha=1.0``, which reproduces the rectangular round/clamp STE of the
        original TrSG paper exactly.
    :type surrogate_function: surrogate.SurrogateFunctionBase

    Example::

        from spikingjelly.activation_based import surrogate
        trsg = TrSG(surrogate.ATan(alpha=2.0))
        spike = trsg(v, v_threshold)   # spike in {0, V_thr}
    """

    def __init__(self, surrogate_function: surrogate.SurrogateFunctionBase = None):
        super().__init__()
        if surrogate_function is None:
            surrogate_function = surrogate.Rect(alpha=1.0)
        self.surrogate_function = surrogate_function

    def forward(self, v: torch.Tensor, v_threshold: torch.Tensor) -> torch.Tensor:
        if not isinstance(v_threshold, torch.Tensor):
            v_threshold = torch.as_tensor(v_threshold, dtype=v.dtype, device=v.device)
        sg = self.surrogate_function
        return trsg_function.apply(
            v,
            v_threshold,
            sg.primitive_function,
            sg.backward,
            sg._sg_params,
        )

    def extra_repr(self) -> str:
        return f"surrogate_function={self.surrogate_function}"
