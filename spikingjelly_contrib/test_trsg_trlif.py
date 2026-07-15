"""CPU verification for the TrSG surrogate and TrLIF neuron prototypes.

Run directly (``python -m spikingjelly_contrib.test_trsg_trlif``) or with pytest.
All tests run on CPU with the ``torch`` backend. They check:

1. TrSG(Rect, alpha=1) reproduces the original ``trlif.py`` ``firing_function``
   exactly (forward + gradient w.r.t. v and V_thr).
2. TrSG for arbitrary shapes equals the autograd of the principled STE
   relaxation ``V_thr * Phi(v/V_thr - 1)`` (hard forward, soft backward).
3. The TrLIFNode prototype matches the reference ``trLIFNode`` from ``trlif.py``
   (forward spikes and gradients to input / tau / threshold).
4. MP-Init EMA update matches the reference math across resets.
5. Threshold-invariance: TrSG keeps the input-gradient magnitude ~constant as
   V_thr varies, where AS-SG and RS-SG do not.
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spikingjelly.activation_based import surrogate, functional

from spikingjelly_contrib.trsg import TrSG, heaviside
from spikingjelly_contrib.trlif_node import TrLIFNode

torch.manual_seed(0)


# --- reference from the released single-file implementation --------------------
def round_pass(x):
    y = x.round()
    return (y - x).detach() + x


def ref_firing_function(thr, v):
    x = v / thr - 0.5
    x = torch.clamp(x, 0.0, 1.0)
    x = round_pass(x)
    x = x * thr
    return x


def trsg_ste_reference(v, thr, sg):
    """Principled STE realization: O = V_thr * S_ste, S_ste = STE(H, Phi)."""
    x = v / thr - 1.0
    phi_primitive = sg.primitive_function(x, **sg._sg_params)
    spike = heaviside(v - thr)
    s_ste = (spike - phi_primitive).detach() + phi_primitive
    return thr * s_ste


def test_trsg_rect_matches_reference_firing_function():
    v = torch.randn(2000, dtype=torch.double) * 2.0
    thr = torch.tensor(1.3, dtype=torch.double)

    v1 = v.clone().requires_grad_(True)
    thr1 = thr.clone().requires_grad_(True)
    out_ref = ref_firing_function(thr1, v1)
    g = torch.randn_like(out_ref)
    (out_ref * g).sum().backward()

    v2 = v.clone().requires_grad_(True)
    thr2 = thr.clone().requires_grad_(True)
    trsg = TrSG(surrogate.Rect(alpha=1.0))
    out_mine = trsg(v2, thr2)
    (out_mine * g).sum().backward()

    assert torch.allclose(out_ref, out_mine), "forward mismatch"
    assert torch.allclose(v1.grad, v2.grad, atol=1e-9), "grad_v mismatch"
    assert torch.allclose(thr1.grad, thr2.grad, atol=1e-9), "grad_thr mismatch"
    print("[1] TrSG(Rect) == trlif.py firing_function  (fwd, grad_v, grad_thr)  OK")


def test_trsg_generic_matches_ste_relaxation():
    shapes = {
        "Rect": surrogate.Rect(alpha=1.0),
        "ATan": surrogate.ATan(alpha=2.0),
        "Sigmoid": surrogate.Sigmoid(alpha=4.0),
        "PiecewiseQuadratic": surrogate.PiecewiseQuadratic(alpha=1.0),
    }
    for name, sg in shapes.items():
        v = torch.randn(5000, dtype=torch.double) * 1.5
        thr = torch.tensor(0.7, dtype=torch.double)
        g = torch.randn_like(v)

        v1 = v.clone().requires_grad_(True)
        thr1 = thr.clone().requires_grad_(True)
        out_ref = trsg_ste_reference(v1, thr1, sg)
        (out_ref * g).sum().backward()

        v2 = v.clone().requires_grad_(True)
        thr2 = thr.clone().requires_grad_(True)
        out_mine = TrSG(sg)(v2, thr2)
        (out_mine * g).sum().backward()

        assert torch.allclose(out_ref, out_mine), f"{name} fwd"
        assert torch.allclose(v1.grad, v2.grad, atol=1e-9), f"{name} grad_v"
        assert torch.allclose(thr1.grad, thr2.grad, atol=1e-9), f"{name} grad_thr"
    print("[2] TrSG(shape) == STE relaxation autograd for", list(shapes), " OK")


def _ref_trlif_forward(x_seq, w, thr_param, decay_input=True):
    """Faithful CPU replica of trlif.py's training dynamics (running_mean=0).

    Mirrors trLIFNode.single_step_forward's training path: PLIF-style charge,
    firing_function (round/clamp STE * thr), then soft reset v = v - spike.
    (trlif.py itself is GPU-only via .get_device(), so we replicate on CPU.)
    """
    import torch.nn.functional as F

    T = x_seq.shape[0]
    decay = torch.sigmoid(w)  # 1 / tau
    thr = F.softplus(thr_param)
    v = torch.zeros_like(x_seq[0])
    spikes = []
    for t in range(T):
        if decay_input:
            v = v + (x_seq[t] - v) * decay
        else:
            v = v * (1.0 - decay) + x_seq[t]
        spike = ref_firing_function(thr, v)
        v = v - spike
        spikes.append(spike)
    return torch.stack(spikes)


def test_trlif_matches_reference_node():
    T, B, N = 6, 4, 32
    x_seq = torch.randn(T, B, N, dtype=torch.double)

    mine = TrLIFNode(
        init_tau=2.0, init_v_threshold=1.0, decay_input=True, learnable_tau=True,
        mp_init=True, surrogate_function=surrogate.Rect(alpha=1.0),
        detach_reset=False, step_mode="m", backend="torch",
    ).double().train()

    # Shared parameters so gradients are directly comparable.
    w = mine.w.detach().clone().requires_grad_(True)
    thr_param = mine.thr_param.detach().clone().requires_grad_(True)

    x1 = x_seq.clone().requires_grad_(True)
    s_ref = _ref_trlif_forward(x1, w, thr_param)
    s_ref.sum().backward()

    x2 = x_seq.clone().requires_grad_(True)
    s_mine = mine(x2)
    s_mine.sum().backward()
    functional.reset_net(mine)

    assert torch.allclose(s_ref, s_mine), "spike sequence mismatch"
    assert torch.allclose(x1.grad, x2.grad, atol=1e-9), "grad_x mismatch"
    assert torch.allclose(w.grad, mine.w.grad, atol=1e-9), "grad_tau(w) mismatch"
    assert torch.allclose(thr_param.grad, mine.thr_param.grad, atol=1e-9), "grad_thr mismatch"
    print("[3] TrLIFNode == trlif.py training dynamics  (spikes, grad_x, grad_w, grad_thr)  OK")


def test_mp_init_ema_update():
    T, B, N = 4, 8, 16
    node = TrLIFNode(init_tau=2.0, init_v_threshold=1.0, mp_init=True,
                     mp_init_momentum=0.1, step_mode="m", backend="torch").train()
    assert float(node.running_mean) == 0.0

    prev = node.running_mean.clone()
    for it in range(5):
        x_seq = torch.randn(T, B, N)
        _ = node(x_seq)
        # Snapshot the values the EMA should consume, then trigger reset.
        v_snapshot = node.v.clone()
        mask_snapshot = node.mask.clone()
        functional.reset_net(node)

        masked = v_snapshot * mask_snapshot.float()
        cnt = mask_snapshot.float().sum().clamp_min(1.0)
        expected_mean = masked.sum() / cnt
        expected = prev * 0.9 + expected_mean * 0.1
        assert torch.allclose(node.running_mean, expected, atol=1e-6), f"EMA iter {it}"
        prev = node.running_mean.clone()
    # Inference must not update the running mean.
    node.eval()
    frozen = node.running_mean.clone()
    _ = node(torch.randn(T, B, N))
    functional.reset_net(node)
    assert torch.allclose(node.running_mean, frozen), "running_mean changed in eval"
    print("[4] MP-Init EMA update matches reference math; frozen at eval  OK")


def test_threshold_invariance():
    """As V_thr grows, TrSG keeps |grad_input| ~constant; AS/RS-SG do not."""
    sg = surrogate.Rect(alpha=1.0)
    thrs = [0.25, 1.0, 4.0]

    def grad_norm(kind, thr_val):
        v = (torch.randn(20000, dtype=torch.double) * thr_val).requires_grad_(True)
        thr = torch.tensor(thr_val, dtype=torch.double)
        if kind == "AS":  # absolute scale: surrogate(v - thr), output {0,1}
            x = v - thr
            out = (heaviside(x).detach() - sg.primitive_function(x, alpha=1.0)).detach() + sg.primitive_function(x, alpha=1.0)
        elif kind == "RS":  # relative scale, output {0,1} (no rescale)
            x = v / thr - 1.0
            soft = sg.primitive_function(x, alpha=1.0)
            out = (heaviside(v - thr).detach() - soft).detach() + soft
        else:  # TrSG
            out = TrSG(sg)(v, thr)
        out.sum().backward()
        return v.grad.abs().mean().item()

    res = {k: [grad_norm(k, t) for t in thrs] for k in ["AS", "RS", "TrSG"]}
    trsg_spread = max(res["TrSG"]) / max(min(res["TrSG"]), 1e-12)
    rs_spread = max(res["RS"]) / max(min(res["RS"]), 1e-12)
    print("    mean |grad_input| across V_thr =", thrs)
    for k in ["AS", "RS", "TrSG"]:
        print(f"      {k:4s}: {['%.4f' % x for x in res[k]]}")
    assert trsg_spread < 1.5, "TrSG grad magnitude should be ~invariant"
    assert rs_spread > trsg_spread, "RS-SG should vary more than TrSG"
    print("[5] Threshold-invariance holds for TrSG (AS floods/starves, RS explodes/vanishes)  OK")


def test_learnable_flags_combinations():
    from spikingjelly.activation_based import neuron

    T, B, N = 3, 4, 8
    for lv in (True, False):
        for lt in (True, False):
            node = TrLIFNode(
                init_tau=2.0, init_v_threshold=1.0, learnable_v_threshold=lv,
                learnable_tau=lt, mp_init=True, step_mode="m", backend="torch",
            ).train()
            x = torch.randn(T, B, N, requires_grad=True)
            node(x).sum().backward()
            # threshold grad present iff learnable_v_threshold
            assert isinstance(node.thr_param, torch.nn.Parameter) == lv
            if lv:
                assert node.thr_param.grad is not None and node.thr_param.grad.abs() > 0
            else:
                assert node.thr_param.grad is None  # buffer, no grad
            # tau grad present iff learnable_tau
            assert node.w.requires_grad == lt
            if lt:
                assert node.w.grad is not None
            else:
                assert node.w.grad is None
            functional.reset_net(node)

    # V_thr=1.0 fixed + Rect(alpha=1) reduces exactly to a standard LIF with a
    # Rect surrogate (arg v-1, output {0,1}).
    T, B, N = 5, 2, 64
    x_seq = torch.randn(T, B, N, dtype=torch.double)
    trlif = TrLIFNode(
        init_tau=2.0, init_v_threshold=1.0, learnable_v_threshold=False,
        learnable_tau=False, mp_init=False, surrogate_function=surrogate.Rect(alpha=1.0),
        step_mode="m", backend="torch",
    ).double().train()
    std = neuron.ParametricLIFNode(
        init_tau=2.0, v_threshold=1.0, v_reset=None, decay_input=True,
        surrogate_function=surrogate.Rect(alpha=1.0), step_mode="m", backend="torch",
    ).double().train()
    # PLIF's tau is learnable; freeze it to the same value for a fair compare.
    std.w.requires_grad_(False)

    x1 = x_seq.clone().requires_grad_(True)
    s1 = trlif(x1); s1.sum().backward(); functional.reset_net(trlif)
    x2 = x_seq.clone().requires_grad_(True)
    s2 = std(x2); s2.sum().backward(); functional.reset_net(std)
    assert torch.allclose(s1, s2), "V_thr=1 TrSG spikes != standard LIF"
    assert torch.allclose(x1.grad, x2.grad, atol=1e-9), "V_thr=1 TrSG grad != standard LIF"
    print("[6] All 4 train/fix combos flow grads correctly; V_thr=1 fixed == standard LIF+Rect  OK")


ALL = [
    test_trsg_rect_matches_reference_firing_function,
    test_trsg_generic_matches_ste_relaxation,
    test_trlif_matches_reference_node,
    test_mp_init_ema_update,
    test_threshold_invariance,
    test_learnable_flags_combinations,
]

if __name__ == "__main__":
    for t in ALL:
        t()
    print("\nAll TrSG / TrLIF CPU checks passed.")
