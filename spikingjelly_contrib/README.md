# SpikingJelly contribution plan — TrSG & MP-Init

This directory contains **reference (torch-backend) prototypes** and a concrete
plan for upstreaming TrSG and MP-Init (arXiv:2511.08708) into SpikingJelly,
matching the maintainers' guidance from the email thread:

- **Wei Fang**: try the new **FlexSN** path (Python `core` → auto-generated
  Triton kernels) instead of hand-written CUDA. Maintenance is led by
  **Yifan Huang**.
- **Yifan Huang**: split into **two PRs** — (1) the **surrogate function**,
  (2) the **neuron model** (threshold + tau joint training with a CUDA/kernel
  backend). Follow `CONTRIBUTING.md`.

Everything here targets the **`master`** branch of SpikingJelly (package
`2.0.0.dev1`), whose layout differs from the PyPI release: `neuron` is a
package (`neuron/base_node.py`, `neuron/plif.py`, `neuron/flexsn.py`, ...),
and formatting/tests use `uv` + `ruff` + `pytest`.

## Files

| File | Role | Maps to (SpikingJelly master) |
|---|---|---|
| `trsg.py` | `TrSG` threshold-robust surrogate wrapper + autograd | `spikingjelly/activation_based/surrogate.py` |
| `trlif_node.py` | `TrLIFNode` (learnable τ + threshold, TrSG, MP-Init), torch backend | `spikingjelly/activation_based/neuron/lif_variants.py` (next to `GatedLIFNode`) |
| `test_trsg_trlif.py` | CPU verification (all pass) | `test/activation_based/test_trsg.py` (new) |

## What was verified on CPU (`python -m spikingjelly_contrib.test_trsg_trlif`)

1. **`TrSG(Rect, α=1)` reproduces the released `trlif.py` `firing_function`
   exactly** — forward, `grad_v`, and `grad_v_threshold` all match to 1e-9.
2. **`TrSG(shape)` for Rect / ATan / Sigmoid / PiecewiseQuadratic equals the
   principled STE realization** `O = V_thr · S_ste`, `S_ste = STE(H, Φ)`.
3. **`TrLIFNode` matches `trlif.py`'s training dynamics** (spikes + grads to
   input, τ, threshold), i.e. PLIF-style charge + TrSG fire + soft reset.
4. **MP-Init EMA** matches the reference math across resets and is frozen at
   eval.
5. **Threshold-invariance** (the paper's core claim): mean `|grad_input|` stays
   ~flat for TrSG as `V_thr` ∈ {0.25, 1, 4} while AS-SG starves and RS-SG
   vanishes:

   | | V_thr=0.25 | 1.0 | 4.0 |
   |---|---|---|---|
   | AS-SG | 0.843 | 0.242 | 0.062 |
   | RS-SG | 0.973 | 0.239 | 0.059 |
   | **TrSG** | **0.242** | **0.242** | **0.248** |

## Design decisions (and why)

### TrSG must be threshold-aware — it is not a drop-in `SurrogateFunctionBase`

SpikingJelly's surrogate contract is strictly `f(v - v_threshold) → {0,1}`
(`BaseNode.neuronal_fire` calls `self.surrogate_function(self.v -
self.v_threshold)`). TrSG needs **both** the relative argument `v/V_thr − 1`
**and** the forward rescale `× V_thr`, so it cannot be a plain surrogate that
only sees the pre-differenced tensor. Two clean options exist; we chose the
second:

- ❌ Store `v_threshold` on the surrogate instance — duplicates neuron state,
  breaks when the threshold is a learnable tensor.
- ✅ Make `TrSG` a **threshold-aware wrapper** called as `trsg(v, v_threshold)`,
  and have the neuron's `neuronal_fire` call it that way. `TrSG` reuses the
  wrapped shape's `primitive_function` (Φ) and `backward` (φ), so it inherits
  every existing/ future SpikingJelly surrogate shape for free.

This keeps the two PRs cleanly separable: PR1 adds `TrSG` + isolated
surrogate-level tests; PR2 adds the neuron that consumes it.

### The exact TrSG gradient

For hard spike `S = H(v − V_thr)`, output `O = V_thr · S`, argument
`x = v/V_thr − 1`:

```
grad_v     = grad_output · φ(x)                     # threshold-invariant magnitude
grad_Vthr  = grad_output · S − grad_v · (x + 1)     # S = O/V_thr (hard), x+1 = v/V_thr
```

Subtlety: the first term of `grad_Vthr` is the **hard** spike `S`, not the soft
primitive `Φ`. In the released `firing_function` the `× thr` multiply saves the
*rounded* value `round(clamp(...)) = S` for its backward, so the threshold
gradient uses `S`. This is the exact autograd of the straight-through
realization `O = V_thr · S_ste`, `S_ste = (S − Φ).detach() + Φ`. Verified in
test [1].

### TrLIFNode

- Subclasses `BaseNode` (like `ParametricLIFNode`); passes a plain float
  threshold to `super().__init__` (the base asserts float) and holds the real
  threshold as `self.thr_param` with `V_thr = softplus(thr_param)`.
- **Independent train/fix flags for both parameters** (`learnable_v_threshold`,
  `learnable_tau`), so all four combinations are one class:

  | `learnable_v_threshold` | `learnable_tau` | reduces to |
  |---|---|---|
  | ✅ | ✅ | joint threshold+τ training (paper default, the CUDA-speedup case) |
  | ✅ | ❌ | threshold-only training |
  | ❌ | ✅ | τ-only training (PLIF + TrSG surrogate) |
  | ❌ | ❌ | fixed LIF; with `init_v_threshold=1.0` this is **exactly** a standard `ParametricLIFNode` + `Rect` surrogate (verified, test [6]) |

  Fixed parameters are stored as buffers (τ as a `requires_grad=False`
  Parameter, PLIF-style; threshold as a buffer) — still in `state_dict`, still
  broadcast, but no gradient and skipped by the optimizer/DDP reducer. The
  `TrSG` surrogate needs no changes for the fixed case: when `V_thr` doesn't
  require grad its backward simply returns `None` for the threshold gradient.
- Learnable τ via `self.w`, `τ = 1/sigmoid(w)` (PLIF convention).
- `v_reset=None` (soft reset). Because the firing output is **already** scaled
  by `V_thr`, `neuronal_reset` subtracts the scaled spike directly
  (`v = v − spike`), **not** the base's `v − V_thr·spike` — overriding this is
  mandatory or reset desyncs.
- **MP-Init**: `running_mean` is a **buffer** (in `state_dict`, follows
  `.to()`, DDP-broadcast like a BN stat). `reset()` EMA-updates it from the
  last-step `v` over active neurons (`torch.no_grad`, `self.training`-gated)
  *before* `super().reset()`; the first step of each simulation seeds `v` from
  it. The active-neuron mask and count are plain per-simulation attributes.

## ⚠️ FlexSN caveat (important for the maintainer's suggestion)

We analyzed `neuron/flexsn.py` and its Triton codegen
(`triton_kernel/torch2triton/`). **A `FlexSN` core that closes over
`nn.Parameter` scalars (learnable `V_thr`, `w`) does *not* get Triton-accelerated
training with gradients to those parameters:**

- The generated Triton scan is strictly element-wise over one flattened
  `N·C·L` dimension; there is **no slot for a shared learnable scalar**, and the
  homogeneous-shape contract rejects passing a 1-element `thr` alongside a large
  `x` (`neuron/flexsn.py:_validate_scan_backend_contract`).
- A closed-over parameter tensor becomes an fx `get_attr` (codegen raises
  `NotImplementedError`) or is lifted to an extra placeholder (an assertion
  `len(args) == num_inputs + num_states` fails). Either way the kernel build
  **silently falls back to the eager Python scan** — gradients flow (via torch
  autograd) but with **no speedup**, defeating the reason to use FlexSN.
- `F.softplus` is also not in the codegen op whitelist (`round`, `clamp`,
  `sigmoid`, comparisons, `where`, `detach`, custom-Function STE **are**).

**The only Triton-compatible route** is to keep `thr`/`w` out of the closure:
compute `softplus`/`sigmoid` in the enclosing module and feed the
**broadcast, full-size** `thr` (and `tau`) tensors as extra `num_inputs`/
`num_states`; the backward kernel then returns full-size grads that reduce to
the scalars via the outer `expand`. This works but costs a `[T, *shape]` tensor
per scalar.

**Recommendation:** raise this with Yifan before committing PR2 to FlexSN. The
existing hand-written PLIF Triton kernel
(`triton_kernel/neuron_kernel/plif.py`) already carries a learnable `tau`
gradient; TrLIF's learnable `thr` + soft reset + round-STE is a close cousin and
may be cleaner as a dedicated kernel in that style. This torch reference is what
either backend must match numerically.

## ⚠️ Do not upstream `trlif_cupy.py` as-is

The repo's `trlif_cupy.py` does **not** match the reference `trlif.py`:

1. Its forward emits `spike ∈ {0,1}`, not `{0, V_thr}` (drops the `× thr`),
   which also skews the input- and threshold-path gradients.
2. Its default surrogate is `Sigmoid`, not the paper's rectangular round/clamp
   STE (`lsq`, α=1, would match).
3. `w` (τ) is `requires_grad=False` — τ never trains, contradicting the paper.
4. An extra `running_mean.clamp(min=0)`, different `init_thr` default, and a
   `self.z` vs `self.thr` attribute name (state-dict incompatibility).

Any accelerated backend should be re-derived from the torch reference here.

## Two-PR checklist (SpikingJelly `master` conventions)

**PR1 — `TrSG` surrogate**
- Add `TrSG` (+ `trsg_function`) to `surrogate.py` with the bilingual CN/EN
  docstring skeleton (see `Rect`/`ATan`), a `.. math::` block, and a References
  footnote `[#TrSG]_`.
- Register `TrSG` in the `:members:` list of
  `docs/source/APIs/spikingjelly.activation_based.surrogate.rst` and add the
  citation to its References.
- Add `test/activation_based/test_trsg.py` (tests 1, 2, 5 here — CPU, pytest).

**PR2 — `TrLIFNode` neuron**
- Add `TrLIFNode` to `neuron/lif_variants.py`; add `"TrLIFNode"` to its
  `__all__` (already wildcard-imported in `neuron/__init__.py`).
- Register in `docs/source/APIs/spikingjelly.activation_based.neuron.rst`
  under "Research-specific Neuron Modules".
- Add EN+CN tutorials under `docs/source/tutorials/{en,cn}/` and wire both
  `index.rst` toctrees (optional but expected for a new neuron).
- Add `test/activation_based/test_trlif.py` (tests 3, 4 here).
- Decide the accelerated backend (FlexSN-with-broadcast vs dedicated kernel,
  per the caveat above) with Yifan; ship torch first, kernel second.

**Both**
- `uv format` (ruff, double quotes, 88 cols) before committing.
- `pytest` locally; guard any CUDA/Triton parity tests with
  `@pytest.mark.skipif(not torch.cuda.is_available(), ...)`.
- Python ≥ 3.11, torch ≥ 2.6.0. Single-purpose PRs, clear what/why.
- `CHANGELOG.md` `## Unreleased` entries (the changelog CI checks the
  generated rst stays in sync).
