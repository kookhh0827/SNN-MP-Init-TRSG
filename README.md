# [WACV2026] Stabilizing Direct Training of Spiking Neural Networks: MP-Init & TrSG

[![arXiv](https://img.shields.io/badge/arXiv-2511.08708-b31b1b.svg)](https://arxiv.org/abs/2511.08708)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](#installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B%20/%202.x-orange.svg)](#installation)

Official implementation of **MP-Init (Membrane Potential Initialization)** and **TrSG (Threshold-robust Surrogate Gradient)** from:

> **Stabilizing Direct Training of Spiking Neural Networks: Membrane Potential Initialization and Threshold-robust Surrogate Gradient**  
> Hyunho Kook, Byeongho Yu, Jeong Min Oh, Eunhyeok Park (POSTECH)  
> arXiv:2511.08708 (Nov 2025)

---

## TL;DR

Directly-trained SNNs often face:
- **Temporal Covariate Shift (TCS)** across timesteps (especially harmful at early timesteps)
- **Unstable gradients** when **LIF neuron parameters** (e.g., threshold `Vthr`) are learnable

We propose two lightweight, drop-in components:

- **MP-Init**: initializes each layer’s membrane potential using a running estimate of its *stationary* value.
- **TrSG**: makes surrogate gradients robust to threshold scale via a *relative* argument + *threshold multiplication* trick.

Both are implemented in a single neuron module: [`trLIFNode`](./trlif.py).

---

## Method overview

### MP-Init (Membrane Potential Initialization)

We set the initial membrane potential of each layer using an EMA running mean of the last-step membrane potential:
- Training: update the running mean using **only active neurons** (silent neurons are excluded).
- Inference: keep the running mean fixed and reuse it as initialization.

**(High-level pseudo-code)**

```text
Initialize per-layer running mean μ_l = 0

For each training batch:
  Set U_l[0] ← μ_l
  Run SNN for t = 1..T
  Compute μ_batch from U_l[T] over active neurons only
  Update μ_l ← (1 - β) μ_l + β μ_batch

Inference:
  Set U_l[0] ← μ_l and run standard forward dynamics
```

---

### TrSG (Threshold-robust Surrogate Gradient)

Naïve surrogate gradients can become **too small / too large** depending on threshold scale.  
TrSG stabilizes gradients w.r.t. `Vthr` by:

1) Using a **relative-scale** argument (ratio)  
2) Multiplying the (binarized) spike by `Vthr` on the forward path (training-time trick)

In this repo (`trlif.py`), the TrSG firing function uses:

```python
x = v / thr - 1/2
x = clamp(x, 0, 1)
s = round_pass(x)
out = s * thr
```

At inference, the same behavior can be kept as-is, or equivalently absorbed into the next layer weights by rescaling.

---

## Repository structure

- `trlif.py`  
  - `trLIFNode`: LIF neuron with **MP-Init + TrSG**
- `resnet19.py`  
  - CIFAR-style spiking ResNet-19 using `trLIFNode`
- `spiking_resnet.py`  
  - ImageNet-style spiking ResNet backbone (**requires `ta_srgd.py`, not included in this snapshot**)
- `train.py`  
  - DDP training / evaluation script (spawns one process per visible GPU)
- `functions.py`  
  - seeding + torchvision transforms
- `example_command.sh`  
  - (currently empty / placeholder)

---

## Installation

This code depends on:
- Python 3.8+
- PyTorch + torchvision
- SpikingJelly (`spikingjelly.activation_based`)

Example:

```bash
conda create -n snn_mpinit_trsg python=3.10 -y
conda activate snn_mpinit_trsg

# Install PyTorch matching your CUDA (see pytorch.org)
pip install torch torchvision

# SpikingJelly
pip install spikingjelly
```

---

## Usage

> `train.py` uses **DDP spawn** and will use **all visible GPUs**.  
> To restrict GPUs: set `CUDA_VISIBLE_DEVICES`.

### 0) Make output dirs

```bash
mkdir -p checkpoints snapshots
```

### 1) Train on CIFAR100 (ResNet-19, T=4)

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --model resnet19 --dataset cifar100 --dataset_folder ./data \
  --T 4 --epochs 200 --batch-size 64 --lr 0.02 \
  --weight_decay 5e-4 --momentum 0.9 \
  --init_tau 2.0 --init_thr 1.0 \
  --print-freq 100 \
  --save_names resnet19_cifar100_T4_mpinit_trsg
```

**Important:** `train.py` treats `--batch-size` as *global batch size* and divides it by `#GPUs` internally:
- 1 GPU: `--batch-size 64` → per-GPU 64  
- 4 GPUs: set `--batch-size 256` → per-GPU 64

### 2) Evaluate a saved snapshot

Snapshots are stored at:
- `./snapshots/<save_names>.pth.tar`

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --model resnet19 --dataset cifar100 --dataset_folder ./data \
  --T 4 --batch-size 64 \
  --evaluate --load_names resnet19_cifar100_T4_mpinit_trsg
```

### 3) Resume training from a checkpoint

Checkpoints are stored at:
- `./checkpoints/<load_names>.pth.tar`

To resume, pass both `--load_names` and `--save_names`:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --model resnet19 --dataset cifar100 --dataset_folder ./data \
  --T 4 --epochs 200 --batch-size 64 --lr 0.02 \
  --load_names resnet19_cifar100_T4_mpinit_trsg \
  --save_names resnet19_cifar100_T4_mpinit_trsg_resume
```

---

## Results (from the paper)

Reported top-1 accuracy (mean ± std):

- **CIFAR10 / ResNet-19**: 95.05 (T=2), 95.34 (T=4), 95.50 (T=6)  
- **CIFAR100 / ResNet-19**: 76.87 (T=2), 77.66 (T=4), 77.91 (T=6)  
- **ImageNet / ResNet-34**: 67.15 (T=4), 68.73 (T=6)  
- **ImageNet / SEW-ResNet-34**: 69.67 (T=4)  
- **DVS-CIFAR10 / ResNet-19**: 81.43 (T=10)

---

## Citation

```bibtex
@article{kook2025stabilizing,
  title   = {Stabilizing Direct Training of Spiking Neural Networks: Membrane Potential Initialization and Threshold-robust Surrogate Gradient},
  author  = {Kook, Hyunho and Yu, Byeongho and Oh, Jeong Min and Park, Eunhyeok},
  journal = {arXiv preprint arXiv:2511.08708},
  year    = {2025}
}
```

---

## Contact

- Hyunho Kook: kookhh0827@postech.ac.kr
