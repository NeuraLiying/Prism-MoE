# ICML-26: Prism-MoE: Efficient Dense-to-MoE Conversion for Visual Autoregressive Generation

> **TL;DR:** Prism-MoE converts a pretrained visual autoregressive image generator into a sparse top-k2 Mixture-of-Experts model via a training-free two-stage initialization plus lightweight distillation, preserving image quality at a fraction of the compute.

<p align="center">
  <a href="https://icml.cc/virtual/2026/poster/65615"><img alt="Paper" src="https://badgen.net/badge/Paper/ICML%202026/d12?labelColor=2f363d&scale=1.2"></a>
  <a href="https://neuraliying.github.io/Prism-MoE/"><img alt="Project Page" src="https://badgen.net/badge/Project%20Page/Visit/1f6feb?labelColor=2f363d&scale=1.2"></a>
  <a href="https://opensource.org/license/apache-2-0"><img alt="License" src="https://badgen.net/badge/License/Apache%202.0/d4a72c?labelColor=2f363d&scale=1.2"></a>
</p>

### Ying Li, Zefang Wang, Zhaode Wang, Zhiwen Chen, Chengfei Lv, Huan Wang

<p align="center">
  <img src="assets/logo.png" width="50%">
</p>

---

## 📰 News

- **[2026-07-03]** 🔥 Code is released.
- **[2026-05-01]** 🎉 Prism-MoE is accepted to **ICML 2026** as a poster.

---

## Overview

<p align="center">
  <img src="assets/teaser_2.png" width="70%">
</p>

### Problem

Visual autoregressive image generation keeps scaling — from next-token models (LlamaGen, Janus-Pro) and next-scale models (VAR, Infinity) to tens-of-billions-parameter systems — which substantially increases inference cost. Mixture-of-Experts (MoE) is a natural way to add capacity through sparse activation and is already standard in large language models, but training MoE from scratch is prohibitively expensive, making low-cost **dense-to-MoE conversion** the appealing route. Yet dense-to-MoE conversion for autoregressive image generation remains underexplored: prior efforts mostly target general upcycling or diffusion with heavy fine-tuning, and the conversion is further complicated by step-dependent activation non-stationarity and spatially heterogeneous token uncertainty.

So we ask: **can MoE be brought into visual autoregressive generation efficiently** — converting a pretrained dense generator into a sparse MoE with only a fraction of the active parameters and training budget, while preserving generation quality?

### Motivation

<p align="center">
  <img src="assets/sub_space.png" width="70%">
</p>

Visual AR representations exhibit strong step-dependent low-rank structures, indicating that feature complexity varies significantly across generation steps.

<p align="center">
  <img src="assets/lb_loss_fail.png" width="70%">
</p>

Expert activation patterns are highly non-uniform and evolve during fine-tuning, suggesting unstable routing behavior and lack of consistent expert specialization.

<p align="center">
  <img src="assets/entropy_distribution.png" width="70%">
</p>

Generated images exhibit spatially heterogeneous entropy patterns, where high-uncertainty regions align with semantic structures, indicating a mismatch between routing behavior and visual complexity.

### Method

<p align="center">
  <img src="assets/method.png">
</p>

**Stage I: Trajectory-Consistent Initialization**

- Importance and stability analysis → Identify critical and stable neurons
- Shared and routed expert decomposition → Separate shared and specialized features
- Closed-form trajectory compensation → Recover the dense trajectory

**Stage II: Condition-Adaptive Sparse Fine-Tuning**

- Dense-to-sparse distillation → Preserve dense representations
- Normalized cosine router → Stabilize expert selection
- Spatially structured routing regularization → Promote coherent expert specialization

### Main Results

<p align="center">
  <img src="assets/exp_1.png" width="70%">
</p>

- **Strong initialization.** At 75% active parameters, Prism-MoE achieves the best sparse initialization result, with FID 5.03.
- **Near dense quality.** After efficient fine-tuning, Prism-MoE reaches FID closely matching the dense VAR-d20 baseline.

### Ablation

<p align="center">
  <img src="assets/ablation_entropy.png" width="70%">
</p>

Baseline exhibits scattered entropy with unstable uncertainty regions. Ours produces structured entropy aligned with semantic regions and more consistent routing behavior.

### Visual Results

<p align="center">
  <img src="assets/appendix_var_375.png" width="70%">
</p>
<p align="center"><em>VAR-d20, 37.5% activation</em></p>

<p align="center">
  <img src="assets/appendix-llamagen-375.png" width="70%">
</p>
<p align="center"><em>LlamaGen, 37.5% activation</em></p>

---

## Implementation

Prism-MoE converts a dense visual autoregressive image generator into a sparse top-k Mixture-of-Experts model and finetunes it with lightweight routing and output-level distillation. The code is organized around three public entry points:

- `dense_to_moe.py`: dense-to-MoE initialization.
- `finetune.py`: top-k2 finetuning with the release recipe.
- `eval.py`: 50K ImageNet FID evaluation.

### Layout

```text
Prism-MoE/
  dense_to_moe.py          # Dense-to-MoE initialization
  finetune.py              # Finetuning entry point
  eval.py                  # Sample generation and FID
  models/                  # VAR and Prism-MoE model code
  initialization/          # Stage I and Stage II initialization internals
  engines/                 # Internal train/eval engines
  utils/, common/          # Data and runtime utilities
  assets/                  # Paper PDF
```

Most users should call the three entry scripts above.

### Setup

Create an environment with PyTorch, CUDA, and the packages in `requirements.txt`.

```bash
pip install -r requirements.txt
```

The examples below assume ImageNet training images are available in a standard folder layout and that dense VAR and VAE checkpoints have already been downloaded.

### Dense-to-MoE Initialization

```bash
python dense_to_moe.py \
  --dense-ckpt /path/to/var_d16.pth \
  --vae-ckpt /path/to/vae_ch160v4096z32.pth \
  --imagenet-dir /path/to/imagenet/train \
  --output outputs/prism_moe_init.pth
```

This command runs the two initialization stages and writes a refined Prism-MoE checkpoint.

### Finetuning

The release recipe is split into a bridge stage and an alignment stage. The bridge stage stabilizes the top-k2 MoE; the alignment stage resumes from the bridge checkpoint and adds a small output-level on-policy distillation signal.

```bash
python finetune.py \
  --stage bridge \
  --moe-ckpt outputs/prism_moe_init.pth \
  --dense-ckpt /path/to/var_d16.pth \
  --vae-ckpt /path/to/vae_ch160v4096z32.pth \
  --imagenet-dir /path/to/imagenet/train \
  --output-dir outputs/bridge
```

```bash
python finetune.py \
  --stage align \
  --moe-ckpt outputs/prism_moe_init.pth \
  --resume-ckpt outputs/bridge/checkpoints/ckpt_ep7.pth \
  --dense-ckpt /path/to/var_d16.pth \
  --vae-ckpt /path/to/vae_ch160v4096z32.pth \
  --imagenet-dir /path/to/imagenet/train \
  --output-dir outputs/align
```

By default, finetuning uses an effective global batch size of 768.

### Evaluation

```bash
python eval.py \
  --moe-ckpt /path/to/prism_moe.pth \
  --vae-ckpt /path/to/vae_ch160v4096z32.pth \
  --ref-npz /path/to/imagenet256_fid_stats.npz \
  --output-dir outputs/fid_eval
```

The evaluator generates 50 images per ImageNet class, computes 50K FID, writes `fid_result.txt` and `generation_stats.json`, and removes generated images unless `--keep-images` is set.
