# Vision Transformer (SigLIP) — Built From Scratch

A from-scratch PyTorch reimplementation of the SigLIP vision encoder (ViT architecture), verified numerically against HuggingFace's official implementation, fine-tuned on satellite imagery, and exported for inference in C++.

## Overview

This project implements every component of a Vision Transformer by hand — patch embeddings, multi-head self-attention, MLP blocks, and the full 12-layer encoder stack — and validates each piece against `google/siglip-base-patch16-224` from HuggingFace to confirm bit-for-bit numerical correctness before building on top of it.

**Pipeline:**
1. Implement ViT/SigLIP architecture from scratch
2. Verify each component (embeddings, attention, full model) against HuggingFace's reference implementation
3. Fine-tune on EuroSAT satellite imagery (10-class land-use classification)
4. Export the trained model to TorchScript and run inference in C++ via LibTorch

## Architecture

- **Patch embeddings**: `Conv2d`-based patchify (16×16 patches, non-overlapping) + learned positional embeddings
- **Multi-head self-attention**: 12 heads, scaled dot-product attention, implemented both as an explicit per-head loop and as an optimized batched reshape version
- **MLP block**: 768 → 3072 → 768 with GELU (tanh approximation)
- **Encoder layer**: pre-norm residual blocks (LayerNorm → Attention → residual, LayerNorm → MLP → residual)
- **Encoder**: 12 stacked encoder layers
- **Full model**: embeddings → encoder → post-layernorm

Config (matches `siglip-base-patch16-224`):

| Parameter | Value |
|---|---|
| Hidden size | 768 |
| Attention heads | 12 |
| Encoder layers | 12 |
| Intermediate (MLP) size | 3072 |
| Patch size | 16 |
| Image size | 224 |
| Patches per image | 196 |

## Verification

Each component was checked against HuggingFace's `SiglipVisionModel` by copying pretrained weights into the from-scratch modules and comparing outputs on identical inputs:

| Component | Max absolute difference vs. HuggingFace |
|---|---|
| Patch + position embeddings | `0.0` |
| Single attention layer | `~1e-6` |
| Full 12-layer model | `~8.4e-5` |

## Fine-tuning results (EuroSAT)

Fine-tuned on [EuroSAT](https://huggingface.co/datasets/tanganke/eurosat) (10-class satellite land-use classification, 21,600 train / 2,700 test images) using a staged approach:

1. **Linear probe**: froze the pretrained backbone, trained only a linear classification head
2. **Partial unfreeze**: unfroze the last 2 encoder layers with a reduced learning rate, continued fine-tuning

| Stage | Test accuracy |
|---|---|
| Linear probe (frozen backbone) | 95.11% |
| + last 2 layers unfrozen | **95.67%** |

## C++ inference

The fine-tuned model is exported via `torch.jit.trace` to a self-contained TorchScript file and loaded for inference in C++ using LibTorch — no Python runtime required at inference time.

```
cpp_inference/
├── CMakeLists.txt
├── main.cpp
└── model_ft_traced.pt
```

## Project structure

```
├── vision_transformer.py      # Full from-scratch ViT/SigLIP implementation + training
├── model_tracing.py           # TorchScript export for C++ deployment
├── cpp_inference/             # C++ inference via LibTorch
│   ├── CMakeLists.txt
│   └── main.cpp
└── README.md
```

## Requirements

```
torch
torchvision
transformers
datasets
pillow
```

For C++ inference: [LibTorch](https://pytorch.org/get-started/locally/) and CMake ≥ 3.18.

## Acknowledgments

Architecture and verification methodology follow Google's [SigLIP](https://arxiv.org/abs/2303.15343) and the original [ViT paper](https://arxiv.org/abs/2010.11929). Reference weights and config from [`google/siglip-base-patch16-224`](https://huggingface.co/google/siglip-base-patch16-224) via HuggingFace Transformers.
