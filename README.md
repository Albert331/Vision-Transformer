# Vision Transformer (SigLIP)

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

## Zero-shot vs. fine-tuned comparison

SigLIP is natively a vision-language model, trained via image-text contrastive learning, and can classify images zero-shot by matching them against natural-language prompts (e.g. `"a satellite photo of forest"`) — with no task-specific training at all. To measure how well the general pretrained checkpoint transfers to satellite imagery without fine-tuning, zero-shot classification was run on the same EuroSAT test set using the full `SiglipModel` (vision + text towers) from HuggingFace.

| Approach | Test accuracy |
|---|---|
| Zero-shot (pretrained SigLIP, no training) | 10.00% (≈ random chance on 10 classes) |
| Fine-tuned, linear probe | 95.11% |
| Fine-tuned, partial unfreeze | **95.67%** |

**Takeaway**: general-purpose SigLIP pretraining (web images with natural-language captions) transfers essentially not at all to top-down satellite imagery — a substantial domain shift from anything seen during pretraining. This isolates fine-tuning, rather than the pretrained representation alone, as what actually makes the model usable for this task, and was verified with a sanity check on an in-distribution natural image (correctly and confidently classified zero-shot) to confirm the low EuroSAT score reflects a genuine domain gap rather than a pipeline error.

## Attention visualization

`SiglipAttention` was extended with an optional `return_attn` flag to expose raw attention weights (`[B, num_heads, 196, 196]`) from any encoder layer, without altering the layer's normal output. Attention received per patch (averaged across heads and query positions) was reshaped back into the 14×14 patch grid and overlaid on the original image.

**Findings:**
- Attention concentrates into a small number of sharp, isolated hotspots rather than spreading smoothly across semantically meaningful regions (field boundaries, crop patterns) — consistent with the **attention sink** phenomenon documented in ViT interpretability research, where a small number of tokens accumulate disproportionate attention largely independent of their visual content.
- Comparing the pretrained backbone against the fine-tuned model (same image, same layer) showed the raw attention maps look visually similar at a glance, but the underlying weights differ meaningfully (max difference 0.22 in attention received at specific patches). A difference map isolated exactly where fine-tuning shifted attention: rather than a broad redistribution, fine-tuning intensified one pre-existing hotspot while leaving most patches essentially unchanged.
- Inspecting the specific patch most affected by fine-tuning showed no distinguishing visual features (a uniform patch of bare soil) — reinforcing that this is sink-like behavior rather than the model learning to attend to genuinely salient content.

## C++ inference

The fine-tuned model is exported via `torch.jit.trace` to a self-contained TorchScript file and loaded for inference in C++ using LibTorch (GPU/CUDA build) — no Python runtime required at inference time.

```
cpp_inference/
├── CMakeLists.txt
├── main.cpp
└── model_ft_traced.pt
```

### Prerequisites

- **CUDA-capable NVIDIA GPU** with drivers installed
- **[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)** — a full toolkit install (not just the driver) is required for LibTorch's CMake build to link correctly; version should match your LibTorch download below
- **[LibTorch](https://pytorch.org/get-started/locally/)** (C++/Java, CUDA build) — download and unzip anywhere; note the path for `CMakeLists.txt`
- **CMake ≥ 3.18**
- **MSVC** (via Visual Studio or Visual Studio Build Tools, "Desktop development with C++" workload)
- A **C++20**-capable toolchain — this LibTorch build's headers require C++20, not C++17

### Build steps

1. Open `cpp_inference/CMakeLists.txt` in Visual Studio via `File → Open → CMake Project...`
2. Edit `set(CMAKE_PREFIX_PATH "...")` in `CMakeLists.txt` to point at your actual LibTorch install location
3. **Select the `x64-Release` configuration** — do not build in Debug. LibTorch's official downloads are Release builds; building your project in Debug against a Release LibTorch causes CRT mismatches and cryptic runtime file-loading errors (`errno 22` on `fopen`), even when the file path is correct.
4. Build → Rebuild All
5. Run (`Ctrl+F5`)

Expected output:
```
Model loaded successfully.
Running on: CUDA
Output shape: [1, 10]
Predicted class: <0-9>
```

(The predicted class will vary — `main.cpp` currently feeds in random noise as a placeholder input rather than a real preprocessed image.)

### Common pitfalls

- **`Could not find a package configuration file provided by "Torch"`** — `CMAKE_PREFIX_PATH` doesn't match your actual LibTorch unzip location (check for a nested `libtorch/libtorch/` folder from the zip extraction).
- **`Caffe2Config.cmake` error about missing CUDA libraries** — the CUDA Toolkit (not just the driver) isn't installed, or isn't discoverable by CMake.
- **`requires at least '/std:c++20'`** — set `CMAKE_CXX_STANDARD` to `20` in `CMakeLists.txt`.
- **`fopen` fails with `errno 22` despite the file existing** — almost always a Debug/Release mismatch between your build configuration and the LibTorch binaries. Switch to `x64-Release`.

## Project structure

```
├── vision_transformer.py      # Full from-scratch ViT/SigLIP implementation + training
├── vlm.py                     # Zero-shot classification via full SigLIP (vision + text)
├── attention_viz.py           # Attention extraction + heatmap visualization
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
matplotlib
scikit-learn
```

For C++ inference: [LibTorch](https://pytorch.org/get-started/locally/) and CMake ≥ 3.18.

## Acknowledgments

Architecture and verification methodology follow Google's [SigLIP](https://arxiv.org/abs/2303.15343) and the original [ViT paper](https://arxiv.org/abs/2010.11929). Reference weights and config from [`google/siglip-base-patch16-224`](https://huggingface.co/google/siglip-base-patch16-224) via HuggingFace Transformers.
