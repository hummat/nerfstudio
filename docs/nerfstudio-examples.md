# Nerfstudio CLI examples

This document mirrors `sdfstudio-examples.md` and collects a few concrete `ns-train` command lines for common Nerfstudio methods. It is not exhaustive; see `docs/nerfology/methods/*.md` and `docs/quickstart` for more detailed guides.

All commands assume you have installed Nerfstudio and activated the appropriate environment.

## Nerfacto on the Nerfstudio “poster” scene

Recommended first test: real‑world capture with the default `nerfacto` method.

```bash
# Download demo data
ns-download-data nerfstudio --capture-name=poster

# Train Nerfacto and launch viewer
ns-train nerfacto --data data/nerfstudio/poster
```

## Nerfacto-big with multi‑GPU training on the “aspen” scene

Larger Nerfacto variant with more capacity; benefits from multiple GPUs.

```bash
# Download data
ns-download-data nerfstudio --capture-name=aspen

# 1 GPU (8192 rays per batch)
export CUDA_VISIBLE_DEVICES=0
ns-train nerfacto-big --vis viewer+wandb \
  --machine.num-devices 1 \
  --pipeline.datamanager.train-num-rays-per-batch 4096 \
  --data data/nerfstudio/aspen

# 2 GPUs (4096 rays per GPU, effectively 8192 rays per batch)
export CUDA_VISIBLE_DEVICES=0,1
ns-train nerfacto-big --vis viewer+wandb \
  --machine.num-devices 2 \
  --pipeline.datamanager.train-num-rays-per-batch 4096 \
  --data data/nerfstudio/aspen
```

## Instant-NGP on a Nerfstudio capture

Instant‑NGP‑style method using the Nerfstudio dataparser.

```bash
# Reuse the poster scene from above (or any nerfstudio-format dataset)
ns-train instant-ngp --data data/nerfstudio/poster
```

## Vanilla NeRF and Mip-NeRF on the Blender “lego” scene

Classic baselines on the standard synthetic Blender dataset.

```bash
# Download Blender dataset (includes lego scene)
ns-download-data blender

# Train original NeRF on lego
ns-train vanilla-nerf blender-data

# Train Mip-NeRF on lego
ns-train mipnerf blender-data
```

## D-NeRF on the D-NeRF “lego” scene

Dynamic NeRF baseline on the D‑NeRF dataset.

```bash
# Download D-NeRF dataset
ns-download-data dnerf

# Train D-NeRF on the default lego sequence
ns-train dnerf dnerf-data
```

## Splatfacto (Gaussian splatting) on a COLMAP / Nerfstudio dataset

Gaussian splatting method (`splatfacto`) running on a Nerfstudio‑format dataset (e.g. processed via `ns-process-data`).

```bash
# Example: process a COLMAP workspace into Nerfstudio format
ns-process-data colmap \
  --data PATH/TO/COLMAP_WORKSPACE \
  --output-dir data/processed/scene

# Train Splatfacto (Gaussian splatting)
ns-train splatfacto --data data/processed/scene
```

## NeuS / NeuS-facto with SDFStudio-style data

Surface reconstruction using SDF‑based methods from inside Nerfstudio. These share data conventions with SDFStudio.

```bash
# Example: train NeuS on an SDFStudio-format dataset
ns-train neus --data PATH/TO/SDFSTUDIO_SCENE

# Example: train NeuS-facto (NeuS with proposal sampling)
ns-train neus-facto --data PATH/TO/SDFSTUDIO_SCENE
```

## Quick reference

- Use `ns-train --help` to list all available methods, including third‑party plugins.
- Use `ns-train {method} --help` to inspect all configurable parameters for that method.
- Use `ns-train {method} {dataparser} --help` to inspect dataset‑specific options (e.g. `blender-data`, `dnerf-data`, `nerfstudio-data`, `phototourism-data`, `colmap`, etc.).

