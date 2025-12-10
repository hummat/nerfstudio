# Nerfstudio methods and representations

This document mirrors the SDFStudio method overview and focuses on the methods exposed through `ns-train` in this repository.

- [Methods](#methods)
- [Representations](#representations)
- [Supervision](#supervision)

# Methods

Nerfstudio implements multiple radiance field, surface, and Gaussian splatting methods behind a single CLI (`ns-train`). The main differences between methods lie in:

- the underlying scene representation (neural radiance field, SDF-based surface model, or explicit Gaussians),
- the sampling strategy and encodings,
- and what types of supervision they support (RGB, depth, semantics, generative priors, etc.).

For detailed explanations of individual methods, see the existing `nerfology/methods/*.md` pages. Here we provide a compact registry-style overview that is convenient when choosing a method or reading `method_configs.py`.

**Note:** All commands use the `ns-` prefix (e.g., `ns-train`, `ns-render`, `ns-export`, `ns-viewer`).

## Method registry overview

The table below summarizes the built‑in methods configured in `nerfstudio/configs/method_configs.py`, with their primary implementation file and a pointer to the corresponding documentation or paper.

| Method | Code (primary) | Paper / docs |
| --- | --- | --- |
| `nerfacto` | `nerfstudio/models/nerfacto.py` | [Nerfstudio: A Modular Framework for Neural Radiance Field Development](https://arxiv.org/abs/2302.04264) |
| `nerfacto-big` | `nerfstudio/models/nerfacto.py` | Larger Nerfacto variant; see [Nerfstudio paper](https://arxiv.org/abs/2302.04264) |
| `nerfacto-huge` | `nerfstudio/models/nerfacto.py` | Largest Nerfacto variant; see [Nerfstudio paper](https://arxiv.org/abs/2302.04264) |
| `depth-nerfacto` | `nerfstudio/models/depth_nerfacto.py` | Depth‑supervised Nerfacto; see [Nerfstudio paper](https://arxiv.org/abs/2302.04264) |
| `instant-ngp` | `nerfstudio/models/instant_ngp.py` | [Instant Neural Graphics Primitives with a Multiresolution Hash Encoding](https://arxiv.org/abs/2201.05989) |
| `instant-ngp-bounded` | `nerfstudio/models/instant_ngp.py` | Bounded Instant‑NGP variant; see [Instant‑NGP paper](https://arxiv.org/abs/2201.05989) |
| `mipnerf` | `nerfstudio/models/mipnerf.py` | [Mip‑NeRF: A Multiscale Representation for Anti‑Aliasing Neural Radiance Fields](https://arxiv.org/abs/2103.13415) |
| `semantic-nerfw` | `nerfstudio/models/semantic_nerfw.py` | [Semantic‑NeRF](https://arxiv.org/abs/2103.15875) + [NeRF‑W](https://arxiv.org/abs/2008.02268) style semantics and transients |
| `vanilla-nerf` | `nerfstudio/models/vanilla_nerf.py` | [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934) |
| `tensorf` | `nerfstudio/models/tensorf.py` | [TensoRF: Tensorial Radiance Fields](https://arxiv.org/abs/2203.09517) |
| `dnerf` | `nerfstudio/models/vanilla_nerf.py` | [D‑NeRF: Neural Radiance Fields for Dynamic Scenes](https://arxiv.org/abs/2011.13961) |
| `phototourism` | `nerfstudio/models/nerfacto.py` | Nerfacto on the [PhotoTourism dataset (IMW 2020)](https://www.cs.ubc.ca/~kmyi/imw2020/data.html) |
| `generfacto` | `nerfstudio/models/generfacto.py` | Nerfstudio text‑to‑3D method (no separate paper; see `nerfology/methods/generfacto.md`) |
| `neus` | `nerfstudio/models/neus.py` | [NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi‑view Reconstruction](https://arxiv.org/abs/2106.10689) |
| `neus-facto` | `nerfstudio/models/neus_facto.py` | NeuS‑Facto: NeuS with proposal sampling (internal Nerfstudio variant, no dedicated paper) |
| `splatfacto` | `nerfstudio/models/splatfacto.py` | [3D Gaussian Splatting for Real‑Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) (“Splatfacto” implementation) |
| `splatfacto-big` | `nerfstudio/models/splatfacto.py` | Higher‑quality Splatfacto; see [3D Gaussian Splatting](https://arxiv.org/abs/2308.04079) |
| `splatfacto-mcmc` | `nerfstudio/models/splatfacto.py` | Splatfacto with MCMC densification; see [3D Gaussian Splatting](https://arxiv.org/abs/2308.04079) |

In addition to the built‑in methods above, Nerfstudio exposes a large set of external / third‑party methods through `nerfstudio/configs/external_methods.py` (e.g. K‑Planes, LERF, NeRFPlayer, BioNeRF, Zip‑NeRF, OpenNeRF, etc.). Those appear as separate `ns-train` methods once their corresponding plugin is installed; see:

- `docs/nerfology/methods/index.md` for an overview list,
- `docs/index.md` (“Third‑party Methods” section),
- and `nerfstudio/configs/external_methods.py` for their CLI method names and installation hints.

## Method categories

At a high level, the methods above fall into a few buckets:

- **Hash‑grid NeRFs (real‑time, recommended):** `nerfacto`, `nerfacto-big`, `nerfacto-huge`, `depth-nerfacto`, `instant-ngp`, `instant-ngp-bounded`, `phototourism`. These all use compact hash encodings together with modern sampling schemes (proposal networks or occupancy grids) for fast training and inference on real captures.
- **Classic NeRF variants (bounded / synthetic):** `vanilla-nerf`, `mipnerf`, `dnerf`, `tensorf`. These targets are best for baseline comparisons and synthetic datasets (Blender, D‑NeRF, etc.).
- **Semantic & structured supervision:** `semantic-nerfw` adds semantics and transient object handling; `depth-nerfacto` incorporates depth cues.
- **SDF‑based surface models:** `neus`, `neus-facto` use SDF fields and surface‑aware rendering. They share data conventions with SDFStudio via `SDFStudioDataParserConfig`.
- **Gaussian splatting:** `splatfacto`, `splatfacto-big`, `splatfacto-mcmc` operate on explicit 3D Gaussians (via `gsplat`) and typically train on full images instead of ray batches.
- **Generative models:** `generfacto` uses a text‑conditioned diffusion model for generative NeRFs.

When in doubt, start with `nerfacto` on new real‑world scenes and only move to more specialized methods when you need specific capabilities (semantics, dynamic scenes, SDF surfaces, generative priors, etc.).

# Representations

Nerfstudio intentionally keeps representation details modular. The main representation families you will encounter in `method_configs.py` and `nerfology` are:

- **Neural radiance fields (NeRF‑style):** Most methods (`nerfacto`, `instant-ngp`, `mipnerf`, `vanilla-nerf`, `tensorf`, `dnerf`, etc.) represent the scene as a volumetric field queried along rays and rendered via volume rendering.
- **SDF‑based surfaces:** `neus` and `neus-facto` use signed distance fields (SDFs) to represent surfaces, with specialized rendering (e.g. NeuS) to convert SDF values into alpha / density along the ray. In practice these are closer to SDFStudio’s surface pipeline than pure NeRFs.
- **Explicit Gaussian splats:** `splatfacto` variants store the scene as a set of 3D Gaussians with learnable position, scale, opacity, and appearance. Rendering is done by rasterizing Gaussians instead of evaluating an MLP per sample.
- **Latent generative fields:** `generfacto` uses a NeRF‑like representation augmented with generative priors (text prompts, diffusion guidance) to generate scenes from text.

The choice of representation impacts both quality and performance. For example:

- radiance fields are flexible and well‑studied but can be slower to render,
- splats are extremely fast to render and edit but produce explicit point‑cloud‑like outputs,
- SDFs give high‑quality meshes and normals but can be more sensitive to supervision and initialization.

# Supervision

Nerfstudio methods can be supervised with different types of signals depending on the dataparser, dataset, and model configuration:

- **RGB reconstruction (default):** Most methods optimize an RGB reconstruction loss over multi‑view images from a dataset.
- **Depth supervision:** `depth-nerfacto` and `instant-ngp` variants support depth supervision when the dataparser exposes depth maps (e.g. ARKitScenes).
- **Semantics:** `semantic-nerfw` uses per‑pixel semantic labels and transient masks to learn both appearance and a semantic field.
- **Normals / SDF priors:** SDF‑based methods (`neus`, `neus-facto`) support surface‑aware regularizers (e.g. Eikonal loss) via `SDFFieldConfig` and can be combined with SDFStudio datasets / priors.
- **Generative priors:** `generfacto` uses text prompts and diffusion guidance as supervision signals rather than a fixed image set; it does not rely on ground‑truth training images.

For more detailed descriptions of losses and supervision, see:

- `nerfstudio/model_components/losses.py` for the actual loss implementations,
- `docs/nerfology/model_components/index.md` and linked notebooks for component‑level visualization,
- and the individual method pages under `docs/nerfology/methods/`.
