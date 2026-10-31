# Sketch2Shape: technical report

> **Status note (2026-11-01):** this report records our **earlier experiments and design decisions**. Items listed below as future work have since been implemented: the post-3D silhouette check, structural accuracy (IoU, thin-part recall, holes), carve-and-add refinement and procedural strings. See README §8 and §8b for the current implementation and measured results.

## Objective
Single-view 2D sketch (+ optional text) → textured, exportable 3D asset that keeps the sketch's structure.

## Pipeline and design decisions
See README §3. Key decisions:
1. **Transparent-canvas fix.** Gradio's Sketchpad returns RGBA with a transparent background. Converting naively to RGB turns it black and breaks thresholding, so we alpha-composite onto white first.
2. **Structure cues.** Otsu strokes on a background-normalised image, a stroke-width-normalised scribble map (ControlNet convention: white on black), and a filled silhouette (morphological close, external-contour fill, convex-hull fallback for open sketches).
3. **Best-of-N with the Sketch Fidelity Score.** `0.6·IoU(sketch silhouette, rembg alpha) + 0.4·(1 − min(chamfer(sketch strokes, Canny(candidate))/20, 1))`. Both are cropped to their bounding boxes first, so scale and offset are ignored. Candidates are generated in parallel, and only the winner goes to 3D, so there is no extra 3D cost.
4. **Router with typed failure handling.** Transient errors are retried with backoff. Quota, auth and "space down" errors skip to the next provider at once. Each run logs which provider produced each stage.
5. **Validation.** The exported GLB is re-loaded. We check non-empty geometry and report watertightness, components, and whether colour or texture survived.

## Provider status (checked 31 Oct 2026, 13:00 IST)
| Endpoint | Status |
|---|---|
| HF Space `trellis-community/TRELLIS` | up; **needs HF_TOKEN** (anonymous ZeroGPU quota = 0 s) |
| HF Space `stabilityai/stable-fast-3d` | up, but every call raises an internal error → removed from the default order |
| HF Space `black-forest-labs/FLUX.1-Kontext-Dev` | **works** with HF_TOKEN, about 30 s per image (primary 2D provider) |
| HF Space `TencentARC/InstantMesh` | up (multi-step API, not wired) |
| HF Space `tencent/Hunyuan3D-2` | returns `NameError`: broken |
| HF Space `stabilityai/TripoSR` | RUNTIME_ERROR: down → TripoSR is used locally on GPU (Colab) instead |
| HF Space `hysts/ControlNet-v1-1` | BUILD_ERROR: down → ControlNet via Replicate or local GPU |

## Benchmark: offline path (no keys, CPU)
Hardware: Intel Core Ultra (Family 6 Model 170) laptop, Windows 11, Python 3.13, no GPU. Synthetic line sketches in `examples/`.

`python batch_run.py --offline`: Sketch-CAD + local TripoSR, zero network.

| sketch | prompt | engine | verts | faces | watertight | colour | 3D s | total s |
|---|---|---|---|---|---|---|---|---|
| mug | a red ceramic mug | **Sketch-CAD revolve** (hollow + handle) | 29,930 | 59,852 | ✓ | PBR material | — | **1.0** |
| star | a gold star | **Sketch-CAD extrude** | 104 | 204 | ✓ | PBR metallic | — | **0.1** |
| chair | a wooden chair | **TripoSR local (CPU)** | 26,546 | 53,088 | ✓ | vertex | 24.4 | 26.7 |
| teddy | a brown teddy bear | **TripoSR local (CPU)** | 12,244 | 24,484 | ✓ | vertex | 25.3 | 26.9 |
| car | a red car | **TripoSR local (CPU)** | 16,218 | 32,432 | ✓ | vertex | 43.4* | 48.2* |

\*includes the one-time TripoSR model load (about 20 s; the app preloads it at startup). TripoSR at marching-cubes resolution 128 takes 18.6 s, and at 192 (the default) 26.6 s.

Earlier inflation-only baseline: about 3 s per sketch, but blobby shapes. TripoSR gives real 3D structure, for example a chair with seat depth and four legs.

**TripoSR compatibility notes:** torchmcubes, which needs a C++ build, is replaced by a scikit-image shim. The checkpoint's ViT layer names are remapped for transformers ≥5. The checkpoint is loaded with `weights_only=True`, so it can only contain tensors and no pickled code.

## Benchmark: hosted path (31 Oct 2026, free HF account)
| sketch | prompt | image | 3d | fidelity (IoU) | verts / faces | colour | 2D s | 3D s | total s |
|---|---|---|---|---|---|---|---|---|---|
| mug | (blank) | FLUX.1 Kontext Space | **TRELLIS Space** | 0.666 (0.77) | 12,437 / 22,021 | 1024² PBR texture | 34.5 | 63.3 | 108.7 |
| car | a red car | FLUX.1 Kontext Space | local (quota exhausted) | 0.824 (0.83) | 23,208 / 46,412 | vertex | 31.9 | 40.1* | 74.2 |
| teddy | a brown teddy bear | FLUX.1 Kontext Space | local (quota exhausted) | 0.809 (0.84) | 41,840 / 83,676 | vertex | 29.0 | 45.5* | 77.4 |
| chair | a wooden chair | FLUX.1 Kontext Space | local (quota exhausted) | 0.328 (0.36) | 23,234 / 46,468 | vertex | 33.6 | 38.9* | 76.9 |

\*includes about 40 s spent trying TRELLIS / SF3D / Replicate before the router fell back to `local`, which shows the fallback working.

**Finding: ZeroGPU quota is the main runtime risk.** One TRELLIS call reserves 120 s of GPU, and a free account's daily quota covered about 3 runs. Mitigations: HF PRO (25 min/day), several teammate tokens in `HF_TOKEN` (rotated automatically on quota errors), Replicate as a paid backup, and the offline path as the last resort.

Gemini image models returned 429 (free-tier image limit 0) and Replicate rejected the supplied token (401), so neither appears in these runs.

## Limitations
- The hidden geometry is hallucinated by the image-to-3D model, or for the offline path it is a symmetric inflation.
- Free hosted endpoints have queues and quotas. Runtime varies between runs, and the router hides failures but cannot make them faster.
- The offline reconstructor fills through-holes and concavities.
- The fidelity score measures 2D agreement of the chosen image, not of the final 3D mesh. A post-3D silhouette check was future work at the time; it is now implemented (README §8b).
