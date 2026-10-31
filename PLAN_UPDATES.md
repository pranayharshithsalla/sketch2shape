# Plan updates (31 Oct 2026, ~H1)

The original plan's strategy still holds: a reliable pipeline, best-of-N with a Fidelity Score, a provider router, and the Colab path. What changed after testing the live endpoints:

| Plan assumption | Reality today | Update |
|---|---|---|
| TripoSR is the hosted "workhorse" | HF Space `stabilityai/TripoSR` is **down** (RUNTIME_ERROR) | Hosted primary = **TRELLIS** (MIT, textured, best quality), then **Stable Fast 3D**, then TRELLIS on Replicate. TripoSR stays on the Colab GPU path. |
| HF Spaces usable without keys | Anonymous ZeroGPU quota = **0 s** | **HF_TOKEN is mandatory** for hosted 3D. Each teammate should create a token. PRO is worth it if available (more GPU-minutes). |
| ControlNet via HF Space | `hysts/ControlNet-v1-1` is **BUILD_ERROR** | 2D providers: **Gemini image model** (free key, best at keeping sketch structure) → Replicate ControlNet scribble → FLUX Kontext via HF Inference → offline render. |
| Hunyuan3D as an extra fallback | Space throws `NameError` | Dropped. |
| "Always returns a .glb" relied on some API being up | Nothing guaranteed it | **New offline CPU reconstructor** (silhouette inflation + marching cubes, coloured, watertight, about 3 s). The demo can never end without a mesh, even with no Wi-Fi. |
| Hour-4 gate on InstantMesh | Not needed: router + offline path already close the loop at H1 | Spend H1–4 on keys and measuring hosted quality instead. |

**Status at H1:** the end-to-end pipeline, UI, best-of-N, repair/validation/export, batch benchmark, Colab notebook and docs are all built and tested on the offline path.

**Next:** add keys → run `batch_run.py` on the hosted path → tune the fidelity weights → hand-drawn sketches from the teammate → fill the REPORT hosted table.

# Update 2 (31 Oct 2026, ~H2): local-first

**Why:** on day one, free quotas died within an hour. Gemini image = 0, HF Inference = 402, and HF ZeroGPU allowed about 3 runs/day. A demo that depends on them will fail in front of the judges.

**Key insight:** only *hosted GPU time* is rationed. **Downloading open weights is unlimited**, so models that run on our own hardware have no quota.

**New engine order (Auto):**
1. **Sketch-CAD:** revolve (lathe objects, hollow containers, handles as tubes) and extrude (flat objects). Instant, exact fidelity to the drawn lines, clean watertight topology, PBR materials. No network.
2. **TRELLIS cloud** while quota lasts (best texture). Quota and auth errors put a provider in cooldown, so later runs don't waste 20 s on it.
3. **TripoSR local** (MIT): the unlimited AI workhorse on CPU/GPU.
4. Offline inflation: the last resort that never fails.

**Blender:** considered as a core stage and kept *optional*. Everything we need in the pipeline (revolve, extrude, bevel, smooth, remesh, export) is pip-installable through trimesh/shapely, which protects the 15 reproducibility points. Blender (`pip install bpy`) is the right tool for a post-processing "finish" step: Cycles turntable renders, procedural materials, a `.blend` export.

# Update 3 (31 Oct 2026, ~H4): standout features

- **Prompt shape modifiers:** the drawing sets the base shape and words refine it (cylindrical, hexagonal, domed, pointed, ribbed, tall, slim, handle size, solid/hollow, glossy/matte).
- **Per-part materials:** body / cap / rim / base / handle each get their own colour and material from the prompt, and are exported as separate PBR materials in one GLB.
- **Colour understanding:** 148 CSS colour names, multi-word names, shades, hex codes, and 26 materials. Specific materials beat generic ones ("gold metal" gives gold).
- **Offline CLIP auto-caption:** a blank prompt still picks the right engine with no network (8/8 on the test sketches).
- **UI:** an "Understood as" panel with colour swatches, a 360° turntable GIF rendered after the model is shown, and examples that pair a sketch with a prompt.
- **Robustness:** more accurate lathe-axis estimation (handles no longer shift it), open strokes kept, flat tops and bases from perspective ellipses, captions capped at 15 s, quota cooldowns persisted to disk, and an `OUTPUT_DIR` setting for OneDrive.
- **Tests:** `tests/test_pipeline.py` has 6 offline smoke tests.

# Update 4 (31 Oct 2026, ~H5): prompt-only creation

- `src/templates.py` has 33 parametric objects (25 lathe profiles plus flat outlines). With no sketch, the object named in the prompt is built from **exact geometry**: body profile, handle curve and spout line. Nothing is re-detected from pixels, so teapots and kettles keep both their handle and their spout.
- Everything else still applies: per-part colours (handle / spout / cap / rim / base), shape words, materials.
- UI: **✏️ Draw it for me** puts the template on the canvas. If it is left unedited, the exact template path is used; if the user draws on it, the normal sketch path runs.
- Fixed: glTF colours are now written in linear space, so viewers show true colours instead of washed-out ones.
- Tests: 10 offline tests, including prompt-only creation and a clear error for unknown objects with no sketch.
