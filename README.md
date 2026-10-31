# Sketch2Shape: from a 2D sketch to a 3D asset

This tool turns a rough sketch into a live 3D model that you can rotate and explore from every side. It helps people visualise an idea and share it with others.

**TechZephyr 4.0 · IIT Bhubaneswar.** Draw or upload a sketch, add an optional prompt, and get a **textured, validated `.glb` / `.obj` / `.stl`** with an interactive preview, a **Sketch Fidelity Score**, and mesh health stats.

> **Local-first, so free-API quotas can't break it.** Three tiers of geometry engine:
> 1. **Sketch-CAD (instant, exact):** lathe objects (mug, vase, bottle, lamp…) are *revolved* from the drawn profile and hollowed if they're containers, with handles added as tubes. Flat objects (sign, star, gear…) are *extruded* with holes kept. PBR materials come from the prompt.
> 2. **Local AI:** TripoSR (MIT) runs on your own CPU/GPU. Weights download once; no keys and no quota.
> 3. **Cloud boost (optional):** FLUX.1 Kontext + TRELLIS on HF ZeroGPU give the best texture quality while the daily quota lasts. Quota and auth failures are remembered and skipped.
>
> If everything else fails, offline inflation still returns a valid, coloured, watertight GLB in about 3 s.

### What makes it stand out
| Feature | What it does |
|---|---|
| **🧠 AI-CAD: sketch → idealised model** | A vision LLM (Gemini) *reads your sketch*: it identifies every part (cup, lid, stem, base, two handles…), measures their proportions from the drawing, and rebuilds it as **clean CAD**: mirrored handles, smooth lathe profiles, uniform thickness, per-part colours. Messy hand drawings and phone photos become polished products, and an honest *front-view fidelity* score is shown. **💬 Refine this model in chat** then continues in the AI Designer ("make the handles thinner"). Auto uses it when `GEMINI_API_KEY` is set; organic subjects (animals, people) go to TripoSR |
| **💬 AI Designer (chat)** | Describe *anything* ("a wooden chair with 4 legs, no armrests and a backrest", "a red sports car", "a snowman with a carrot nose"). An LLM (Gemini) plans it as parts with sizes, positions, colours and materials, and the app builds a multi-material GLB in about 3–15 s. Then **keep talking**: "make the legs thinner", "add armrests", "paint the seat red", "make it twice as tall". Each turn edits the current plan, keeps everything else, and saves a numbered version with **Undo**. Offline, the first request falls back to the templates |
| **Prompt-only creation** | No sketch needed for common objects: "a brown bottle with a green holder", "a white teapot with a gold lid and blue spout". 33 parametric templates (bottle, mug, cup, vase, wine glass, bowl, jar, lamp, teapot, kettle, jug, trophy, chess pawn, candle, star, heart, gear, shield, coin…) are built as exact geometry, so handles and spouts stay correct. **✏️ Draw it for me** puts the template on the canvas to edit first |
| **Prompt refines the drawing** | The drawing sets the shape and words adjust it: `cylindrical`, `hexagonal`, `domed`, `pointed`, `ribbed`, `tall`, `slim`, `thick/thin/no handle`, `solid/hollow`, `glossy/matte` |
| **Per-part materials** | "a red mug with a **black handle**", "a bottle with a **gold cap**", "a cup with a **gold rim**", "... **wooden lid**": separate PBR materials (body / cap / rim / base / handle) in one GLB |
| **Any colour** | All 148 CSS colour names, multi-word names ("sky blue", "navy blue"), `light`/`dark`/`pale`/`deep` shades, and `#hex` codes; 26 materials (gold, chrome, leather, marble, terracotta…) |
| **Offline auto-caption** | Local CLIP names a blank-prompt sketch ("a coffee mug", 8/8 on our test set, 0.13 s), so Auto still picks the right engine with no internet |
| **"Understood as" panel** | Shows the object, engine, shape words and colour swatches the system took from your prompt |
| **Sketch Fidelity Score** | Best-of-N 2D candidates ranked by silhouette IoU + edge agreement with your lines |
| **Never fails** | Provider router with retries, persisted quota cooldowns, and an offline last resort |
| **Proof of real 3D** | Interactive viewer, a 360° turntable GIF, watertight check, and re-open validation of the exported GLB |

---

## 1. Quick start (CPU laptop, about 5 min)

```bash
git clone https://github.com/pranayharshithsalla/sketch2shape.git
cd sketch2shape
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/setup_local_models.py   # one-time: TripoSR source + 1.7 GB weights (local AI tier)
cp .env.example .env        # optional: add keys (see §4). With no keys the local tiers run.
python app.py               # open http://localhost:7860
```

The first run downloads the rembg segmentation model (`isnet-general-use`, 179 MB, cached in `~/.rembg`).

**CLI / batch (no UI):**
```bash
python batch_run.py                 # all sketches in examples/, best available providers
python batch_run.py --offline       # key-free local path
python batch_run.py my_sketch.png --prompt "a blue ceramic vase"
```
Each run writes `outputs/<run>/` with every intermediate image, `model.glb`, `model_obj.zip` (obj+mtl+texture), `model.stl`, `turntable.png` and `run.json` (prompt, providers, scores, timings, mesh report). The UI also saves `spin.gif`.

**Tests (offline, about 10 s):**
```bash
pip install pytest
python -m pytest -q tests
```

**Key-free GPU path:** open [`colab/sketch2shape.ipynb`](colab/sketch2shape.ipynb) on a free Colab T4. It runs SD1.5 + ControlNet scribble + TripoSR locally on the GPU, with no tokens.

## 2. How to use the app
0. **Want to just describe it?** Open the **💬 AI Designer** tab, type what you want, then refine it turn by turn in plain language (needs `GEMINI_API_KEY`; free tier).
   **No sketch, no key?** Just type a prompt such as `a brown bottle with a green holder` and press Generate, or press **✏️ Draw it for me** to get an editable starting sketch.
1. **Draw** on the canvas (brush / eraser / clear on the toolbar), **or** open **Upload** and drop a PNG/JPG. Photos of paper sketches work, since background normalisation is built in. Upload takes priority if both are given.
2. Optionally type a prompt ("a red ceramic mug") and/or pick a **material preset**. Leave it blank and the sketch is **auto-captioned**. The object word in the prompt drives **Auto** engine selection (e.g. "vase" → revolve, "star" → extrude), or you can pick the **Geometry engine** yourself. In Sketch-CAD the drawing sets the shape and **prompt shape words refine it**: `cylindrical`, `hexagonal`/`square`/`octagonal`, `domed`/`pointed top`, `ribbed`, `tall`/`short`, `wide`/`slim`, `thick`/`thin`/`no handle`, `solid`/`hollow`, `glossy`/`matte`. The log line "prompt shape modifiers" shows which were applied.
3. Click **Generate 3D**. You'll see the preprocessing steps, the candidate images ranked by fidelity, the 3D preview, mesh stats, per-stage timings, and download buttons.
4. Under **Advanced** you can set the number of candidates, force a provider, skip the generative 2D step, and set the seed.

## 3. Architecture

```
sketch (canvas / upload)
  │ 1 preprocess      white-composite (fixes transparent canvas), paper normalisation, invert dark bg,
  │                   CLAHE, Otsu, despeckle, crop+pad to square 512, stroke-width normalise, filled silhouette
  │ 2 auto-caption    Gemini / CLIP zero-shot (only when the prompt is empty)
  │ 3 sketch→image    N candidates in parallel: Gemini image │ ControlNet-scribble │ FLUX Kontext │ local render
  │ 4 segment         rembg (IS-Net) → RGBA, re-centred
  │ 5 fidelity        0.6·silhouette IoU + 0.4·edge chamfer agreement vs the sketch → keep best-of-N
  │ 6 image→3D        TRELLIS │ Stable Fast 3D │ TRELLIS@Replicate │ TripoSR (GPU) │ local inflation (CPU)
  │ 7 repair          drop floaters, dedupe/degenerate faces, fix normals, fill holes, centre + unit-scale
  ▼ 8 export+verify   .glb / .obj(+mtl,tex) / .stl; GLB is re-opened to check verts>0 and colour kept
```

| File | Role |
|---|---|
| `app.py` | Gradio UI (Sketchpad, upload, prompt, Model3D, galleries, stats, downloads) |
| `src/pipeline.py` | Orchestrates the stages, per-stage timings; every error names its stage |
| `src/preprocess.py` | Input normalisation + structural cues (clean lines, scribble map, silhouette) |
| `src/sketch2image.py` | Auto-caption, prompt building, image provider router |
| `src/segment.py` | rembg background removal (+ threshold fallback) |
| `src/fidelity.py` | Sketch Fidelity Score + best-of-N |
| `src/image2mesh.py` | 3D provider router + offline silhouette-inflation reconstructor |
| `src/export.py` | trimesh repair, validation, multi-format export, health report |
| `src/router.py` | Retry with exponential backoff; quota/auth errors fail fast to the next provider |
| `src/render.py` | OpenGL-free renderer: turntable PNG + 360° GIF (proof of real 3D) |
| `src/sketch3d.py` | Sketch-CAD: revolve / extrude / inflate, prompt shape modifiers, per-part split |
| `src/prompt.py` | Prompt understanding: CSS colours, shades, hex, materials, per-part colours |
| `src/ai_assembly.py` | LLM part planner (Gemini) + primitive builder (box, cylinder, sphere, cone, capsule, torus, lathe, tube) |
| `src/designer.py` | Conversational sessions: plan, edit, version, undo |
| `src/templates.py` | Prompt-only creation: parametric template sketches with exact body / handle / spout geometry |
| `src/caption_local.py` | Offline CLIP auto-caption for blank prompts |
| `src/triposr_local.py` | TripoSR on local CPU/GPU (mcubes shim, transformers-5 weight remap) |
| `tests/` | Offline smoke tests (pytest) |
| `src/gpu_local.py` | Optional CUDA providers (SD1.5+ControlNet, TripoSR) for Colab |

### Why these choices
- **Fidelity over photorealism.** The generative step only adds material and appearance. Several candidates are generated and the one whose silhouette and edges best match *your* lines is kept. A judge's fresh sketch is answered with a number, not just a picture.
- **Router, not a single model.** Free hosted queues and quotas fail unpredictably during a live demo. Each provider is tried in order, and quota or auth errors skip straight to the next one.
- **Offline fallback.** Silhouette inflation (distance-transform height field, h = √(d(2R−d)), mirrored, signed-distance side walls, marching cubes, Taubin smoothing, colours from the image) is not state of the art. It does guarantee a recognisable, watertight, coloured mesh with zero dependencies on the network.

## 4. Models, APIs and licences (disclosure)

| Component | Used for | Access | Licence |
|---|---|---|---|
| Gemini 2.5 Flash Image / Flash | sketch→image, captioning | `GEMINI_API_KEY` (free tier) | Google API ToS |
| FLUX.1 Kontext [dev] | sketch→image (primary) | official HF ZeroGPU Space (`HF_TOKEN`), or HF Inference Providers | FLUX.1 [dev] non-commercial |
| SD 1.5 + ControlNet v1.1 scribble | sketch→image | Replicate `jagilley/controlnet-scribble` or local GPU (Colab) | CreativeML OpenRAIL-M / Apache-2.0 |
| TRELLIS (image-large) | image→3D (textured) | HF Space `trellis-community/TRELLIS` (`HF_TOKEN`) or Replicate `firtoz/trellis` | MIT |
| Stable Fast 3D | image→3D (textured, about 10 s) | HF Space `stabilityai/stable-fast-3d` (`HF_TOKEN`) | Stability AI Community |
| TripoSR | image→3D (vertex colour), unlimited local AI tier | **local CPU or GPU** (weights from HF Hub, no token) | MIT |
| Sketch-CAD (ours) | revolve / extrude / inflate straight from the sketch | local CPU | — |
| rembg / IS-Net | background removal | local CPU | MIT / Apache-2.0 |
| CLIP ViT-B/32 | offline auto-caption (zero-shot over 100 object labels) | local CPU (weights from HF Hub, about 600 MB) | MIT |
| trimesh, scikit-image, OpenCV, Gradio | mesh, marching cubes, CV, UI | local | MIT / BSD / Apache-2.0 |

## 5. Hardware, runtime, limitations
- **Dev machine:** Intel Core Ultra laptop CPU (Windows 11, Python 3.13), **no GPU**.
- **Offline (`--offline`, no network):** Sketch-CAD objects take about **1 s**. TripoSR on CPU takes about **25–27 s** per model, plus a one-time 20 s model load that the app preloads at startup. See `outputs/benchmark_offline.md`. The first run also downloads rembg (179 MB) and TripoSR (1.7 GB) once.
- **Hosted path (measured):** FLUX.1 Kontext Space about 30–35 s per candidate, TRELLIS Space about 63 s, **about 110 s end to end**. See [REPORT.md](REPORT.md).
- **HF ZeroGPU quota:** anonymous users get **0 s**, so set `HF_TOKEN`. A free account covers only about 3 full runs a day; PRO gives 25 min/day. You can put several comma-separated tokens in `HF_TOKEN` and they rotate automatically.
- A single view cannot specify hidden surfaces. Back sides are *plausible*, not exact.
- The offline reconstructor makes "inflated" shapes. Concave or open structures (cup interiors, holes in handles) are filled in.
- On the local render path the Fidelity Score is about 0.98 by construction, since that render *is* the sketch silhouette. It is only discriminative for generative candidates.
- Meshes are for visualisation and further editing, not manufacturing-ready topology.

## 6. Troubleshooting
| Symptom | Fix |
|---|---|
| `ZeroGPU quota` in the log | Add `HF_TOKEN` to `.env`, wait for quota reset, or rely on fallback |
| Providers show only `local` | `.env` missing or keys empty; restart `app.py` after editing |
| rembg download fails (offline) | Pipeline falls back to white-background thresholding automatically |
| Black canvas / empty-sketch error | Draw darker or thicker strokes; transparent canvases are composited onto white |
| Port in use | `python app.py --port 7861`; `--share` gives a public link |

## 7. Repo layout
`examples/` input sketches (`scripts/make_examples.py` draws the synthetic set; hand-drawn ones go here too) · `outputs/` sample results (labelled **pre-generated**) · `colab/` GPU notebook · `REPORT.md` benchmark and design report.
