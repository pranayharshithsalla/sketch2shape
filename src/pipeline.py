"""Orchestrates the 8 stages, records per-stage timings, and names the failing stage
in every error."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import config
from .export import finalize
from .fidelity import best_of
from .image2mesh import image_to_mesh
from .preprocess import PreprocessError, Sketch, preprocess
from .templates import FLAT as FLAT_TEMPLATES
from .templates import available_names, make_template
from .segment import remove_bg, rgba_on_white
from .sketch2image import auto_caption, build_prompt, generate_candidates, names_object
from .sketch3d import build as build_cad
from .sketch3d import choose_mode

log = logging.getLogger(__name__)


class StageError(RuntimeError):
    def __init__(self, stage: str, err: Exception):
        super().__init__(f"[{stage}] {err}")
        self.stage = stage


@dataclass
class Result:
    run_dir: Path
    glb: str = ""
    obj_zip: str = ""
    stl: str = ""
    prompt: str = ""
    caption: str = ""
    image_provider: str = ""
    mesh_provider: str = ""
    scores: list = field(default_factory=list)
    best: int = -1
    timings: dict = field(default_factory=dict)
    report: dict = field(default_factory=dict)
    logs: list = field(default_factory=list)
    images: dict = field(default_factory=dict)   # name -> path, for the UI
    candidates: list = field(default_factory=list)


def _save(arr: np.ndarray, path: Path) -> str:
    Image.fromarray(arr).save(path)
    return str(path)


def run(sketch_input, prompt: str = "", material: str = "auto", n_candidates: int | None = None,
        image_provider: str = "auto", mesh_provider: str = "auto", skip_2d: bool = False,
        seed: int = 42, run_name: str | None = None, progress=None) -> Result:
    n = n_candidates or config.NUM_CANDIDATES
    run_dir = config.OUTPUTS / (run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    res = Result(run_dir=run_dir)
    t_all = time.time()

    def stage(name, frac):
        if progress:
            progress(frac, desc=name)
        res.logs.append(f"▶ {name}")
        return time.time()

    def done(name, t0):
        res.timings[name] = round(time.time() - t0, 2)
        res.logs.append(f"  ✓ {name} ({res.timings[name]}s)")

    # 1. Preprocess
    t = stage("Preprocess", 0.02)
    template, tpl = None, None
    try:
        sk = preprocess(sketch_input)
    except PreprocessError as e:
        # Prompt-only creation: no sketch, so draw a template for the object the prompt names
        tpl = make_template(prompt) if prompt.strip() else None
        if tpl is None:
            hint = (" Or type a prompt naming one of: " + ", ".join(available_names())) if prompt.strip() else ""
            raise StageError("Preprocess", PreprocessError(f"{e}{hint}"))
        template = tpl["name"]
        gray = cv2.cvtColor(tpl["img"], cv2.COLOR_RGB2GRAY)
        sk = Sketch(clean=gray, scribble=((gray < 128) * 255).astype(np.uint8), mask=tpl["body"],
                    rgb=tpl["img"], stroke_px=4)
        res.logs.append(f"  no sketch given: built a '{template}' template from the prompt")
    except Exception as e:
        raise StageError("Preprocess", e)
    res.images["sketch_clean"] = _save(sk.clean, run_dir / "1_sketch_clean.png")
    res.images["sketch_mask"] = _save((sk.mask * 255).astype(np.uint8), run_dir / "1_sketch_mask.png")
    done("Preprocess", t)

    # 2-3. Caption + sketch-guided generation (best-of-N)
    if skip_2d:
        image_provider = "local"
    t = stage("Caption", 0.08)
    # caption when the prompt does not name the object (blank, or only "make it gold")
    res.caption = "" if names_object(prompt) else auto_caption(sk)
    res.prompt = build_prompt(prompt, res.caption or "an object", material)
    res.logs.append(f"  prompt: {res.prompt[:140]}…")
    done("Caption", t)

    # Sketch-CAD branch: lathe / flat objects (or an explicitly chosen CAD mode) are built
    # straight from the silhouette: exact fidelity, instant, no network or quota.
    cad_mode = mesh_provider if mesh_provider in ("revolve", "extrude", "inflate") else None
    if mesh_provider in ("auto", "sketch_cad"):
        cad_mode = choose_mode(sk.mask, f"{prompt} {res.caption}")
        if template:  # templates are drawn for exactly these engines
            cad_mode = "extrude" if template in FLAT_TEMPLATES else "revolve"
    if cad_mode:
        t = stage(f"Sketch-CAD: {cad_mode}", 0.3)
        try:
            raw, res.mesh_provider, lg = build_cad(sk.mask, run_dir, cad_mode, f"{prompt} {res.caption}", material,
                                                   strokes=sk.scribble > 0,
                                                   known_parts=tpl["parts"] if template else None)
            res.logs += ["  " + l for l in lg]
        except Exception as e:
            raise StageError("Sketch-CAD", e)
        res.image_provider = "none (geometry built directly from the sketch)"
        res.scores, res.best = [{"score": 1.0, "iou": 1.0, "edge": 1.0}], 0
        res.images["chosen"] = res.images["sketch_mask"]
        done("Sketch-CAD", t)
        return _finish(res, raw, run_dir, material, t_all, stage, done, progress)

    t = stage(f"Generate {n} candidate image(s)", 0.12)
    try:
        cands, res.image_provider, lg = generate_candidates(sk, res.prompt, n, image_provider, seed)
        res.logs += ["  " + l for l in lg]
    except Exception as e:
        raise StageError("Sketch-to-image", e)
    done("Sketch-to-image", t)

    # 4. Background removal
    t = stage("Remove background", 0.40)
    try:
        cands_rgba = [remove_bg(c) for c in cands]
    except Exception as e:
        raise StageError("Background removal", e)
    done("Background removal", t)

    # 5. Fidelity scoring
    t = stage("Score fidelity", 0.48)
    res.best, res.scores = best_of(sk.mask, sk.scribble, cands_rgba)
    if res.scores[res.best]["score"] == 0.0:
        # every candidate lost the object: fall back to the offline render of the sketch
        res.logs.append("  all candidates empty after background removal; using local render")
        from .sketch2image import gen_local
        cands_rgba = [remove_bg(gen_local(sk, res.prompt, seed))]
        res.best, res.scores = best_of(sk.mask, sk.scribble, cands_rgba)
    for i, (c, s) in enumerate(zip(cands_rgba, res.scores)):
        p = _save(rgba_on_white(c), run_dir / f"2_candidate_{i}_{s['score']:.2f}.png")
        res.candidates.append((p, f"#{i}  fidelity {s['score']:.2f}" + ("  ★ best" if i == res.best else "")))
    chosen = cands_rgba[res.best]
    res.images["chosen"] = _save(chosen, run_dir / "3_chosen_rgba.png")
    res.logs.append(f"  scores: {[s['score'] for s in res.scores]} -> best #{res.best}")
    done("Fidelity", t)

    # 6. Image to 3D
    t = stage("Reconstruct 3D (this is the slow step)", 0.55)
    try:
        prov = "auto" if mesh_provider == "sketch_cad" else mesh_provider
        raw, res.mesh_provider, lg = image_to_mesh(chosen, run_dir, prov, seed)
        res.logs += ["  " + l for l in lg]
    except Exception as e:
        raise StageError("Image-to-3D", e)
    done("Image-to-3D", t)

    return _finish(res, raw, run_dir, material, t_all, stage, done, progress)


def _finish(res: Result, raw: str, run_dir: Path, material: str, t_all: float, stage, done, progress) -> Result:
    # 7-8. Repair, validate, export
    t = stage("Repair, validate, export", 0.92)
    try:
        out = finalize(raw, run_dir)
    except Exception as e:
        raise StageError("Export", e)
    res.glb, res.obj_zip, res.stl, res.report = out["glb"], out["obj_zip"], out["stl"], out["report"]
    done("Export", t)

    res.timings["total"] = round(time.time() - t_all, 2)
    (run_dir / "run.json").write_text(json.dumps({
        "prompt": res.prompt, "caption": res.caption, "material": material,
        "image_provider": res.image_provider, "mesh_provider": res.mesh_provider,
        "scores": res.scores, "best": res.best, "timings": res.timings, "report": res.report,
        "logs": res.logs}, indent=2, ensure_ascii=False), encoding="utf-8")
    if progress:
        progress(1.0, desc="Done")
    return res
