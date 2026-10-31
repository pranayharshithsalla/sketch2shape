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

from . import ai_assembly, config
from .export import finalize
from .fidelity import best_of
from .rank import card_label, rank
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
    plan: dict | None = None  # AI-CAD part plan (lets the chat refine a sketch result)
    accuracy: dict | None = None  # structural accuracy report (IoU, thin recall, holes, history)
    raw_glb: str = ""             # model before carve/add refinement (for the before/after toggle)
    analysis: object = None       # InputAnalysis of the sketch


def _save(arr: np.ndarray, path: Path) -> str:
    Image.fromarray(arr).save(path)
    return str(path)


def run(sketch_input, prompt: str = "", material: str = "auto", n_candidates: int | None = None,
        image_provider: str = "auto", mesh_provider: str = "auto", skip_2d: bool = False,
        seed: int = 42, run_name: str | None = None, progress=None, pick: int | None = None,
        hole_overrides: dict | None = None, procedural: bool = True) -> Result:
    """Full pipeline. If the structural Accuracy stays below the target after refinement, the
    build is retried once with a different engine and the better result is returned."""
    kw = dict(material=material, n_candidates=n_candidates, image_provider=image_provider, skip_2d=skip_2d,
              seed=seed, progress=progress, pick=pick, hole_overrides=hole_overrides, procedural=procedural)
    r = _run_once(sketch_input, prompt, mesh_provider=mesh_provider, run_name=run_name, **kw)
    acc = (r.accuracy or {}).get("accuracy")
    if acc is None or acc >= config.ACCURACY_TARGET or not config.ACCURACY_RETRY:
        return r
    from .image2mesh import available as mesh_available
    alt = ("triposr_local" if "triposr_local" in mesh_available() and r.mesh_provider != "triposr_local"
           else "inflate" if r.mesh_provider != "inflate" else None)
    if alt is None:
        return r
    r.logs.append(f"▶ Accuracy {acc:.2f} < {config.ACCURACY_TARGET}: retrying with engine '{alt}'")
    try:
        r2 = _run_once(sketch_input, prompt, mesh_provider=alt, run_name=f"{r.run_dir.name}_retry", **kw)
    except Exception as e:
        r.logs.append(f"  retry failed: {str(e)[:160]}")
        return r
    acc2 = (r2.accuracy or {}).get("accuracy") or 0.0
    best, other = (r2, r) if acc2 > acc else (r, r2)
    best.accuracy["history"] = (other.accuracy or {}).get("history", []) + [
        {"step": f"backend switch: {r.mesh_provider} {acc:.2f} → {alt} {acc2:.2f}",
         "accuracy": best.accuracy["accuracy"]}] + best.accuracy.get("history", [])
    best.logs.append(f"  kept '{best.mesh_provider}' (accuracy {best.accuracy['accuracy']:.2f})")
    return best


def _run_once(sketch_input, prompt: str = "", material: str = "auto", n_candidates: int | None = None,
        image_provider: str = "auto", mesh_provider: str = "auto", skip_2d: bool = False,
        seed: int = 42, run_name: str | None = None, progress=None, pick: int | None = None,
        hole_overrides: dict | None = None, procedural: bool = True) -> Result:
    """pick: index of the candidate the user chose (None = auto, the top-ranked one)."""
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
        sk = preprocess(sketch_input, text=prompt, overrides=hole_overrides)
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
    res.analysis = sk.analysis
    res.procedural = procedural
    if sk.analysis is not None:
        from .analyze import overlay
        res.images["analysis"] = _save(overlay(sk.analysis, sk.clean), run_dir / "1_analysis.png")
        res.logs.append(f"  input analysis: {sk.analysis.summary()}")
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

    # AI-CAD branch: a vision LLM reads the sketch (+ prompt), identifies its parts and
    # proportions, and rebuilds it as clean, symmetric, multi-material CAD geometry.
    # Flat objects (star, gear, sign…) ARE their outline, so extrusion beats any re-plan.
    flat = choose_mode(sk.mask, f"{prompt} {res.caption}") == "extrude"
    want_ai = mesh_provider == "ai_cad" or (
        mesh_provider == "auto" and not template and not flat
        and not is_organic(f"{prompt} {res.caption}"))
    if want_ai and ai_assembly.available():
        t = stage("AI-CAD: AI reads the sketch and rebuilds it", 0.3)
        try:
            ok, png = cv2.imencode(".png", sk.clean)
            spec, model = ai_assembly.plan(prompt.strip(), sketch_png=png.tobytes())
            raw, part_logs = ai_assembly.build_plan(spec, run_dir, prompt, material)
            res.mesh_provider, res.image_provider = "ai_cad", f"none (sketch read by {model})"
            res.plan = spec
            res.logs.append(f"  AI-CAD ({model}): {spec.get('object', 'object')}, {len(spec['parts'])} parts")
            res.logs += ["    " + l for l in part_logs]
            fid = _front_fidelity(raw, sk)
            # Sketch-Fidelity auto-retry: re-plan with the score as feedback (max 2), keep the best
            from .designer import FIDELITY_RETRY_BELOW, MAX_FIDELITY_RETRIES
            for k in range(MAX_FIDELITY_RETRIES):
                if fid["score"] >= FIDELITY_RETRY_BELOW:
                    break
                fb = (f"{prompt.strip()}\n[Your previous model matched the sketch outline only "
                      f"{fid['score']:.0%}. Follow the drawn silhouette, proportions and part positions "
                      f"much more closely, seen from the same side as the drawing.]")
                try:
                    spec2, model2 = ai_assembly.plan(fb, sketch_png=png.tobytes())
                    retry_dir = run_dir / f"retry{k + 1}"
                    retry_dir.mkdir(exist_ok=True)
                    raw2, logs2 = ai_assembly.build_plan(spec2, retry_dir, prompt, material)
                    fid2 = _front_fidelity(raw2, sk)
                except Exception as e:
                    res.logs.append(f"  fidelity retry {k + 1} failed: {str(e)[:100]}")
                    break
                res.logs.append(f"  fidelity retry {k + 1}: {fid['score']:.0%} -> {fid2['score']:.0%}")
                if fid2["score"] > fid["score"]:
                    spec, model, raw, fid = spec2, model2, raw2, fid2
                    res.plan = spec
            res.scores, res.best = [fid], 0
            res.images["chosen"] = res.images["sketch_mask"]
            done("AI-CAD", t)
            return _finish(res, raw, run_dir, material, t_all, stage, done, progress)
        except Exception as e:  # never fail the run: fall back to the local engines
            res.logs.append(f"  AI-CAD unavailable ({str(e)[:160]}); using local engines")
            done("AI-CAD", t)

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
    t = stage("Rank candidates (sketch 50% · 3D-ready 30% · prompt 20%)", 0.48)
    res.best, res.scores = rank(sk.mask, sk.scribble, cands_rgba, prompt.strip() or res.caption)
    if pick is not None and 0 <= pick < len(cands_rgba):
        res.logs.append(f"  user picked candidate #{pick + 1} (auto-rank chose #{res.best + 1})")
        res.best = pick
    if res.scores[res.best]["score"] == 0.0:
        # every candidate lost the object: fall back to the offline render of the sketch
        res.logs.append("  all candidates empty after background removal; using local render")
        from .sketch2image import gen_local
        cands_rgba = [remove_bg(gen_local(sk, res.prompt, seed))]
        res.best, res.scores = rank(sk.mask, sk.scribble, cands_rgba, prompt.strip() or res.caption)
    for i, (c, s) in enumerate(zip(cands_rgba, res.scores)):
        p = _save(rgba_on_white(c), run_dir / f"2_candidate_{i}_{s['score']:.2f}.png")
        res.candidates.append((p, card_label(i, s, res.best)))
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


ORGANIC_WORDS = {"teddy", "bear", "dog", "cat", "bird", "fish", "horse", "cow", "pig", "rabbit", "duck",
                 "dragon", "dinosaur", "animal", "person", "man", "woman", "girl", "boy", "face",
                 "head", "character", "monster", "doll", "tree", "plant", "flower", "leaf", "fruit",
                 "apple", "banana", "mushroom", "elephant", "lion", "tiger", "giraffe", "penguin",
                 "turtle", "snake", "shark", "whale", "octopus", "frog", "owl", "butterfly"}


def is_organic(text: str) -> bool:
    """Organic subjects suit learned image-to-3D (TripoSR) better than part-based CAD."""
    import re
    return bool(set(re.findall(r"[a-z]+", (text or "").lower())) & ORGANIC_WORDS)


def _front_fidelity(glb: str, sk: Sketch) -> dict:
    """Honest fidelity for a finished model: render its silhouette from each of the four sides,
    score each against the sketch with the same metric used for 2D candidates, and keep the
    best-matching view (a sketch may show the front or the side of an object)."""
    try:
        from .fidelity import fidelity
        from .render import _load, render_view
        mesh = _load(glb)
        best = {"score": 0.0, "iou": 0.0, "edge": 0.0, "view": 0}
        for yaw in (0, 90, 180, 270):
            img = np.asarray(render_view(mesh, yaw, 256, pitch_deg=0)).astype(np.int16)
            fg = (np.abs(img - np.array([245, 245, 248])).sum(-1) > 12).astype(np.uint8) * 255
            f = fidelity(sk.mask, sk.scribble, np.dstack([img.astype(np.uint8), fg]))
            if f["score"] > best["score"]:
                best = {**f, "view": yaw}
        return best
    except Exception:
        return {"score": 0.0, "iou": 0.0, "edge": 0.0}


def _finish(res: Result, raw: str, run_dir: Path, material: str, t_all: float, stage, done, progress) -> Result:
    # 7. Structural accuracy: carve/add to the sketch (holes, thin parts, strings), auto-escalate
    if res.analysis is not None and res.analysis.mask.any():
        t = stage("Check holes & thin parts, refine to the sketch", 0.85)
        try:
            from .refine import ensure_accuracy
            raw_in = raw
            raw, res.accuracy = ensure_accuracy(raw, res.analysis, run_dir / "refine",
                                                procedural=getattr(res, "procedural", True))
            if res.accuracy.get("refined"):
                res.raw_glb = finalize(raw_in, run_dir, name="model_raw")["glb"]
            a = res.accuracy
            res.logs.append(f"  accuracy {a['before']['accuracy']:.2f} -> {a['accuracy']:.2f} (IoU {a['iou']:.2f}, "
                            f"thin {a['thin_recall']:.2f}, holes {a['holes_open']}/{a['holes_expected']})"
                            + (f"  ⚠ {a['warning']}" if a.get("warning") else ""))
        except Exception as e:
            res.logs.append(f"  structural check skipped: {str(e)[:160]}")
        done("Structure", t)
    # 8-9. Repair, validate, export
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
        "accuracy": {k: v for k, v in (res.accuracy or {}).items() if k != "raw_glb"},
        "logs": res.logs}, indent=2, ensure_ascii=False), encoding="utf-8")
    if progress:
        progress(1.0, desc="Done")
    return res


# ------------------------------------------------------------- Studio: candidates as a step
def make_candidates(sketch_input, prompt: str = "", n: int = 4, seed: int = 42,
                    image_provider: str = "auto", out_dir: Path | None = None,
                    hole_overrides: dict | None = None, as_is: bool = False) -> dict:
    """Stages 1-5 only: preprocess, caption, generate n candidates, remove backgrounds and
    rank them. Nothing expensive (3D) runs until the user picks one (or Auto does).
    as_is (or an auto-detected clean illustration): skip generation and use the input itself,
    so no diffusion model can redraw away a handle or strings.
    -> {dir, prompt, caption, provider, cands: [{path, rgba_path, score...}], best, timings}"""
    t_all, timings = time.time(), {}
    out_dir = out_dir or config.OUTPUTS / datetime.now().strftime("cands_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    t = time.time()
    try:
        sk = preprocess(sketch_input, text=prompt, overrides=hole_overrides)
    except PreprocessError:
        tpl = make_template(prompt) if prompt.strip() else None
        if tpl is None:
            raise
        gray = cv2.cvtColor(tpl["img"], cv2.COLOR_RGB2GRAY)
        sk = Sketch(clean=gray, scribble=((gray < 128) * 255).astype(np.uint8), mask=tpl["body"],
                    rgb=tpl["img"], stroke_px=4)
    timings["preprocess"] = round(time.time() - t, 2)
    if sketch_input is not None:
        Image.fromarray(np.asarray(sketch_input)[..., :3].astype(np.uint8)).save(out_dir / "input.png")
    (out_dir / "input_prompt.txt").write_text(prompt, encoding="utf-8")
    from .segment import is_clean_illustration, merge_holes
    bypass = sketch_input is not None and (as_is or is_clean_illustration(np.asarray(sketch_input)))
    t = time.time()
    caption = "" if names_object(prompt) else auto_caption(sk)
    full = build_prompt(prompt, caption or "an object", "auto") + structure_phrase(sk.analysis)
    timings["caption"] = round(time.time() - t, 2)
    t = time.time()
    if bypass:
        imgs, provider = [np.asarray(sketch_input)[..., :3].astype(np.uint8)], "as-is (no generation)"
        logs = ["input used as-is: clean illustration, diffusion skipped"]
    else:
        imgs, provider, logs = generate_candidates(sk, full, n, image_provider, seed)
    timings["generate"] = round(time.time() - t, 2)
    t = time.time()
    rgbas = [merge_holes(remove_bg(c), sk.analysis) for c in imgs]
    timings["remove_bg"] = round(time.time() - t, 2)
    t = time.time()
    best, scores = rank(sk.mask, sk.scribble, rgbas, prompt.strip() or caption, sk.analysis)
    timings["rank"] = round(time.time() - t, 2)
    cands = []
    for i, (c, s) in enumerate(zip(rgbas, scores)):
        rp = out_dir / f"cand_{i}_rgba.png"
        Image.fromarray(c).save(rp)
        vp = _save(rgba_on_white(c), out_dir / f"cand_{i}.png")
        cands.append({**s, "path": vp, "rgba_path": str(rp), "label": card_label(i, s, best)})
    timings["total"] = round(time.time() - t_all, 2)
    (out_dir / "candidates.json").write_text(json.dumps(
        {"prompt": full, "caption": caption, "provider": provider, "best": best, "timings": timings,
         "scores": scores, "logs": logs}, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"dir": out_dir, "prompt": full, "caption": caption, "provider": provider, "cands": cands,
            "best": best, "timings": timings, "sketch": sk, "bypass": bypass}


def structure_phrase(analysis) -> str:
    """Structural words for the image model, from what the sketch must keep."""
    if analysis is None:
        return ""
    bits = []
    if any(h["kind"] == "through-hole" for h in analysis.holes):
        bits.append("with an open looped handle and see-through openings")
    if any(f["kind"] == "strings" for f in analysis.families):
        bits.append("hollow open frame with visible thin strings")
    if any(f["kind"] == "spokes" for f in analysis.families):
        bits.append("open wheel with visible thin spokes")
    return (", " + ", ".join(bits)) if bits else ""


def candidate_to_3d(rgba_path: str, out_dir: Path, mesh_provider: str = "auto", seed: int = 42,
                    name: str = "model", analysis=None, procedural: bool = True) -> dict:
    """Stages 6-9 for the ONE chosen candidate: image-to-3D, structural refinement against the
    sketch (when `analysis` is given), repair, validate, export."""
    t = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    rgba = np.asarray(Image.open(rgba_path).convert("RGBA"))
    raw, provider, logs = image_to_mesh(rgba, out_dir, mesh_provider, seed)
    t3 = round(time.time() - t, 2)
    acc, raw_glb = None, None
    if analysis is not None:
        from .refine import ensure_accuracy
        refined, acc = ensure_accuracy(raw, analysis, out_dir / "refine", procedural=procedural)
        if acc.get("refined"):
            raw_glb = finalize(raw, out_dir, name=name + "_raw")["glb"]
            raw = refined
    out = finalize(raw, out_dir, name=name)
    return {**out, "provider": provider, "logs": logs, "accuracy": acc, "raw_glb": raw_glb,
            "timings": {"image_to_3d": t3, "export": round(time.time() - t - t3, 2),
                        "total": round(time.time() - t, 2)}}
