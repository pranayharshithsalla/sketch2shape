"""Studio: one conversation where every turn can be text, a sketch, an uploaded image, or any
mix. The first turn builds a model (from words, or by reading the sketch/photo); every later
turn edits the PREVIOUS result: "make the legs thinner", or a sketch of "a handle like this",
or a reference photo. Each turn sends the current plan + recent conversation (+ the attached
image) to the planner, rebuilds the model and saves a numbered version that can be undone.
"""
from __future__ import annotations

import copy
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import ai_assembly, config
from .export import finalize


@dataclass
class Session:
    run_dir: Path | None = None
    versions: list[dict] = field(default_factory=list)   # [{plan, glb, files, report, prompt}]
    history: list[tuple[str, str]] = field(default_factory=list)  # (user, assistant)
    ref_png: bytes | None = None  # latest sketch/photo the user attached (for Accuracy)
    ref_text: str = ""            # prompt that came with it (resolves single-line loops)
    hole_overrides: dict = field(default_factory=dict)  # click-to-toggle corrections {region: is_hole}
    procedural: bool = True       # rebuild strings / wires as clean tubes

    @property
    def current(self) -> dict | None:
        return self.versions[-1] if self.versions else None


def _summary(spec: dict) -> str:
    parts = spec.get("parts") or []
    names = [str(p.get("name", p.get("role", "part"))) for p in parts]
    shown = ", ".join(names[:12]) + (f" … (+{len(names) - 12})" if len(names) > 12 else "")
    return f"{len(parts)} parts: {shown}"


def image_png(image) -> bytes | None:
    """Sketch canvas value / uploaded image / path -> PNG bytes on white, max 768 px (or None)."""
    if image is None:
        return None
    import cv2
    import numpy as np
    from .preprocess import PreprocessError, _to_rgba, composite_on_white
    try:
        rgb = composite_on_white(_to_rgba(image))
    except PreprocessError:
        return None
    if (rgb < 200).mean() < 0.0005:  # blank canvas
        return None
    h, w = rgb.shape[:2]
    s = 768 / max(h, w)
    if s < 1:
        rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return buf.tobytes() if ok else None


def sketch_accuracy(glb: str, ref_png: bytes | None) -> float | None:
    """Front-view silhouette + edge agreement of the model with the user's sketch (0-1)."""
    if ref_png is None:
        return None
    try:
        import cv2
        import numpy as np
        from .pipeline import _front_fidelity
        from .preprocess import preprocess
        img = cv2.cvtColor(cv2.imdecode(np.frombuffer(ref_png, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return float(_front_fidelity(glb, preprocess(img))["score"])
    except Exception:
        return None


CONTROL = {  # what an attached sketch/photo (or a text edit) is allowed to change
    "both": "",
    "shape": ("[Use the attached image ONLY for the shape, parts and proportions. Keep the current "
              "colours, materials and finish exactly as they are.] "),
    "style": ("[Shape is LOCKED: keep every part, position and dimension exactly as it is. Change ONLY "
              "colours, materials, textures and finish (from the attached image if there is one).] "),
}
FIDELITY_RETRY_BELOW = 0.62   # front-silhouette accuracy that triggers an automatic re-plan
MAX_FIDELITY_RETRIES = 1  # each retry is another Gemini plan + build


def turn(session: Session, message: str, image=None, control: str = "both",
         photo: bool = False) -> tuple[Session, str]:
    """Apply one user turn: text and/or a sketch/photo. Returns (session, assistant reply).
    control: 'both' | 'shape' | 'style' - what this input may change (see CONTROL)."""
    message = (message or "").strip()
    png = image_png(image)
    if not message and png is None:
        return session, "Tell me what to build or change, or attach a sketch or photo."
    if session.run_dir is None:
        session.run_dir = config.OUTPUTS / datetime.now().strftime("design_%Y%m%d_%H%M%S")
        session.run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    current = copy.deepcopy(session.current["plan"]) if session.current else None
    if session.current and current is None:
        # the previous version came from a local engine (no part plan): edits need the AI
        if not ai_assembly.available():
            return session, "Editing a model needs the AI planner (set GEMINI_API_KEY)."

    if not ai_assembly.available():
        return session, _local_fallback(session, message, png, "no GEMINI_API_KEY is set")
    guide = CONTROL.get(control, "") if (current is not None or png is not None) else ""
    ask = guide + (message or "Build this.")
    try:
        spec, model = ai_assembly.plan(ask, current, session.history, sketch_png=png)
    except Exception as e:
        if current is None:  # first request and the AI is unreachable: local engines
            return session, _local_fallback(session, message, png, str(e)[:120])
        raise

    n = len(session.versions) + 1
    vdir = session.run_dir / f"v{n}"
    vdir.mkdir(exist_ok=True)
    if png is not None:
        (vdir / "attached.png").write_bytes(png)  # keep the reference with its version
        if control != "style":  # a style reference is not a shape target
            session.ref_png = png
    kind = "prompt" if png is None else ("photo" if photo else "sketch")
    message = message or f"({kind})"
    out = _build(spec, vdir, message, n)
    acc = sketch_accuracy(out["glb"], session.ref_png)

    # Sketch-Fidelity auto-retry: a new design read from a sketch that misses its outline is
    # re-planned with the score as feedback (max 2 retries); the best attempt is kept.
    notes = []
    if png is not None and current is None and control != "style" and acc is not None:
        for k in range(MAX_FIDELITY_RETRIES):
            if acc >= FIDELITY_RETRY_BELOW:
                break
            fb = (f"{ask}\n[Your previous model matched the sketch outline only {acc:.0%}. Follow the drawn "
                  f"silhouette, proportions and part positions much more closely.]")
            try:
                spec2, model2 = ai_assembly.plan(fb, None, session.history, sketch_png=png)
                out2 = _build(spec2, vdir, message, n, suffix=f"_retry{k + 1}")
                acc2 = sketch_accuracy(out2["glb"], session.ref_png) or 0.0
            except Exception as e:
                notes.append(f"retry {k + 1} failed ({str(e)[:60]})")
                break
            notes.append(f"fidelity retry {k + 1}: {acc:.0%} → {acc2:.0%}")
            if acc2 > acc:
                spec, model, out, acc = spec2, model2, out2, acc2

    # structural accuracy vs the sketch: holes, thin parts, strings (carve/add + procedural tubes)
    acc_report, raw_glb = None, None
    if session.ref_png is not None:
        if png is not None and control != "style":
            session.ref_text = message
        try:
            out, acc_report, raw_glb = _structural(session, out, vdir, n)
            if acc_report is not None:
                acc = acc_report["accuracy"]
                if acc_report.get("refined"):
                    notes.append(f"structure {acc_report['before']['accuracy']:.0%} → {acc:.0%}")
                if acc_report.get("warning"):
                    notes.append("⚠ " + acc_report["warning"])
        except Exception as e:
            notes.append(f"structure check skipped ({str(e)[:60]})")

    decals = list(session.current.get("decals") or []) if session.current else []
    plain = out["glb"]
    if decals:  # lettering/logos survive AI edits: re-bake them on the rebuilt model
        try:
            out = _rebake(out["glb"], decals, vdir, n)
        except Exception as e:
            notes.append(f"decals could not be re-applied ({str(e)[:60]})")
            decals = []
    session.versions.append({"plan": spec, "glb": out["glb"], "files": [out["glb"], out["obj_zip"], out["stl"]],
                             "report": out["report"], "prompt": message, "model": model, "kind": kind,
                             "control": control, "decals": decals, "plain_glb": plain,
                             "seconds": round(time.time() - t0, 1), "accuracy": acc,
                             "accuracy_report": acc_report, "raw_glb": raw_glb})
    reply = (spec.get("reply") or f"Here is your {spec.get('object', 'design')}.").strip()
    session.history.append((message, reply))
    rep = out["report"]
    extra = f" · {'; '.join(notes)}" if notes else ""
    detail = (f"*v{n} · {spec.get('object', 'design')} · {_summary(spec)} · {rep['faces']:,} faces · "
              f"{'watertight' if rep['watertight'] else 'open'} · {session.current['seconds']} s · {model}{extra}*")
    return session, f"{reply}\n\n{detail}"


def analysis_of(session: Session):
    """InputAnalysis of the session's reference sketch (with the user's hole corrections)."""
    if session.ref_png is None:
        return None
    import cv2
    import numpy as np
    from .preprocess import preprocess
    img = cv2.cvtColor(cv2.imdecode(np.frombuffer(session.ref_png, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return preprocess(img, text=session.ref_text, overrides=session.hole_overrides)


def _structural(session: Session, out: dict, vdir: Path, n: int):
    """-> (exported model dict, accuracy report, raw glb or None)."""
    from .refine import ensure_accuracy
    sk = analysis_of(session)
    if sk is None or sk.analysis is None:
        return out, None, None
    glb, rep = ensure_accuracy(out["glb"], sk.analysis, vdir / "refine", procedural=session.procedural)
    if not rep.get("refined"):
        return out, rep, None
    return finalize(glb, vdir, name=f"model_v{n}"), rep, out["glb"]


def _build(spec: dict, vdir: Path, message: str, n: int, suffix: str = "") -> dict:
    raw, _ = ai_assembly.build_plan(spec, vdir, message)
    return finalize(raw, vdir, name=f"model_v{n}{suffix}")


def _decal_image(d: dict):
    from PIL import Image
    from . import decal as dc
    if d.get("text"):
        return dc.text_image(d["text"], d["color"], d["font"])
    return Image.open(d["logo"]).convert("RGBA")


def _rebake(glb: str, decals: list[dict], vdir: Path, n: int) -> dict:
    """Bake every decal, in order, onto `glb` (a model without decals) and export."""
    from . import decal as dc
    cur = glb
    for k, d in enumerate(decals):
        cur = dc.bake(cur, vdir / f"decal_{k}.glb", _decal_image(d), d["x"], d["y"], d["size"],
                      d["rotation"], d["side"])["glb"]
    return finalize(cur, vdir, name=f"model_v{n}")


def add_decal(session: Session, text: str = "", logo=None, color: str = "#FFFFFF", font: str = "Bold sans",
              x: float = 0.5, y: float = 0.5, size: float = 0.4, rotation: float = 0.0,
              side: str = "front", replace_last: bool = False) -> tuple[Session, str]:
    """Bake text or a logo into the current model's texture as a new version.
    replace_last: re-place the decal just added (moving it) instead of stacking another one."""
    cur = session.current
    if not cur:
        return session, "Create a model first, then add text or a logo."
    text = (text or "").strip()
    if not text and logo is None:
        return session, "Type the text (or upload a logo) to put on the model."
    t0 = time.time()
    base = cur
    if replace_last and cur.get("kind") == "decal" and len(session.versions) > 1:
        session.versions.pop()          # move the last decal: start again from the version under it
        base = session.current
    decals = list(base.get("decals") or [])
    n = len(session.versions) + 1
    vdir = session.run_dir / f"v{n}"
    vdir.mkdir(parents=True, exist_ok=True)
    d = {"text": text, "color": color, "font": font, "x": float(x), "y": float(y), "size": float(size),
         "rotation": float(rotation), "side": side}
    if not text:
        import numpy as np
        from PIL import Image
        lp = vdir / "logo.png"
        (logo if isinstance(logo, Image.Image) else Image.fromarray(np.asarray(logo))).convert("RGBA").save(lp)
        d["logo"] = str(lp)
    # always bake onto the model as it was before any decal, so moved decals leave no trace
    plain = base.get("plain_glb") or base["glb"]
    out = _rebake(plain, decals + [d], vdir, n)
    session.versions.append({**base, "glb": out["glb"], "files": [out["glb"], out["obj_zip"], out["stl"]],
                             "report": out["report"], "kind": "decal", "decals": decals + [d],
                             "plain_glb": plain, "prompt": f"decal “{text or 'logo'}”",
                             "seconds": round(time.time() - t0, 1)})
    what = f"“{text}”" if text else "your logo"
    session.history.append((f"(add {what} on the {side})", "Baked into the texture."))
    return session, (f"Baked {what} into the model's texture on the {side} ({round(time.time() - t0, 1)} s). "
                     "It is part of the .glb, so it shows in any viewer and survives later edits.")


def recolor_parts(session: Session, names: list[str], colour: str,
                  material: str | None = None) -> tuple[Session, str]:
    """Recolour one or many parts of the current model as a new version. Baked decals are
    re-applied on top, and the AI plan is updated so later AI edits keep the new colours."""
    from .recolor import label, recolor
    cur = session.current
    if not cur:
        return session, "Create a model first, then pick the parts to recolour."
    if not names:
        return session, "Tick at least one part to recolour."
    t0 = time.time()
    n = len(session.versions) + 1
    vdir = session.run_dir / f"v{n}"
    vdir.mkdir(parents=True, exist_ok=True)
    plain = cur.get("plain_glb") or cur["glb"]
    res = recolor(plain, vdir / "recoloured.glb", names, colour, material)
    out_plain = finalize(res["glb"], vdir, name=f"model_v{n}_plain")
    decals = list(cur.get("decals") or [])
    out = _rebake(out_plain["glb"], decals, vdir, n) if decals else finalize(out_plain["glb"], vdir, name=f"model_v{n}")
    plan = copy.deepcopy(cur.get("plan"))
    if plan and plan.get("parts"):
        for nm in res["changed"]:
            m = re.match(r"^(\d+)_", nm)
            if m and int(m.group(1)) < len(plan["parts"]):
                p = plan["parts"][int(m.group(1))]
                p["color"] = colour
                if material:
                    p["material"] = material
    what = ", ".join(label(x) for x in res["changed"][:6]) + (" …" if len(res["changed"]) > 6 else "")
    session.versions.append({**cur, "plan": plan, "glb": out["glb"], "plain_glb": out_plain["glb"],
                             "files": [out["glb"], out["obj_zip"], out["stl"]], "report": out["report"],
                             "kind": "colour", "prompt": f"colour {what} {colour}{' ' + material if material else ''}",
                             "seconds": round(time.time() - t0, 1), "raw_glb": None})
    session.history.append((f"(recolour {what} → {colour}{' ' + material if material else ''})", "Done."))
    return session, (f"🎨 Recoloured {len(res['changed'])} part(s): {what} → {colour}"
                     f"{' (' + material + ')' if material else ''}. Saved as v{n}; undo or revert any time.")


def add_version(session: Session, glb: str, files: list, report: dict, prompt: str, model: str, kind: str,
                seconds: float | None, plan: dict | None = None, accuracy: float | None = None) -> None:
    """Record a model made outside the planner (candidate → 3D, local engines) as a version."""
    if session.run_dir is None:
        session.run_dir = config.OUTPUTS / datetime.now().strftime("design_%Y%m%d_%H%M%S")
        session.run_dir.mkdir(parents=True, exist_ok=True)
    session.versions.append({"plan": plan, "glb": glb, "files": files, "report": report, "prompt": prompt,
                             "model": model, "kind": kind, "seconds": seconds, "accuracy": accuracy,
                             "decals": [], "plain_glb": glb})


def revert(session: Session, index: int) -> tuple[Session, str]:
    """One-tap revert: version `index` (0-based) becomes the newest version again. Nothing is
    deleted, so a revert can itself be reverted."""
    if not (0 <= index < len(session.versions)):
        return session, "No such version."
    if index == len(session.versions) - 1:
        return session, f"v{index + 1} is already the current version."
    v = copy.copy(session.versions[index])
    v["kind"], v["prompt"] = "revert", f"revert to v{index + 1}"
    session.versions.append(v)
    session.history.append((f"(revert to v{index + 1})", f"Restored version {index + 1}."))
    return session, f"↩️ Restored v{index + 1} as v{len(session.versions)}."


ICONS = {"prompt": "💬", "sketch": "✏️", "photo": "📷", "decal": "🔤", "revert": "↩️", "candidate": "🎲",
         "local": "🧩", "colour": "🎨"}


def timeline(session: Session) -> list[str]:
    """Labels for the version timeline (v1 prompt → v2 photo → v3 sketch tweak …)."""
    out = []
    for i, v in enumerate(session.versions):
        acc = f" · {v['accuracy']:.0%}" if v.get("accuracy") is not None else ""
        out.append(f"v{i + 1} {ICONS.get(v.get('kind'), '•')} {str(v.get('prompt') or '')[:36]}{acc}")
    return out


def undo(session: Session) -> tuple[Session, str]:
    if len(session.versions) <= 1:
        return session, "Nothing to undo."
    session.versions.pop()
    session.history.append(("(undo)", f"Went back to version {len(session.versions)}."))
    return session, f"↩️ Back to version {len(session.versions)}: {session.current['prompt']}"


def _local_fallback(session: Session, message: str, png: bytes | None, why: str) -> str:
    """First turn without the AI planner: build with the local engines (a sketch/photo goes
    through Sketch-CAD / TripoSR; words alone use the parametric templates)."""
    import cv2
    import numpy as np
    from .pipeline import run
    from .templates import available_names, make_template
    sketch = None
    if png is not None:
        sketch = cv2.cvtColor(cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    elif make_template(message) is None:
        return (f"The AI planner is unavailable right now ({why}). Offline I can build: "
                + ", ".join(available_names()) + ", or anything you sketch or upload.")
    if png is not None:
        session.ref_png, session.ref_text = png, message
    r = run(sketch, message, run_name=f"{session.run_dir.name}_local_v{len(session.versions) + 1}",
            hole_overrides=session.hole_overrides, procedural=session.procedural)
    session.versions.append({"plan": r.plan, "glb": r.glb, "files": [r.glb, r.obj_zip, r.stl],
                             "report": r.report, "prompt": message or "(sketch/photo)", "kind": "local",
                             "model": r.mesh_provider, "seconds": r.timings.get("total"), "decals": [],
                             "plain_glb": r.glb,
                             "accuracy": (r.accuracy or {}).get("accuracy", sketch_accuracy(r.glb, png)),
                             "accuracy_report": r.accuracy, "raw_glb": r.raw_glb or None})
    reply = (f"The AI planner is unavailable ({why}), so I built this with the local "
             f"`{r.mesh_provider}` engine. Follow-up edits need the AI planner.")
    session.history.append((message or "(sketch/photo)", reply))
    return reply
