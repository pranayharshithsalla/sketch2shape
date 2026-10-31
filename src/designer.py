"""Studio: one conversation where every turn can be text, a sketch, an uploaded image, or any
mix. The first turn builds a model (from words, or by reading the sketch/photo); every later
turn edits the PREVIOUS result: "make the legs thinner", or a sketch of "a handle like this",
or a reference photo. Each turn sends the current plan + recent conversation (+ the attached
image) to the planner, rebuilds the model and saves a numbered version that can be undone.
"""
from __future__ import annotations

import copy
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


def turn(session: Session, message: str, image=None) -> tuple[Session, str]:
    """Apply one user turn: text and/or a sketch/photo. Returns (session, assistant reply)."""
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
    try:
        spec, model = ai_assembly.plan(message or "Build this.", current, session.history, sketch_png=png)
    except Exception as e:
        if current is None:  # first request and the AI is unreachable: local engines
            return session, _local_fallback(session, message, png, str(e)[:120])
        raise

    n = len(session.versions) + 1
    vdir = session.run_dir / f"v{n}"
    vdir.mkdir(exist_ok=True)
    if png is not None:
        (vdir / "attached.png").write_bytes(png)  # keep the reference with its version
    message = message or "(sketch/photo)"
    raw, part_logs = ai_assembly.build_plan(spec, vdir, message)
    out = finalize(raw, vdir, name=f"model_v{n}")
    session.versions.append({"plan": spec, "glb": out["glb"], "files": [out["glb"], out["obj_zip"], out["stl"]],
                             "report": out["report"], "prompt": message, "model": model,
                             "seconds": round(time.time() - t0, 1)})
    reply = (spec.get("reply") or f"Here is your {spec.get('object', 'design')}.").strip()
    session.history.append((message, reply))
    rep = out["report"]
    detail = (f"*v{n} · {spec.get('object', 'design')} · {_summary(spec)} · {rep['faces']:,} faces · "
              f"{'watertight' if rep['watertight'] else 'open'} · {session.current['seconds']} s · {model}*")
    return session, f"{reply}\n\n{detail}"


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
    r = run(sketch, message,
            run_name=f"{session.run_dir.name}_local_v{len(session.versions) + 1}")
    session.versions.append({"plan": r.plan, "glb": r.glb, "files": [r.glb, r.obj_zip, r.stl],
                             "report": r.report, "prompt": message or "(sketch/photo)",
                             "model": r.mesh_provider, "seconds": r.timings.get("total")})
    reply = (f"The AI planner is unavailable ({why}), so I built this with the local "
             f"`{r.mesh_provider}` engine. Follow-up edits need the AI planner.")
    session.history.append((message or "(sketch/photo)", reply))
    return reply
