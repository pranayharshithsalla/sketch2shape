"""Conversational AI designer: describe an object, get a 3D model, then keep refining it in
plain language ("make the legs thinner", "add armrests", "paint the seat red").

Each turn sends the current plan + recent conversation to the planner, rebuilds the model,
and saves a numbered version so any step can be undone.
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


def turn(session: Session, message: str) -> tuple[Session, str]:
    """Apply one user message. Returns (session, assistant reply). Raises on failure."""
    message = message.strip()
    if not message:
        return session, "Tell me what to build, or what to change."
    if session.run_dir is None:
        session.run_dir = config.OUTPUTS / datetime.now().strftime("design_%Y%m%d_%H%M%S")
        session.run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    current = copy.deepcopy(session.current["plan"]) if session.current else None

    if not ai_assembly.available():
        return session, _template_fallback(session, message, "no GEMINI_API_KEY is set")
    try:
        spec, model = ai_assembly.plan(message, current, session.history)
    except Exception as e:
        if current is None:  # first request and the AI is unreachable: use a template
            return session, _template_fallback(session, message, str(e)[:120])
        raise

    n = len(session.versions) + 1
    vdir = session.run_dir / f"v{n}"
    vdir.mkdir(exist_ok=True)
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


def _template_fallback(session: Session, message: str, why: str) -> str:
    """Offline first turn: build from the parametric templates if the object is known."""
    from .pipeline import run
    from .templates import available_names, make_template
    if make_template(message) is None:
        return (f"The AI planner is unavailable right now ({why}). Offline I can build: "
                + ", ".join(available_names()) + ". Or draw a sketch in the Sketch tab.")
    r = run(None, message, run_name=f"{session.run_dir.name}_template_v{len(session.versions) + 1}")
    session.versions.append({"plan": None, "glb": r.glb, "files": [r.glb, r.obj_zip, r.stl],
                             "report": r.report, "prompt": message, "model": "template",
                             "seconds": r.timings.get("total")})
    reply = (f"The AI planner is unavailable ({why}), so I built this from my offline template. "
             "Follow-up edits need the AI planner.")
    session.history.append((message, reply))
    return reply
