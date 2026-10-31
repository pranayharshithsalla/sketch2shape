"""AI part assembly: a language model reads ANY prompt, plans the object as parts
(primitives with sizes, positions, rotations, colours, materials) and we build that plan
as clean 3D geometry with one PBR material per part.

It is conversational: follow-up requests ("make the legs thinner", "add armrests",
"paint the seat red") are applied to the current plan, keeping everything else.
"""
from __future__ import annotations

import json
import logging
import os
import re
import uuid
from pathlib import Path

import numpy as np
import requests
import trimesh

from . import config
from .router import ProviderError, with_retries

log = logging.getLogger(__name__)

# Fast model first, better model as fallback; both are on the free tier
PLANNER_MODELS = [m.strip() for m in os.getenv(
    "PLANNER_MODELS", "gemini-flash-lite-latest,gemini-3.5-flash,gemini-flash-latest").split(",") if m.strip()]
STRONG_MODEL = os.getenv("PLANNER_STRONG_MODEL", "gemini-3.5-flash")

SYSTEM = """You are a 3D modelling planner. Turn the user's description into a JSON build plan
made of simple parts. Think about the real object's structure, proportions and every detail
the user asked for (counts, presence/absence of parts, colours, materials, shapes).

Coordinate system: units are centimetres, Y is UP, the object stands on the floor (y=0),
centred on x=0,z=0. Front faces +Z. Use realistic real-world sizes.

Output ONLY JSON with this schema:
{
  "object": "short name",
  "reply": "one friendly sentence telling the user what you built or changed",
  "parts": [
    {
      "name": "seat",                      // descriptive, e.g. "front left leg"
      "role": "body|handle|cap|lid|leg|base|rim|spout|wheel|window|detail|...",
      "shape": "box|cylinder|sphere|cone|capsule|torus|lathe|tube",
      "size": [...],                       // box: [width,height,depth]; cylinder/cone/capsule: [radius,height];
                                           // sphere: [radius] or [rx,ry,rz]; torus: [major_radius,minor_radius]
      "profile": [[radius,y],...],         // lathe only: outline from bottom to top (radius>=0), revolved around Y
      "points": [[x,y,z],...],             // tube only: path points; size=[radius]
      "position": [x,y,z],                 // centre of the part (lathe: base centre)
      "rotation": [rx,ry,rz],              // degrees; cylinders/cones/capsules point along +Y before rotation
      "color": "#rrggbb",
      "material": "wood|metal|gold|silver|plastic|ceramic|glass|fabric|leather|rubber|stone|paint"
    }
  ]
}
Rules:
- Organic things (animals, people, plants, food) are approximated with spheres, capsules and
  cones in a friendly toy-like style; never refuse.
- Use lathe for round vessels (vases, bottles, cups, lamps): give 6-14 profile points.
- Parts must touch/overlap so the object is connected; nothing floats unless the object floats.
- Respect every explicit instruction (e.g. "4 legs", "no armrests", "green handle").
- 3 to 40 parts. Pick colours that match the description (or sensible real colours).
- When given a CURRENT PLAN and a change request, return the COMPLETE updated plan: apply
  exactly the requested change and keep every other part, size, position and colour as is.
  If the request is a new, unrelated object, start a fresh plan."""

MATERIAL_PBR = {  # (metallic, roughness)
    "metal": (0.95, 0.3), "gold": (1.0, 0.25), "silver": (1.0, 0.2), "chrome": (1.0, 0.08),
    "copper": (1.0, 0.3), "steel": (0.95, 0.3), "wood": (0.0, 0.7), "plastic": (0.0, 0.45),
    "ceramic": (0.0, 0.15), "glass": (0.0, 0.05), "fabric": (0.0, 0.95), "leather": (0.0, 0.6),
    "rubber": (0.0, 0.95), "stone": (0.0, 0.9), "paint": (0.0, 0.4),
}


def available() -> bool:
    return bool(config.GEMINI_API_KEY) and not config.OFFLINE["on"]


SKETCH_BRIEF = """The attached image is the user's hand-drawn SKETCH of the object (front view).
Read it carefully: identify every part (body, cup, lid, neck, cap, stem, base, handles, spouts,
legs, wheels, ...), count them, and measure their relative heights, widths and positions from
the drawing (the drawing's full height = the object's height). Keep the sketch's design and
proportions, but IDEALISE it into a clean, well-made product: perfectly symmetric (mirror paired
parts such as two handles exactly), smooth curves, straight lines where intended, consistent
thickness, parts properly attached. Ignore stray marks, shading and paper texture.
Round vessels/cups/bases/stems should be lathe parts with smooth profiles. Handles are tubes
(points along the handle's centre line, attached to the body at both ends)."""


EDIT_IMAGE_BRIEF = """The attached image is a REFERENCE the user shared with this change request
(a sketch or a photo). It is NOT a new object to build. Borrow ONLY the aspect the request
names from it (e.g. "the handle like this" -> copy that handle's shape onto the CURRENT
object's handles; "this colour" -> use its colour; "this profile" -> reshape that part).
KEEP the current object's type, name and every other part exactly as they are.
Only if the request explicitly says to replace/rebuild/start over from the image, build the
object in the image instead."""


def _ask(model: str, prompt: str, image_png: bytes | None = None, editing: bool = False) -> dict:
    parts = [{"text": prompt}]
    if image_png is not None:
        import base64
        img = {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(image_png).decode()}}
        if editing:  # prompt already holds CURRENT PLAN + conversation + CHANGE REQUEST
            parts = [{"text": EDIT_IMAGE_BRIEF}, img, {"text": prompt}]
        else:
            parts = [{"text": SKETCH_BRIEF}, img,
                     {"text": f"User's description: {prompt or '(none: infer it from the sketch)'}"}]
    body = {"systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"parts": parts}],
            "generationConfig": {"temperature": 0.3, "responseMimeType": "application/json"}}
    r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                      headers={"x-goog-api-key": config.GEMINI_API_KEY}, json=body,
                      timeout=45 if image_png is not None else 30)
    if r.status_code != 200:
        raise ProviderError(f"planner HTTP {r.status_code}: {r.text[:200]}")
    text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M)
    return json.loads(text)


def _request_text(prompt: str, current: dict | None, history: list[tuple[str, str]] | None) -> str:
    if not current:
        return prompt
    convo = "\n".join(f"User: {u}\nYou: {a}" for u, a in (history or [])[-6:])
    return (f"CURRENT PLAN:\n{json.dumps(current, separators=(',', ':'))}\n\n"
            f"CONVERSATION SO FAR:\n{convo or '(none)'}\n\n"
            f"CHANGE REQUEST: {prompt}\n\nReturn the complete updated plan as JSON.")


def plan(prompt: str, current: dict | None = None,
         history: list[tuple[str, str]] | None = None,
         sketch_png: bytes | None = None) -> tuple[dict, str]:
    """-> (plan dict, model used). With `current`, the request edits that plan. With
    `sketch_png` (a sketch or photo), a new plan is read from the image, or, when editing,
    the image is a reference for the change ("make the handle like this")."""
    text = _request_text(prompt, current, history)
    errs = []
    # a new design is worth the stronger model (richer structure); edits use the fast one first
    order = (PLANNER_MODELS if current and sketch_png is None
             else sorted(PLANNER_MODELS, key=lambda m: m != STRONG_MODEL))
    for m in order:
        try:
            spec = with_retries(_ask, m, text, sketch_png, bool(current),
                                name=f"planner:{m}", retries=0)
            if not isinstance(spec, dict) or not spec.get("parts"):
                raise ProviderError("plan has no parts")
            return _self_check(spec, m), m
        except Exception as e:
            errs.append(f"{m}: {str(e)[:120]}")
    raise ProviderError("AI planner unavailable: " + " | ".join(errs))


def floating_parts(spec: dict) -> list[str]:
    """Names of parts whose box doesn't touch any other part (gap > 2% of object size).
    Models sometimes misplace a lid or cup in the air; one repair round fixes that."""
    boxes = []
    for p in spec.get("parts") or []:
        try:
            m = _primitive(p)
            if m is None:
                continue
            rx, ry, rz = np.radians(_vec(p.get("rotation"), 3, [0, 0, 0]))
            m.apply_transform(trimesh.transformations.euler_matrix(rx, ry, rz, "sxyz"))
            if str(p.get("shape")).lower() != "tube":
                m.apply_translation(_vec(p.get("position"), 3, [0, 0, 0]))
            boxes.append((str(p.get("name", p.get("role", "part"))), m.bounds))
        except Exception:
            continue
    if len(boxes) < 2:
        return []
    lo = np.min([b[0] for _, b in boxes], axis=0)
    hi = np.max([b[1] for _, b in boxes], axis=0)
    tol = 0.02 * float((hi - lo).max())
    out = []
    for i, (name, a) in enumerate(boxes):
        touches = any(np.all(a[0] - tol <= b[1]) and np.all(b[0] - tol <= a[1])
                      for j, (_, b) in enumerate(boxes) if j != i)
        if not touches:
            out.append(name)
    return out


def _self_check(spec: dict, model: str) -> dict:
    """If parts float free, ask the same model once to reattach them; keep the better plan."""
    loose = floating_parts(spec)
    if not loose:
        return spec
    try:
        fix = _ask(model, _request_text(
            f"These parts are floating and don't touch the rest of the object: {', '.join(loose)}. "
            "Move/resize ONLY what is needed so every part is attached (touching or slightly "
            "overlapping its neighbour). Keep the design, colours and proportions.", spec, []))
        if isinstance(fix, dict) and fix.get("parts") and len(floating_parts(fix)) < len(loose):
            fix["reply"] = spec.get("reply") or fix.get("reply")  # the user sees the real answer
            log.info("self-check reattached %d floating part(s)", len(loose) - len(floating_parts(fix)))
            return fix
    except Exception as e:
        log.warning("self-check skipped: %s", e)
    return spec


# ------------------------------------------------------------------ geometry
def _num(v, default):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _vec(v, n, default):
    v = list(v) if isinstance(v, (list, tuple)) else []
    return [_num(v[i], default[i]) if i < len(v) else default[i] for i in range(n)]


def _y_up(mesh):  # trimesh primitives are built along Z; our convention is Y-up
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    return mesh


def _primitive(p: dict) -> trimesh.Trimesh | None:
    shape = str(p.get("shape", "box")).lower()
    s = p.get("size") or [10]
    s = [max(abs(_num(x, 1.0)), 0.05) for x in (s if isinstance(s, (list, tuple)) else [s])]
    if shape == "box":
        w, h, d = (s + [s[-1]] * 3)[:3]
        return trimesh.creation.box(extents=[w, h, d])
    if shape == "cylinder":
        r, h = (s + [s[0] * 2])[:2]
        return _y_up(trimesh.creation.cylinder(radius=r, height=h, sections=48))
    if shape == "cone":
        r, h = (s + [s[0] * 2])[:2]
        m = _y_up(trimesh.creation.cone(radius=r, height=h, sections=48))
        m.apply_translation([0, -h / 2, 0])  # centre it like the other primitives
        return m
    if shape == "capsule":
        r, h = (s + [s[0] * 2])[:2]
        return _y_up(trimesh.creation.capsule(radius=r, height=max(h - 2 * r, 0.01), count=[24, 24]))
    if shape == "sphere":
        m = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        rx, ry, rz = (s + [s[0]] * 3)[:3] if len(s) >= 3 else (s[0],) * 3
        m.apply_scale([rx, ry, rz])
        return m
    if shape == "torus":
        R, r = (s + [s[0] * 0.25])[:2]
        return _y_up(trimesh.creation.torus(major_radius=R, minor_radius=min(r, R * 0.9),
                                            major_sections=48, minor_sections=16))
    if shape == "lathe":
        prof = [(max(_num(q[0], 0), 0), _num(q[1], 0)) for q in (p.get("profile") or [])
                if isinstance(q, (list, tuple)) and len(q) >= 2][:60]
        if len(prof) < 2:
            return None
        ys = [y for _, y in prof]
        if all(b >= a for a, b in zip(ys, ys[1:])) or all(b <= a for a, b in zip(ys, ys[1:])):
            prof = sorted(prof, key=lambda t: t[1])  # simple outline: bottom to top
        # else: keep the given order; it is a closed wall section (outer up, inner down), as
        # models write hollow cups and mugs, and re-sorting it by height would scramble it
        pts = prof
        if pts[0][0] > 0:
            pts = [(0.0, pts[0][1])] + pts
        if pts[-1][0] > 0:
            pts = pts + [(0.0, pts[-1][1])]
        m = trimesh.creation.revolve(np.array(pts), sections=64)
        m.fix_normals()
        return _y_up(m)
    if shape == "tube":
        pts = [_vec(q, 3, [0, 0, 0]) for q in (p.get("points") or [])]
        r = s[0]
        if len(pts) < 2:
            return None
        try:  # smooth swept tube along a Catmull-Rom spline through the model's points
            from shapely.geometry import Point
            path = _catmull_rom(np.array(pts, float), samples=12)
            ring = Point(0, 0).buffer(r, quad_segs=8)
            tube = trimesh.creation.sweep_polygon(ring, path)
            caps = [trimesh.creation.icosphere(subdivisions=2, radius=r).apply_translation(path[k])
                    for k in (0, -1)]
            return trimesh.util.concatenate([tube] + caps)
        except Exception as e:
            log.debug("sweep failed (%s); using cylinder segments", e)
        segs = []
        for a, b in zip(pts[:-1], pts[1:]):
            a, b = np.array(a), np.array(b)
            if np.linalg.norm(b - a) < 1e-6:
                continue
            segs.append(trimesh.creation.cylinder(radius=r, segment=[a, b], sections=24))
            segs.append(trimesh.creation.icosphere(subdivisions=2, radius=r).apply_translation(b))
        segs.append(trimesh.creation.icosphere(subdivisions=2, radius=r).apply_translation(pts[0]))
        return trimesh.util.concatenate(segs) if segs else None
    return trimesh.creation.box(extents=(s + [s[-1]] * 3)[:3])  # unknown shape: a box


def _catmull_rom(p: np.ndarray, samples: int = 12) -> np.ndarray:
    """Smooth curve through all points (centripetal-free uniform Catmull-Rom)."""
    if len(p) < 3:
        t = np.linspace(0, 1, samples * 2)[:, None]
        return p[0] * (1 - t) + p[-1] * t
    q = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    out = []
    for i in range(1, len(q) - 2):
        p0, p1, p2, p3 = q[i - 1], q[i], q[i + 1], q[i + 2]
        for t in np.linspace(0, 1, samples, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(p[-1])
    return np.array(out)


def _hex(c, default=(150, 150, 150)):
    m = re.fullmatch(r"#?([0-9a-fA-F]{6})", str(c or "").strip())
    return tuple(int(m.group(1)[i:i + 2], 16) for i in (0, 2, 4)) if m else default


def build_plan(spec: dict, out_dir: Path, prompt: str = "", preset: str = "auto") -> tuple[str, list[str]]:
    """Build the planner's parts as a multi-material GLB. Returns (glb path, log lines)."""
    from .prompt import parse as parse_look
    from .sketch3d import apply_material
    from . import textures
    look = parse_look(prompt, preset)  # explicit prompt colours win over the model's guesses
    built, logs = [], []
    for i, p in enumerate(spec.get("parts") or []):
        if not isinstance(p, dict):
            continue
        try:
            m = _primitive(p)
        except Exception as e:
            logs.append(f"part {i} skipped ({e})")
            continue
        if m is None or not len(m.faces):
            continue
        rx, ry, rz = np.radians(_vec(p.get("rotation"), 3, [0, 0, 0]))
        R = trimesh.transformations.euler_matrix(rx, ry, rz, "sxyz")
        m.apply_transform(R)
        pos = _vec(p.get("position"), 3, [0, 0, 0])
        shape = str(p.get("shape", "box")).lower()
        if shape != "tube":  # tube points are already absolute
            m.apply_translation(pos)
        built.append((i, p, m, shape, R[:3, :3] @ np.array([0, 1.0, 0])))
    if not built:
        raise ProviderError("AI plan contained no buildable parts")
    lo = np.min([m.bounds[0] for *_, m, _, _ in built], axis=0)
    hi = np.max([m.bounds[1] for *_, m, _, _ in built], axis=0)
    tile = float((hi - lo).max()) / 3.0  # same texture density across all parts
    scene, n_ok, textured = trimesh.Scene(), 0, []
    for i, p, m, shape, axis in built:
        role = str(p.get("role", "body")).lower()
        mat = str(p.get("material", "paint")).lower()
        metal, rough = MATERIAL_PBR.get(mat, (0.0, 0.5))
        rgb = _hex(p.get("color"))
        part_key = {"lid": "cap", "holder": "handle", "grip": "handle"}.get(role, role)
        if part_key in look["parts"]:
            rgb, metal, rough = look["parts"][part_key]
            mat = look["mat_names"].get(part_key) or mat
        elif look.get("body_explicit") and (part_key == "body" or "body" in str(p.get("name", "")).lower()):
            rgb, metal, rough = look["body"]  # the user's colour/material beats the model's guess
            mat = look["mat_names"].get("body") or mat
        mapping = "planar" if shape == "box" else ("cyl" if shape == "tube" else "axis")
        tm = textures.apply(m, mat, rgb, metal, rough, tile, mapping, axis=axis)
        if tm is not None:
            textured.append(textures.canonical(mat))
        else:
            apply_material(m, rgb, metal, rough)
        scene.add_geometry(tm if tm is not None else m,
                           geom_name=f"{i:02d}_{re.sub(r'[^a-z0-9]+', '_', str(p.get('name', role)).lower())}")
        n_ok += 1
        logs.append(f"{p.get('name', role)}: {p.get('shape')} {p.get('size')} {p.get('color')} {mat}")
    if textured:
        logs.append("textures: " + ", ".join(sorted(set(textured))))
    if n_ok == 0:
        raise ProviderError("AI plan contained no buildable parts")
    path = out_dir / f"ai_parts_{uuid.uuid4().hex[:6]}.glb"
    scene.export(path)
    return str(path), logs
