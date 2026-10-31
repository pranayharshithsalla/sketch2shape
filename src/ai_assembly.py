"""AI part assembly: a language model reads ANY prompt, plans the object as parts
(primitives with sizes, positions, rotations, colours, materials) and we build that plan
as clean 3D geometry with one PBR material per part.

"a wooden chair with 4 legs, no armrests and a backrest" -> seat box + 4 leg cylinders +
backrest box, each placed and coloured by the model. The model also decides when the
object is too organic for parts (an animal, a character) and asks for the image route.
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
    "PLANNER_MODELS", "gemini-3.5-flash,gemini-flash-latest,gemini-flash-lite-latest").split(",") if m.strip()]

SYSTEM = """You are a 3D modelling planner. Turn the user's description into a JSON build plan
made of simple parts. Think about the real object's structure, proportions and every detail
the user asked for (counts, presence/absence of parts, colours, materials, shapes).

Coordinate system: units are centimetres, Y is UP, the object stands on the floor (y=0),
centred on x=0,z=0. Front faces +Z. Use realistic real-world sizes.

Output ONLY JSON with this schema:
{
  "object": "short name",
  "approach": "parts" | "image",
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
- Use "approach":"image" (with empty parts) ONLY for organic shapes that cannot be built from
  primitives (animals, people, characters, plants, sculptures, food).
- Use lathe for round vessels (vases, bottles, cups, lamps): give 6-14 profile points.
- Parts must touch/overlap so the object is connected; nothing floats unless the object floats.
- Respect every explicit instruction (e.g. "4 legs", "no armrests", "green handle").
- 3 to 40 parts. Pick colours that match the description (or sensible real colours)."""

MATERIAL_PBR = {  # (metallic, roughness)
    "metal": (0.95, 0.3), "gold": (1.0, 0.25), "silver": (1.0, 0.2), "chrome": (1.0, 0.08),
    "copper": (1.0, 0.3), "steel": (0.95, 0.3), "wood": (0.0, 0.7), "plastic": (0.0, 0.45),
    "ceramic": (0.0, 0.15), "glass": (0.0, 0.05), "fabric": (0.0, 0.95), "leather": (0.0, 0.6),
    "rubber": (0.0, 0.95), "stone": (0.0, 0.9), "paint": (0.0, 0.4),
}


def available() -> bool:
    return bool(config.GEMINI_API_KEY)


def _ask(model: str, prompt: str) -> dict:
    body = {"systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.4, "responseMimeType": "application/json"}}
    r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                      headers={"x-goog-api-key": config.GEMINI_API_KEY}, json=body, timeout=45)
    if r.status_code != 200:
        raise ProviderError(f"planner HTTP {r.status_code}: {r.text[:200]}")
    text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M)
    return json.loads(text)


def plan(prompt: str) -> tuple[dict, str]:
    """-> (plan dict, model used). Tries planner models in order."""
    errs = []
    for m in PLANNER_MODELS:
        try:
            return with_retries(_ask, m, prompt, name=f"planner:{m}", retries=0), m
        except Exception as e:
            errs.append(f"{m}: {str(e)[:120]}")
    raise ProviderError("AI planner unavailable: " + " | ".join(errs))


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
        prof = [(max(_num(a, 0), 0), _num(b, 0)) for a, b in (p.get("profile") or []) if True][:40]
        if len(prof) < 2:
            return None
        prof = sorted(prof, key=lambda t: t[1])
        pts = [(0.0, prof[0][1])] + prof + [(0.0, prof[-1][1])]
        m = trimesh.creation.revolve(np.array(pts), sections=64)
        return _y_up(m)
    if shape == "tube":
        pts = [_vec(q, 3, [0, 0, 0]) for q in (p.get("points") or [])]
        r = s[0]
        if len(pts) < 2:
            return None
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


def _hex(c, default=(150, 150, 150)):
    m = re.fullmatch(r"#?([0-9a-fA-F]{6})", str(c or "").strip())
    return tuple(int(m.group(1)[i:i + 2], 16) for i in (0, 2, 4)) if m else default


def build_plan(spec: dict, out_dir: Path, prompt: str = "") -> tuple[str, list[str]]:
    """Build the planner's parts as a multi-material GLB. Returns (glb path, log lines)."""
    from .prompt import parse as parse_look
    from .sketch3d import apply_material
    look = parse_look(prompt)  # explicit prompt colours win over the model's guesses
    scene = trimesh.Scene()
    logs, n_ok = [], 0
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
        m.apply_transform(trimesh.transformations.euler_matrix(rx, ry, rz, "sxyz"))
        pos = _vec(p.get("position"), 3, [0, 0, 0])
        if str(p.get("shape")).lower() != "tube":  # tube points are already absolute
            m.apply_translation(pos)
        role = str(p.get("role", "body")).lower()
        mat = str(p.get("material", "paint")).lower()
        metal, rough = MATERIAL_PBR.get(mat, (0.0, 0.5))
        rgb = _hex(p.get("color"))
        part_key = {"lid": "cap", "holder": "handle", "grip": "handle"}.get(role, role)
        if part_key in look["parts"]:
            rgb, metal, rough = look["parts"][part_key]
        apply_material(m, rgb, metal, rough)
        scene.add_geometry(m, geom_name=f"{i:02d}_{re.sub(r'[^a-z0-9]+', '_', str(p.get('name', role)).lower())}")
        n_ok += 1
        logs.append(f"{p.get('name', role)}: {p.get('shape')} {p.get('size')} {p.get('color')} {mat}")
    if n_ok == 0:
        raise ProviderError("AI plan contained no buildable parts")
    path = out_dir / f"ai_parts_{uuid.uuid4().hex[:6]}.glb"
    scene.export(path)
    return str(path), logs
