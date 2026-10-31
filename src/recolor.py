"""Recolour chosen parts of a finished model (one or many at once), optionally with a new
material, keeping every other part exactly as it is.

  * textured parts keep their pattern (wood grain, marble veins): the texture's light/dark
    detail is re-tinted to the new colour, or replaced by a new procedural material;
  * flat PBR parts get a new base colour (written in linear space, as glTF expects);
  * vertex-coloured parts (TripoSR, refined parts) are re-tinted per vertex, keeping shading.
Parts are addressed by the names export keeps ("03_handle", "petal_2", "strings"...).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from .sketch3d import linear_to_srgb, srgb_to_linear


def _rgb(hex_or_css: str) -> tuple[int, int, int]:
    s = str(hex_or_css or "").strip()
    m = re.findall(r"[\d.]+", s)
    if s.startswith("rgb") and len(m) >= 3:
        return tuple(int(float(v)) for v in m[:3])
    s = s.lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def label(name: str) -> str:
    """'03_left_handle' -> 'left handle'."""
    n = re.sub(r"^\d+_", "", name)
    return n.replace("_", " ").strip() or name


def current_colour(g: trimesh.Trimesh) -> str:
    """Average visible colour of a part as #rrggbb."""
    v = g.visual
    try:
        if isinstance(v, trimesh.visual.TextureVisuals):
            mat = v.material
            tex = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
            if tex is not None:
                c = np.asarray(tex.convert("RGB")).reshape(-1, 3).mean(0)
            else:
                c = linear_to_srgb(np.asarray(mat.baseColorFactor)[:3])
        elif v.kind in ("vertex", "face"):
            c = np.asarray(v.vertex_colors)[:, :3].mean(0)
        else:
            c = (180, 180, 180)
    except Exception:
        c = (180, 180, 180)
    return "#" + "".join(f"{int(x):02x}" for x in np.clip(c, 0, 255))


def list_parts(glb: str) -> list[dict]:
    """[{name, label, colour, faces}] in a stable order (by name)."""
    scene = trimesh.load(glb, force="scene")
    parts = [g for g in scene.dump() if isinstance(g, trimesh.Trimesh) and len(g.faces)]
    from .export import part_names
    rows = [{"name": n, "label": label(n), "colour": current_colour(g), "faces": len(g.faces)}
            for g, n in zip(parts, part_names(parts))]
    # 'handle 2' -> 'handle' when it is the only handle (the number is just a build index)
    base = lambda l: re.sub(r"[\s_]*\d+$", "", l).strip() or l
    counts: dict[str, int] = {}
    for r in rows:
        counts[base(r["label"])] = counts.get(base(r["label"]), 0) + 1
    for r in rows:
        if counts[base(r["label"])] == 1:
            r["label"] = base(r["label"])
    return sorted(rows, key=lambda r: r["name"])


def groups(parts: list[dict]) -> dict[str, list[str]]:
    """Parts that share a base name ('petal 1'…'petal 8') -> one selectable group."""
    out: dict[str, list[str]] = {}
    for p in parts:
        base = re.sub(r"[\s_]*\d+$", "", p["label"]).strip()
        out.setdefault(base, []).append(p["name"])
    return {f"all {b}s ({len(v)})": v for b, v in out.items() if len(v) > 1}


def _tint(img: np.ndarray, rgb) -> np.ndarray:
    """Keep the texture's detail (luminance around its mean), change its hue."""
    lum = img.astype(np.float32).mean(-1, keepdims=True)
    rel = lum / max(float(lum.mean()), 1.0)
    out = np.asarray(rgb, np.float32)[None, None, :] * np.clip(rel, 0.35, 1.8)
    return np.clip(out, 0, 255).astype(np.uint8)


def recolor(glb: str, out_glb: str | Path, names: list[str], colour: str,
            material: str | None = None) -> dict:
    """Recolour the parts named in `names`. material: a texture/material name ('wood', 'gold',
    'marble', 'glossy plastic'…) to apply too, or None to keep each part's own finish."""
    from . import textures
    from .export import part_names
    rgb = _rgb(colour)
    scene = trimesh.load(glb, force="scene")
    parts = [g for g in scene.dump() if isinstance(g, trimesh.Trimesh) and len(g.faces)]
    pnames = part_names(parts)
    want = set(names)
    out, changed = trimesh.Scene(), []
    mat_name = textures.canonical(material) if material else None
    allv = np.vstack([g.vertices for g in parts])
    tile = float(np.ptp(allv, axis=0).max()) / 3.0  # same texture density as the builders
    for g, n in zip(parts, pnames):
        if n in want:
            g = g.copy()
            v = g.visual
            if mat_name:
                spec = textures.SPECS[mat_name]
                new = textures.apply(g, mat_name, rgb, spec[3], spec[4], tile)
                if new is not None:
                    g = new
            elif isinstance(v, trimesh.visual.TextureVisuals):
                mat = v.material
                tex = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
                if tex is not None and v.uv is not None:
                    img = Image.fromarray(_tint(np.asarray(tex.convert("RGB")), rgb))
                    if hasattr(mat, "baseColorTexture"):
                        mat.baseColorTexture = img
                    else:
                        mat.image = img
                else:
                    a = int(np.asarray(getattr(mat, "baseColorFactor", [0, 0, 0, 255]))[3])
                    mat.baseColorFactor = [*srgb_to_linear(rgb), a]
            elif v.kind in ("vertex", "face"):
                vc = np.asarray(v.vertex_colors).copy()
                vc[:, :3] = _tint(vc[None, :, :3], rgb)[0]
                g.visual = trimesh.visual.ColorVisuals(g, vertex_colors=vc)
            else:
                g.visual = trimesh.visual.ColorVisuals(
                    g, vertex_colors=np.tile([*rgb, 255], (len(g.vertices), 1)).astype(np.uint8))
            changed.append(n)
        g.metadata["name"] = n
        out.add_geometry(g, geom_name=n)
    if not changed:
        raise ValueError("None of the selected parts are in this model.")
    out_glb = Path(out_glb)
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    out.export(out_glb)
    return {"glb": str(out_glb), "changed": changed}
