"""Stages 7-8: mesh repair, validation and multi-format export (.glb/.obj/.stl) with a
health report. Textures/vertex colours are preserved; the GLB is re-opened to verify."""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import numpy as np
import trimesh

log = logging.getLogger(__name__)


class ExportError(RuntimeError):
    pass


def _has_colour(g: trimesh.Trimesh) -> str:
    v = g.visual
    if isinstance(v, trimesh.visual.TextureVisuals):
        mat = getattr(v, "material", None)
        img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
        return "texture" if img is not None else "material"
    if isinstance(v, trimesh.visual.ColorVisuals) and v.kind in ("vertex", "face"):
        return f"{v.kind} colours"
    return "none"


def part_names(meshes) -> list[str]:
    """Stable, unique names for the parts (the planner's '03_handle' etc. survive every export,
    so parts can be picked by name later, e.g. to recolour them)."""
    out, seen = [], set()
    for i, m in enumerate(meshes):
        n = str(m.metadata.get("name") or f"part_{i}")
        base, k = n, 2
        while n in seen:
            n, k = f"{base}_{k}", k + 1
        seen.add(n)
        out.append(n)
    return out


def geometry_only(g: trimesh.Trimesh) -> trimesh.Trimesh:
    """Same faces (same order) with duplicate vertices merged. Textured meshes duplicate
    vertices along UV seams, which makes a closed solid *look* open to connectivity checks."""
    m = trimesh.Trimesh(g.vertices.copy(), g.faces.copy(), process=False)
    m.merge_vertices()
    return m


def is_closed(g: trimesh.Trimesh) -> bool:
    return geometry_only(g).is_watertight


def _repair(g: trimesh.Trimesh, min_frac: float = 0.01) -> trimesh.Trimesh:
    textured = isinstance(g.visual, trimesh.visual.TextureVisuals) and g.visual.uv is not None
    if not textured:  # merging would break UV seams
        g.merge_vertices()
    g.update_faces(g.nondegenerate_faces())
    g.update_faces(g.unique_faces())
    g.remove_unreferenced_vertices()
    # Drop floating specks (< 1% of faces) left by reconstruction; components are found on
    # the merged geometry so UV seams / per-face UV islands don't count as separate pieces
    geo = geometry_only(g)
    labels = trimesh.graph.connected_component_labels(geo.face_adjacency, node_count=len(geo.faces))
    sizes = np.bincount(labels)
    if len(sizes) > 1:
        keep = sizes[labels] >= min_frac * len(g.faces)
        if keep.any() and not keep.all():
            g.update_faces(keep)
            g.remove_unreferenced_vertices()
            geo = geometry_only(g)
    if not textured:
        trimesh.repair.fix_normals(g)
        if not g.is_watertight:
            trimesh.repair.fill_holes(g)
    return g


def finalize(raw_glb: str, out_dir: Path, name: str = "model") -> dict:
    """Repair, normalise to a unit box centred at the origin, export and validate."""
    try:
        scene = trimesh.load(raw_glb, force="scene")
    except Exception as e:
        raise ExportError(f"could not open generated mesh: {e}")
    geoms = {k: g for k, g in scene.geometry.items() if isinstance(g, trimesh.Trimesh) and len(g.faces)}
    if not geoms:
        raise ExportError("generated mesh is empty (0 faces)")

    # Each part keeps its own material (body / cap / handle ...), so parts stay separate meshes
    meshes = []
    for g in scene.dump():  # applies scene transforms
        if isinstance(g, trimesh.Trimesh) and len(g.faces):
            meshes.append(_repair(g))
    lo = np.min([m.bounds[0] for m in meshes], axis=0)
    hi = np.max([m.bounds[1] for m in meshes], axis=0)
    scale = 1.0 / max((hi - lo).max(), 1e-9)
    out = trimesh.Scene()
    for m, name in zip(meshes, part_names(meshes)):
        m.apply_translation(-(lo + hi) / 2)
        m.apply_scale(scale)
        m.metadata["name"] = name
        out.add_geometry(m, geom_name=name)
    geometry_only = trimesh.util.concatenate([trimesh.Trimesh(m.vertices, m.faces, process=False)
                                              for m in meshes])

    out_dir.mkdir(parents=True, exist_ok=True)
    glb = out_dir / f"{name}.glb"
    stl = out_dir / f"{name}.stl"
    obj_zip = out_dir / f"{name}_obj.zip"
    (out if len(meshes) > 1 else meshes[0]).export(glb)
    geometry_only.export(stl)
    obj_dir = out_dir / "obj"
    obj_dir.mkdir(exist_ok=True)
    try:
        (out if len(meshes) > 1 else meshes[0]).export(obj_dir / f"{name}.obj")
    except Exception:  # some exporters cannot write multi-material scenes: geometry still ships
        geometry_only.export(obj_dir / f"{name}.obj")
    with zipfile.ZipFile(obj_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in obj_dir.iterdir():
            z.write(f, f.name)

    # Validation: re-open what we wrote
    check = trimesh.load(glb, force="scene")
    n_v = sum(len(g.vertices) for g in check.geometry.values())
    n_f = sum(len(g.faces) for g in check.geometry.values())
    if n_v == 0 or n_f == 0:
        raise ExportError("exported GLB re-opened with empty geometry")
    colour = ", ".join(sorted({_has_colour(g) for g in check.geometry.values()}))
    report = {
        "vertices": int(n_v),
        "faces": int(n_f),
        "extents": [round(float(x), 3) for x in (hi - lo) * scale],
        "watertight": all(is_closed(m) for m in meshes),
        "parts": len(meshes),
        "colour": colour,
        "glb_reopens": True,
        "glb_size_kb": round(glb.stat().st_size / 1024, 1),
    }
    return {"glb": str(glb), "obj_zip": str(obj_zip), "stl": str(stl), "report": report}
