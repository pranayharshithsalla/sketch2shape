"""LOD Mesh Decimation and Optimization for Sketch2Shape.
Transforms dense voxel-carved or generative meshes (often 500K - 2M faces) into
optimized, lightweight models suitable for real-time WebXR, mobile AR, and games.
Supports fast_simplification when installed, with a robust NumPy fallback.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import trimesh

log = logging.getLogger(__name__)

# Standard target face counts for Level of Detail (LOD)
LOD_PROFILES = {
    "high": 40000,    # High quality desktop / cinematic
    "medium": 12000,  # Standard web 3D / AR viewer
    "low": 3000,      # Mobile gaming / distant LOD
}

try:
    import fast_simplification
    HAS_FAST_SIMPLIFICATION = True
except ImportError:
    HAS_FAST_SIMPLIFICATION = False


def decimate_mesh(mesh: trimesh.Trimesh, target_faces: int = 15000, max_iterations: int = 8) -> trimesh.Trimesh:
    """Decimates a mesh down to target_faces.
    Uses fast_simplification if available, otherwise runs a fast, linear-time
    spatial clustering decimation in pure NumPy/SciPy.
    """
    if len(mesh.faces) <= target_faces or len(mesh.vertices) < 10:
        return mesh.copy()

    # Try fast_simplification if available
    if HAS_FAST_SIMPLIFICATION:
        try:
            reduction = 1.0 - (float(target_faces) / len(mesh.faces))
            if 0.0 < reduction < 1.0:
                v, f = fast_simplification.simplify(
                    mesh.vertices.astype(np.float32),
                    mesh.faces.astype(np.int32),
                    target_reduction=float(reduction),
                )
                cand = trimesh.Trimesh(vertices=v, faces=f, process=True)
                if hasattr(mesh.visual, "material"):
                    cand.visual = mesh.visual.copy()
                elif hasattr(mesh.visual, "vertex_colors"):
                    cand.visual = mesh.visual.copy()
                return cand
        except Exception as e:
            log.debug("fast_simplification failed, falling back to NumPy: %s", e)

    # Pure NumPy spatial clustering decimation
    bbox = mesh.bounds[1] - mesh.bounds[0]
    diag = float(np.linalg.norm(bbox))
    if diag <= 1e-6:
        return mesh.copy()

    low = diag * 0.0003
    high = diag * 0.2
    best_mesh = mesh
    best_diff = float("inf")

    has_v_colors = (
        isinstance(mesh.visual, trimesh.visual.ColorVisuals)
        and mesh.visual.kind == "vertex"
        and mesh.visual.vertex_colors is not None
        and len(mesh.visual.vertex_colors) == len(mesh.vertices)
    )

    for _ in range(max_iterations):
        mid = (low + high) / 2.0
        grid = np.floor((mesh.vertices - mesh.bounds[0]) / mid).astype(np.int64)
        _, inv, counts = np.unique(grid, axis=0, return_inverse=True, return_counts=True)
        n_clusters = len(counts)

        new_verts = np.zeros((n_clusters, 3), dtype=np.float64)
        np.add.at(new_verts, inv, mesh.vertices)
        new_verts /= counts[:, None]

        new_faces = inv[mesh.faces]
        valid = (
            (new_faces[:, 0] != new_faces[:, 1])
            & (new_faces[:, 1] != new_faces[:, 2])
            & (new_faces[:, 0] != new_faces[:, 2])
        )
        new_faces = new_faces[valid]

        if len(new_faces) == 0:
            high = mid
            continue

        sorted_faces = np.sort(new_faces, axis=1)
        _, unique_idx = np.unique(sorted_faces, axis=0, return_index=True)
        new_faces = new_faces[unique_idx]

        curr_faces = len(new_faces)
        diff = abs(curr_faces - target_faces)

        cand = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=True)

        if has_v_colors:
            new_colors = np.zeros((n_clusters, 4), dtype=np.float64)
            np.add.at(new_colors, inv, mesh.visual.vertex_colors.astype(np.float64))
            new_colors /= counts[:, None]
            cand.visual = trimesh.visual.ColorVisuals(vertex_colors=np.clip(new_colors, 0, 255).astype(np.uint8))
        elif hasattr(mesh.visual, "material"):
            cand.visual = mesh.visual.copy()

        if diff < best_diff:
            best_diff = diff
            best_mesh = cand

        if curr_faces > target_faces:
            low = mid
        else:
            high = mid

    return best_mesh


def optimize_scene(scene_or_mesh, target_faces: int = 15000) -> trimesh.Scene | trimesh.Trimesh:
    """Optimizes an entire multi-part scene or standalone mesh proportionally."""
    if isinstance(scene_or_mesh, trimesh.Trimesh):
        return decimate_mesh(scene_or_mesh, target_faces=target_faces)

    scene = scene_or_mesh
    geoms = {k: g for k, g in scene.geometry.items() if isinstance(g, trimesh.Trimesh) and len(g.faces)}
    total_faces = sum(len(g.faces) for g in geoms.values())
    if total_faces <= target_faces or total_faces == 0:
        return scene.copy()

    out_scene = trimesh.Scene()
    ratio = target_faces / total_faces

    for name, geom in geoms.items():
        part_target = max(60, int(len(geom.faces) * ratio))
        dec = decimate_mesh(geom, target_faces=part_target)
        dec.metadata.update(geom.metadata)
        out_scene.add_geometry(dec, geom_name=name)

    return out_scene


def generate_lods(glb_path: str | Path, out_dir: str | Path | None = None, profiles: dict | None = None) -> dict:
    """Takes a .glb file and creates optimized LOD models alongside it.
    Returns metadata with face counts, file sizes, and compression ratios.
    """
    glb_path = Path(glb_path)
    out_dir = Path(out_dir) if out_dir else glb_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    profiles = profiles or LOD_PROFILES

    scene = trimesh.load(glb_path, force="scene")
    orig_size_bytes = os.path.getsize(glb_path)
    orig_faces = sum(len(g.faces) for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh))
    orig_verts = sum(len(g.vertices) for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh))

    results = {
        "original": {
            "path": str(glb_path),
            "faces": orig_faces,
            "vertices": orig_verts,
            "size_kb": round(orig_size_bytes / 1024, 1),
        },
        "lods": {},
    }

    t0 = time.time()
    for lod_name, target in profiles.items():
        if orig_faces <= target:
            continue
        opt_scene = optimize_scene(scene, target_faces=target)
        lod_file = out_dir / f"{glb_path.stem}_lod_{lod_name}.glb"
        opt_scene.export(lod_file)

        lod_size = os.path.getsize(lod_file)
        lod_faces = sum(len(g.faces) for g in opt_scene.geometry.values() if isinstance(g, trimesh.Trimesh))
        lod_verts = sum(len(g.vertices) for g in opt_scene.geometry.values() if isinstance(g, trimesh.Trimesh))

        reduction_pct = round((1.0 - (lod_size / max(orig_size_bytes, 1))) * 100, 1)

        results["lods"][lod_name] = {
            "path": str(lod_file),
            "faces": lod_faces,
            "vertices": lod_verts,
            "size_kb": round(lod_size / 1024, 1),
            "face_reduction_pct": round((1.0 - (lod_faces / max(orig_faces, 1))) * 100, 1),
            "size_reduction_pct": reduction_pct,
        }

    results["optimization_time_s"] = round(time.time() - t0, 2)
    return results
