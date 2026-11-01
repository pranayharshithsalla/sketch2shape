"""OpenSCAD (.scad) Exporter for Sketch2Shape.
Exports clean OpenSCAD code for generated 3D models:
  - Native parametric CSG (rotate_extrude / linear_extrude) when 2D profiles are available
  - Polyhedral mesh export for complex/freeform reconstructed geometries
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import trimesh

from .optimize import decimate_mesh

log = logging.getLogger(__name__)


def export_openscad(
    mesh_or_scene: trimesh.Trimesh | trimesh.Scene,
    out_path: str | Path,
    name: str = "sketch2shape_model",
    max_polyhedron_faces: int = 15000,
    profile_2d: np.ndarray | None = None,
    mode: str = "auto",
) -> str:
    """Exports a 3D model as an OpenSCAD script (.scad).
    If a 2D profile is available (e.g. from revolve/extrude), exports native parametric CSG;
    otherwise exports a polyhedral mesh representation.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(mesh_or_scene, trimesh.Scene):
        meshes = [g for g in mesh_or_scene.dump() if isinstance(g, trimesh.Trimesh) and len(g.faces)]
        if not meshes:
            raise ValueError("Scene has no valid geometry to export")
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = mesh_or_scene

    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]

    lines = [
        "// ===================================================================",
        "// Sketch2Shape OpenSCAD Export (.scad)",
        f"// Model Name: {name}",
        f"// Dimensions (X, Y, Z): {extents[0]:.3f} x {extents[1]:.3f} x {extents[2]:.3f}",
        "// Open in OpenSCAD (https://openscad.org) or FreeCAD",
        "// ===================================================================",
        "",
        "$fn = 72; // Resolution for curved primitives",
        "scale_multiplier = 1.0; // Global scale multiplier",
        "",
    ]

    # Native parametric revolve profile if supplied
    if mode == "revolve" and profile_2d is not None and len(profile_2d) >= 3:
        pts_str = ", ".join(f"[{p[0]:.4f}, {p[1]:.4f}]" for p in profile_2d)
        lines += [
            "// Native Parametric OpenSCAD Lathe (rotate_extrude)",
            f"profile_points = [{pts_str}];",
            "",
            "module model_lathe() {",
            "    scale([scale_multiplier, scale_multiplier, scale_multiplier])",
            "    rotate_extrude(angle=360, convexity=10)",
            "        polygon(points=profile_points);",
            "}",
            "",
            "model_lathe();",
        ]
    elif mode == "extrude" and profile_2d is not None and len(profile_2d) >= 3:
        pts_str = ", ".join(f"[{p[0]:.4f}, {p[1]:.4f}]" for p in profile_2d)
        lines += [
            "// Native Parametric OpenSCAD Extrusion (linear_extrude)",
            f"outline_points = [{pts_str}];",
            f"extrusion_height = {extents[2]:.4f};",
            "",
            "module model_extrusion() {",
            "    scale([scale_multiplier, scale_multiplier, scale_multiplier])",
            "    linear_extrude(height=extrusion_height, center=true, convexity=10)",
            "        polygon(points=outline_points);",
            "}",
            "",
            "model_extrusion();",
        ]
    else:
        # Decimate mesh if too dense for OpenSCAD parser (OpenSCAD struggles >30K faces)
        if len(mesh.faces) > max_polyhedron_faces:
            mesh = decimate_mesh(mesh, target_faces=max_polyhedron_faces)

        verts_str = ",\n    ".join(f"[{v[0]:.5f}, {v[1]:.5f}, {v[2]:.5f}]" for v in mesh.vertices)
        faces_str = ",\n    ".join(f"[{f[0]}, {f[1]}, {f[2]}]" for f in mesh.faces)

        lines += [
            f"// OpenSCAD Mesh Export (polyhedron: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces)",
            "module base_geometry() {",
            "    scale([scale_multiplier, scale_multiplier, scale_multiplier])",
            "    polyhedron(",
            "        points = [",
            f"    {verts_str}",
            "        ],",
            "        faces = [",
            f"    {faces_str}",
            "        ],",
            "        convexity = 10",
            "    );",
            "}",
            "",
            "// Render Model",
            "base_geometry();",
        ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return str(out_path)
