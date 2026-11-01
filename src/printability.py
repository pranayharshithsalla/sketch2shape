"""3D Print Pre-Flight & Structural Stability Analyzer for Sketch2Shape.
Evaluates printability metrics:
  - Overhang angles & support requirement estimation
  - Center of Mass (CoM) & tipping/stability margin against the bed contact polygon
  - Manifold / watertight validation
  - Estimated filament mass and print time
  - Heatmap mesh generation (Green = safe, Yellow = caution, Red = needs support)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import scipy.spatial
import trimesh

log = logging.getLogger(__name__)


def _unit_vector(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, np.maximum(norm, 1e-12))


def analyze_printability(
    mesh_or_scene: trimesh.Trimesh | trimesh.Scene,
    max_overhang_deg: float = 45.0,
    assumed_print_height_cm: float = 10.0,
    infill_ratio: float = 0.20,
) -> dict:
    """Performs comprehensive 3D printing pre-flight analysis on a mesh or scene."""
    if isinstance(mesh_or_scene, trimesh.Scene):
        meshes = [g for g in mesh_or_scene.dump() if isinstance(g, trimesh.Trimesh) and len(g.faces)]
        if not meshes:
            return {"status": "EMPTY", "error": "Scene contains no geometric meshes"}
        # Merge for structural/volume analysis
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = mesh_or_scene

    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]
    z_min, z_max = bounds[0][2], bounds[1][2]
    height = max(extents[2], 1e-6)

    # 1. Overhang Angle Analysis
    # Downward normal component (-z). An angle of 0 deg = pointing straight down (-z).
    normals = _unit_vector(mesh.face_normals)
    down_dot = -normals[:, 2]  # cos of angle with straight down vector (0, 0, -1)
    
    # Angle from horizontal print bed (90 deg = vertical wall, 0 deg = horizontal flat overhang ceiling)
    # Surface angle relative to horizontal bed: arcsin(|n_z|) or arccos(down_dot)
    # In 3D printing standards, an overhang angle theta > 45 deg from vertical needs support:
    # i.e., when normal points downward and tilt from vertical exceeds 45 deg.
    is_downward = down_dot > 0.0
    # angle with the vertical axis:
    tilt_deg = np.zeros(len(mesh.faces), dtype=np.float64)
    valid_down = is_downward & (down_dot <= 1.0)
    tilt_deg[valid_down] = np.degrees(np.arcsin(np.clip(down_dot[valid_down], 0.0, 1.0)))

    needs_support_mask = is_downward & (tilt_deg > (90.0 - max_overhang_deg))
    face_areas = mesh.area_faces
    total_area = max(float(face_areas.sum()), 1e-9)
    support_area = float(face_areas[needs_support_mask].sum())
    support_pct = round((support_area / total_area) * 100.0, 1)

    # 2. Center of Mass & Ground Stability Analysis
    try:
        com = mesh.center_mass
    except Exception:
        com = mesh.vertices.mean(axis=0)

    # Base contact vertices: vertices located within 2% of the lowest Z point
    contact_tol = max(height * 0.02, 1e-4)
    contact_mask = mesh.vertices[:, 2] <= (z_min + contact_tol)
    base_points_2d = mesh.vertices[contact_mask][:, :2]

    is_stable = False
    stability_margin = 0.0
    tipping_angle_deg = 0.0
    stability_rating = "UNSTABLE"

    if len(base_points_2d) >= 3:
        try:
            hull = scipy.spatial.ConvexHull(base_points_2d)
            hull_pts = base_points_2d[hull.vertices]

            # Point-in-polygon test for Center of Mass (x, y)
            com_2d = com[:2]
            # Signed distance to all hull edges
            edge_dists = []
            n_hull = len(hull_pts)
            inside = True
            for i in range(n_hull):
                p1 = hull_pts[i]
                p2 = hull_pts[(i + 1) % n_hull]
                edge = p2 - p1
                edge_len = np.linalg.norm(edge)
                if edge_len < 1e-9:
                    continue
                normal = np.array([-edge[1], edge[0]]) / edge_len
                # In standard counter-clockwise convex hull, interior points have dot(pt - p1, normal) >= 0
                dist = np.dot(com_2d - p1, normal)
                edge_dists.append(dist)
                if dist < -1e-6:
                    inside = False

            if inside and edge_dists:
                is_stable = True
                min_dist = max(0.0, min(edge_dists))
                stability_margin = round(float(min_dist), 4)
                # Tipping angle = arctan(margin / h_com)
                h_com = max(com[2] - z_min, 1e-6)
                tipping_angle_deg = round(float(np.degrees(np.arctan(min_dist / h_com))), 1)

                if tipping_angle_deg >= 20.0:
                    stability_rating = "EXCELLENT"
                elif tipping_angle_deg >= 10.0:
                    stability_rating = "STABLE"
                else:
                    stability_rating = "MARGINAL"
            else:
                stability_rating = "UNSTABLE (Will tip over on flat table)"
        except Exception as e:
            log.debug("Stability hull calculation note: %s", e)
            stability_rating = "CANNOT_DETERMINE"

    # 3. Print Volume, Filament Mass & Print Time Estimates
    # Normalized model scaled to assumed_print_height_cm
    scale_factor = assumed_print_height_cm / height
    scaled_volume_cm3 = 0.0
    is_watertight = getattr(mesh, "is_watertight", False)

    if is_watertight:
        try:
            scaled_volume_cm3 = abs(float(mesh.volume)) * (scale_factor ** 3)
        except Exception:
            scaled_volume_cm3 = float(np.prod(extents)) * (scale_factor ** 3) * 0.4
    else:
        # Bounding envelope approximation for open meshes
        scaled_volume_cm3 = float(np.prod(extents)) * (scale_factor ** 3) * 0.35

    # PLA density ~ 1.24 g/cm3, assume 20% infill with 2 perimeters
    effective_density = 1.24 * (infill_ratio + 0.15)
    est_mass_g = round(scaled_volume_cm3 * effective_density, 1)

    # Print time estimate: ~0.8 minutes per gram on standard 50mm/s FDM printers
    est_print_time_mins = round(est_mass_g * 0.9 + 15, 0)

    return {
        "watertight": bool(is_watertight),
        "overhang": {
            "max_overhang_threshold_deg": max_overhang_deg,
            "support_required_area_pct": support_pct,
            "needs_supports": bool(support_pct > 3.0),
        },
        "stability": {
            "center_of_mass": [round(float(c), 4) for c in com],
            "is_stable": is_stable,
            "stability_margin": stability_margin,
            "tipping_angle_deg": tipping_angle_deg,
            "rating": stability_rating,
        },
        "estimate": {
            "print_height_cm": assumed_print_height_cm,
            "volume_cm3": round(scaled_volume_cm3, 1),
            "filament_pla_grams": est_mass_g,
            "estimated_print_time_hours": round(est_print_time_mins / 60.0, 1),
        },
    }


def generate_overhang_heatmap(
    mesh: trimesh.Trimesh,
    out_path: str | Path,
    max_overhang_deg: float = 45.0,
) -> str:
    """Generates an overhang angle inspection mesh colored:
      - Green: safe angles (0 to 30 deg)
      - Yellow: mild overhangs (30 to 45 deg)
      - Red: steep overhangs (> 45 deg, supports needed)
    """
    out_path = Path(out_path)
    normals = _unit_vector(mesh.face_normals)
    down_dot = -normals[:, 2]

    # Initialize face colors to safe green
    face_colors = np.full((len(mesh.faces), 4), [60, 200, 100, 255], dtype=np.uint8)

    # Mild overhang: 30 to 45 deg
    mild_mask = (down_dot > np.sin(np.radians(30))) & (down_dot <= np.sin(np.radians(max_overhang_deg)))
    face_colors[mild_mask] = [240, 200, 50, 255]

    # Steep overhang: > 45 deg
    steep_mask = down_dot > np.sin(np.radians(max_overhang_deg))
    face_colors[steep_mask] = [230, 50, 50, 255]

    colored_mesh = mesh.copy()
    colored_mesh.visual = trimesh.visual.ColorVisuals(face_colors=face_colors)
    colored_mesh.export(out_path)
    return str(out_path)
