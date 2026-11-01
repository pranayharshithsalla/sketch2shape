"""Unit tests for newly added features:
  - LOD mesh decimation & optimization (src/optimize.py)
  - 3D print pre-flight & structural stability (src/printability.py)
  - OpenSCAD export (src/cad_export.py)
"""
from pathlib import Path
import numpy as np
import pytest
import trimesh

from src.cad_export import export_openscad
from src.optimize import decimate_mesh, generate_lods, optimize_scene
from src.printability import analyze_printability, generate_overhang_heatmap


@pytest.fixture
def sample_sphere():
    return trimesh.creation.icosphere(subdivisions=4, radius=1.0)


@pytest.fixture
def sample_scene():
    s = trimesh.Scene()
    m1 = trimesh.creation.cylinder(radius=0.5, height=2.0)
    m2 = trimesh.creation.box(extents=[1.0, 1.0, 0.2])
    m2.apply_translation([0, 0, -1.0])
    s.add_geometry(m1, geom_name="cylinder")
    s.add_geometry(m2, geom_name="base")
    return s


def test_mesh_decimation(sample_sphere):
    orig_faces = len(sample_sphere.faces)
    assert orig_faces > 2000

    target = 600
    decimated = decimate_mesh(sample_sphere, target_faces=target)
    assert len(decimated.faces) < orig_faces
    assert abs(len(decimated.faces) - target) < 250
    assert len(decimated.vertices) > 0


def test_optimize_scene(sample_sphere):
    """Verifies that scene decimation strictly reduces face count towards the target."""
    scene = trimesh.Scene()
    scene.add_geometry(sample_sphere, geom_name="sphere")
    orig_faces = sum(len(g.faces) for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh))
    assert orig_faces == 5120

    target = 800
    opt = optimize_scene(scene, target_faces=target)
    new_faces = sum(len(g.faces) for g in opt.geometry.values() if isinstance(g, trimesh.Trimesh))

    # Assert strictly reduced and close to target
    assert new_faces < orig_faces
    assert abs(new_faces - target) < 250
    assert len(opt.geometry) == 1


def test_generate_lods(sample_sphere, tmp_path):
    glb_path = tmp_path / "test_model.glb"
    sample_sphere.export(glb_path)

    res = generate_lods(glb_path, out_dir=tmp_path, profiles={"low": 500, "medium": 1500})
    assert "lods" in res
    assert "low" in res["lods"]
    assert Path(res["lods"]["low"]["path"]).exists()
    assert res["lods"]["low"]["faces"] < len(sample_sphere.faces)


def test_printability_analysis_stable_and_unstable():
    """Tests both stable base-supported shapes and unstable off-balance / top-heavy shapes."""
    # 1. Stable shape: wide base, center of mass directly centered
    base = trimesh.creation.cylinder(radius=2.0, height=0.5)
    post = trimesh.creation.cylinder(radius=0.3, height=2.0)
    post.apply_translation([0, 0, 1.25])
    stable_mesh = trimesh.util.concatenate([base, post])

    res_stable = analyze_printability(stable_mesh, max_overhang_deg=45.0, up_axis="z")
    assert res_stable["stability"]["is_stable"] is True
    assert res_stable["stability"]["rating"] in ("EXCELLENT", "STABLE")
    assert res_stable["stability"]["stability_margin"] > 0.5
    assert res_stable["estimate"]["volume_cm3"] > 0

    # 2. Unstable shape: tiny base at origin with heavy cantilevered mass far to the right (+X)
    foot = trimesh.creation.box(extents=[0.2, 0.2, 0.1])  # Base footprint at X in [-0.1, 0.1]
    cantilever = trimesh.creation.box(extents=[4.0, 1.0, 1.0])
    cantilever.apply_translation([3.0, 0, 2.0])  # Heavy mass out at X ~ 3.0
    unstable_mesh = trimesh.util.concatenate([foot, cantilever])

    res_unstable = analyze_printability(unstable_mesh, max_overhang_deg=45.0, up_axis="z")
    assert res_unstable["stability"]["is_stable"] is False
    assert res_unstable["stability"]["rating"].startswith("UNSTABLE")
    assert res_unstable["stability"]["stability_margin"] == 0.0


def test_overhang_heatmap(sample_scene, tmp_path):
    merged = trimesh.util.concatenate(list(sample_scene.geometry.values()))
    out_file = tmp_path / "heatmap.glb"
    p = generate_overhang_heatmap(merged, out_file, up_axis="z")
    assert Path(p).exists()
    loaded = trimesh.load(p, force="mesh")
    assert isinstance(loaded.visual, trimesh.visual.ColorVisuals)
    assert len(loaded.visual.face_colors) == len(merged.faces)


def test_openscad_export_parametric_and_mesh(sample_scene, tmp_path):
    """Tests honest export: native CSG for 2D profiles, and polyhedral mesh export for 3D meshes."""
    # 1. Native parametric revolve
    revolve_file = tmp_path / "lathe.scad"
    profile = np.array([[0.0, 0.0], [0.5, 0.0], [0.6, 1.0], [0.0, 1.0]])
    export_openscad(sample_scene, revolve_file, name="lathe_test", mode="revolve", profile_2d=profile)
    assert revolve_file.exists()
    revolve_code = revolve_file.read_text(encoding="utf-8")
    assert "rotate_extrude" in revolve_code
    assert "polygon(points=profile_points)" in revolve_code
    assert "Native Parametric OpenSCAD Lathe" in revolve_code

    # 2. Native parametric linear extrusion
    extrude_file = tmp_path / "extrusion.scad"
    outline = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    export_openscad(sample_scene, extrude_file, name="extrude_test", mode="extrude", profile_2d=outline)
    assert extrude_file.exists()
    extrude_code = extrude_file.read_text(encoding="utf-8")
    assert "linear_extrude" in extrude_code
    assert "polygon(points=outline_points)" in extrude_code
    assert "Native Parametric OpenSCAD Extrusion" in extrude_code

    # 3. Polyhedral mesh export (honestly labelled as mesh export, not parametric)
    mesh_file = tmp_path / "mesh.scad"
    export_openscad(sample_scene, mesh_file, name="mesh_test")
    assert mesh_file.exists()
    mesh_code = mesh_file.read_text(encoding="utf-8")
    assert "polyhedron" in mesh_code
    assert "OpenSCAD Mesh Export" in mesh_code
