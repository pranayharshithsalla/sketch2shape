"""Hard-case accuracy: holes, loops and thin structures must survive into the 3D model.
Offline (Sketch-CAD / inflation + carve/add refinement), ~3-5 min on CPU for all eight cases.

For each case in samples/hard_cases/: the analysis finds the expected through-holes, and after
refinement the exported model keeps them open, reproduces >= 85% of the thin structures, has
silhouette IoU >= 0.9 against the input, and the .glb re-opens with geometry.
"""
import json
from pathlib import Path

import pytest
import trimesh

from src import config
from src.preprocess import preprocess

config.set_offline()
HC = Path(__file__).resolve().parent.parent / "samples" / "hard_cases"
CASES = json.loads((HC / "cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", list(CASES))
def test_analysis_finds_expected_structure(name):
    c = CASES[name]
    a = preprocess(str(HC / f"{name}.png"), text=c["prompt"]).analysis
    assert a.hole_count == c["expected_holes"], a.summary()
    assert bool(a.families) == c["string_grid"], a.summary()


@pytest.mark.parametrize("name", list(CASES))
def test_refined_model_keeps_holes_and_thin_parts(name, monkeypatch):
    from src.pipeline import run
    monkeypatch.setattr(config, "ACCURACY_RETRY", False)  # one engine + refinement, keeps the test fast
    c = CASES[name]
    r = run(str(HC / f"{name}.png"), c["prompt"], mesh_provider="sketch_cad", run_name=f"pytest_hard_{name}")
    a = r.accuracy
    assert a is not None, "structural check did not run"
    assert a["holes_open"] == c["expected_holes"] and a["holes_extra"] == 0, a
    assert a["thin_recall"] >= 0.85, a
    assert a["iou"] >= 0.9, a
    scene = trimesh.load(r.glb, force="scene")
    assert sum(len(g.faces) for g in scene.geometry.values()) > 100
