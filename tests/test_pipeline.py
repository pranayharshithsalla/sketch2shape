"""Smoke tests: run with `python -m pytest -q` (offline, ~1 min on CPU; no API keys needed)."""
import numpy as np
import pytest
import trimesh

from src import config
from src.export import finalize
from src.preprocess import PreprocessError, preprocess
from src.prompt import parse
from src.sketch3d import build, parse_modifiers

config.set_offline()
EX = config.EXAMPLES


def test_prompt_colours_and_parts():
    p = parse("a navy blue cup, handle in white")
    assert p["body"][0] == (0, 0, 128) and p["parts"]["handle"][0] == (240, 240, 236)
    assert parse("a gold metal bottle")["body"][1] == 1.0          # gold beats generic metal
    assert parse("a #ff8800 vase")["body"][0] == (255, 136, 0)
    assert "cap" in parse("a bottle with a silver cap")["parts"]


def test_shape_modifiers():
    m = parse_modifiers("a tall slim hexagonal bottle without handle, glossy")
    assert m["sections"] == 6 and m["height"] > 1 and m["width"] < 1 and m["handle"] == 0
    assert m["finish"] == "glossy"


def test_empty_canvas_is_a_clear_error():
    with pytest.raises(PreprocessError):
        preprocess(np.full((256, 256, 4), 0, np.uint8))


@pytest.mark.parametrize("name,prompt,mode", [
    ("mug", "a red mug with a black handle", "revolve"),
    ("star", "a gold star", "extrude"),
    ("teddy", "a brown teddy bear", "inflate"),
])
def test_sketch_cad_exports_valid_glb(tmp_path, name, prompt, mode):
    sk = preprocess(EX / f"{name}.png")
    raw, used, _ = build(sk.mask, tmp_path, mode, prompt, strokes=sk.scribble > 0)
    out = finalize(raw, tmp_path)
    rep = out["report"]
    assert used == mode and rep["faces"] > 100 and rep["watertight"] and rep["glb_reopens"]
    s = trimesh.load(out["glb"], force="scene")
    assert sum(len(g.faces) for g in s.geometry.values()) == rep["faces"]
    if name == "mug":
        assert rep["parts"] == 2  # body + separately coloured handle


@pytest.mark.parametrize("prompt,parts", [
    ("a brown bottle with a green holder", {"body", "handle"}),
    ("a white teapot with a gold lid and blue spout", {"body", "cap", "handle", "spout"}),
    ("a copper gear", {"body"}),
])
def test_prompt_only_creation(tmp_path, prompt, parts):
    """No sketch at all: the prompt's template is built with exact parts and colours."""
    from src.pipeline import run
    r = run(None, prompt, run_name=f"pytest_{prompt.split()[2]}")
    assert r.report["watertight"] and r.report["parts"] == len(parts)
    assert any("template from the prompt" in l for l in r.logs)


def test_no_sketch_and_unknown_object_is_a_clear_error():
    from src.pipeline import StageError, run
    with pytest.raises(StageError, match="Or type a prompt naming one of"):
        run(None, "a purple dragon", run_name="pytest_unknown")


def test_ai_plan_builds_multimaterial_glb(tmp_path):
    """Offline: a hand-written planner output (what the LLM returns) builds into a valid GLB."""
    from src.ai_assembly import _request_text, build_plan
    spec = {"object": "table", "parts": [
        {"name": "top", "role": "body", "shape": "box", "size": [100, 4, 60], "position": [0, 72, 0],
         "color": "#8b5a2b", "material": "wood"},
        *[{"name": f"leg {i}", "role": "leg", "shape": "cylinder", "size": [3, 70], "color": "#222222",
           "material": "metal", "position": [x, 35, z]} for i, (x, z) in
          enumerate([(-45, -25), (45, -25), (-45, 25), (45, 25)])],
        {"name": "vase", "role": "detail", "shape": "lathe", "position": [0, 74, 0], "color": "#2d5fcd",
         "profile": [[4, 0], [7, 8], [3, 18], [4, 22]]},
    ]}
    raw, logs = build_plan(spec, tmp_path, "a table")
    rep = finalize(raw, tmp_path)["report"]
    assert rep["parts"] == 6 and rep["watertight"] and rep["faces"] > 100
    assert "CHANGE REQUEST: add a drawer" in _request_text("add a drawer", spec, [("a table", "done")])


def test_ai_cad_helpers_offline(tmp_path):
    """Hollow lathe walls from the model keep their order; explicit body colour wins; organic routing."""
    from src.ai_assembly import build_plan
    from src.pipeline import is_organic
    spec = {"object": "mug", "parts": [
        {"name": "mug body", "role": "body", "shape": "lathe", "color": "#123456", "position": [0, 0, 0],
         "profile": [[0, 0], [4, 0], [4.5, 10], [4.0, 10], [3.6, 0.6], [0, 0.6]]},  # outer up, inner down
        {"name": "handle", "role": "handle", "shape": "tube", "size": [0.5], "color": "#000000",
         "points": [[4.3, 8, 0], [7, 7, 0], [7, 3, 0], [4.2, 2, 0]]}]}
    raw, _ = build_plan(spec, tmp_path, "a violet mug with a black handle")
    rep = finalize(raw, tmp_path)["report"]
    assert rep["parts"] == 2 and rep["watertight"]
    s = trimesh.load(raw, force="scene")
    body = next(g for k, g in s.geometry.items() if "body" in k)
    assert tuple(body.visual.material.baseColorFactor[:3]) != (0x12, 0x34, 0x56)  # violet, not the model's
    assert is_organic("a brown teddy bear") and not is_organic("a gold trophy")


@pytest.mark.parametrize("material", ["wood", "marble", "gold", "leather", "glass", "fabric", "stone"])
def test_material_textures(tmp_path, material):
    """Every material gets a real texture, UVs survive export, seams don't break watertightness."""
    from src.textures import SPECS, apply
    assert material in SPECS
    sk = preprocess(EX / "mug.png")
    raw, _, logs = build(sk.mask, tmp_path, "revolve", f"a {material} mug", strokes=sk.scribble > 0)
    assert any(f"body:{material}" in l for l in logs)
    out = finalize(raw, tmp_path)
    assert out["report"]["watertight"] and "texture" in out["report"]["colour"]
    box = apply(trimesh.creation.box(extents=[2, 1, 1]), material, (120, 80, 40), 0, 0.5, 1.0, "planar")
    assert box.visual.uv.shape == (len(box.vertices), 2)
