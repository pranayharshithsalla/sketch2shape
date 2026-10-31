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
