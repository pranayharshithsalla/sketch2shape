"""Tests for ranking, decal baking, sessions/versions and the REST API (offline, no keys)."""
import numpy as np
import pytest
import trimesh
from PIL import Image

from src import config, decal, designer
from src.pipeline import run
from src.rank import rank, readiness

config.set_offline()
EX = config.EXAMPLES


def _disc(size=256, r=80, cx=128, cy=128):
    yy, xx = np.mgrid[:size, :size]
    return (xx - cx) ** 2 + (yy - cy) ** 2 < r * r


def _rgba(mask, color=(200, 60, 50)):
    img = np.zeros(mask.shape + (4,), np.uint8)
    img[mask] = (*color, 255)
    return img


# ---------------------------------------------------------------- ranking
def test_readiness_penalises_cropped_and_fragmented():
    good = readiness(_rgba(_disc()))
    cropped = readiness(_rgba(_disc(cx=10)))                     # cut by the frame edge
    split = readiness(_rgba(_disc(r=30, cx=60) | _disc(r=30, cx=200)))  # two pieces
    assert good["ready"] > 0.9
    assert cropped["uncropped"] < good["uncropped"] and cropped["ready"] < good["ready"]
    assert split["single"] < 0.6 and split["ready"] < good["ready"]


def test_rank_prefers_the_candidate_matching_the_sketch():
    sketch = _disc()
    strokes = (np.logical_xor(sketch, _disc(r=76)) * 255).astype(np.uint8)
    square = np.zeros_like(sketch)
    square[40:220, 30:230] = True
    best, scores = rank(sketch, strokes, [_rgba(square), _rgba(_disc(r=78)), _rgba(_disc(cx=8))])
    assert best == 1
    assert all({"score", "sketch", "ready", "prompt"} <= set(s) for s in scores)
    assert scores[0]["prompt"] is None  # no prompt: weight renormalised, not guessed


# ---------------------------------------------------------------- decals
@pytest.fixture(scope="module")
def mug(tmp_path_factory):
    d = tmp_path_factory.mktemp("mug")
    return run(str(EX / "mug.png"), "a white ceramic mug", mesh_provider="revolve", run_name=f"pytest_{d.name}")


def test_decal_is_baked_into_the_texture_on_the_front_only(mug, tmp_path):
    out = decal.bake(mug.glb, tmp_path / "d.glb", decal.text_image("TEST", "#000000"), x=0.5, y=0.5, size=0.5)
    assert out["texels"] > 1000
    scene = trimesh.load(out["glb"], force="scene")
    # one ordinary textured mesh per part: no extra decal geometry
    assert len(scene.geometry) == len(trimesh.load(mug.glb, force="scene").geometry)
    body = max(scene.geometry.values(), key=lambda g: g.area)
    tex = np.asarray(body.visual.material.baseColorTexture.convert("RGB"))
    w = tex.shape[1]
    dark_front = (tex[:, : w // 2].max(-1) < 60).sum()
    dark_back = (tex[:, w // 2:].max(-1) < 60).sum()
    assert dark_front > 500 and dark_back < 0.02 * dark_front


def test_decal_that_misses_the_model_is_a_clear_error(mug, tmp_path):
    with pytest.raises(ValueError):
        decal.bake(mug.glb, tmp_path / "x.glb", decal.text_image("X"), x=3.0, y=3.0, size=0.1)


# ---------------------------------------------------------------- sessions / versions
def test_versions_revert_and_decals(mug):
    s = designer.Session()
    designer.add_version(s, mug.glb, [mug.glb], mug.report, "a mug", "revolve", "sketch", 1.0)
    s, msg = designer.add_decal(s, "HELLO", color="#000000", size=0.5)
    assert len(s.versions) == 2 and s.current["kind"] == "decal" and "HELLO" in msg
    s, _ = designer.add_decal(s, "MOVED", color="#000000", x=0.4, replace_last=True)  # re-place, not stack
    assert len(s.versions) == 2 and [d["text"] for d in s.current["decals"]] == ["MOVED"]
    s, _ = designer.revert(s, 0)
    assert len(s.versions) == 3 and s.current["glb"] == mug.glb and s.current["kind"] == "revert"
    tl = designer.timeline(s)
    assert tl[0].startswith("v1") and "revert" in tl[-1]


# ---------------------------------------------------------------- REST API
def test_rest_api_generate_decal_and_ar(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src import api
    app = FastAPI()
    api.register(app)
    c = TestClient(app)
    assert c.get("/api/health").json()["ok"]
    assert c.post("/api/generate", data={"prompt": ""}).status_code == 400
    with open(EX / "mug.png", "rb") as f:
        r = c.post("/api/generate", files={"sketch": ("mug.png", f, "image/png")},
                   data={"prompt": "a blue mug", "engine": "revolve"}).json()
    assert r["validation"]["glb_reopens"] and r["files"]["glb"].endswith(".glb")
    glb = c.get(r["files"]["glb"])
    assert glb.status_code == 200 and glb.content[:4] == b"glTF"
    assert "model-viewer" in c.get(r["ar"]).text
    d = c.post("/api/decal", data={"model": r["files"]["glb"], "text": "API"}).json()
    assert "texture" in d["validation"]["colour"]
    assert c.get("/models/../.env").status_code == 404
