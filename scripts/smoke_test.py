"""End-to-end smoke test: sample sketch -> ranked candidates -> textured .glb -> decal -> re-open.
Runs fully offline (local engines only), so it works from a clean clone with no API keys.

    python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    import numpy as np
    import trimesh
    from PIL import Image

    from src import config, decal
    from src.export import finalize
    from src.pipeline import make_candidates, run

    config.set_offline()
    out = config.OUTPUTS / "smoke"
    sketch = np.asarray(Image.open(ROOT / "examples" / "mug.png").convert("RGBA"))
    checks = []

    t = time.time()
    c = make_candidates(sketch, "a red ceramic mug", 4, out_dir=out / "cands")
    checks.append(("4 ranked candidates", len(c["cands"]) == 4 and 0 <= c["best"] < 4))

    r = run(sketch, "a red ceramic mug", mesh_provider="sketch_cad", run_name="smoke/run")
    scene = trimesh.load(r.glb, force="scene")
    faces = sum(len(g.faces) for g in scene.geometry.values())
    checks.append(("sketch -> .glb re-opens with faces", faces > 100))
    checks.append(("texture/colour preserved", r.report["colour"] in ("texture", "material", "vertex colours")))
    checks.append(("watertight (printable)", bool(r.report["watertight"])))
    checks.append((".obj zip and .stl written", Path(r.obj_zip).exists() and Path(r.stl).exists()))

    baked = decal.bake(r.glb, out / "decal" / "baked.glb", decal.text_image("SMOKE", "#FFFFFF"), size=0.5)
    fin = finalize(baked["glb"], out / "decal", name="model")
    tex = [g.visual.material.baseColorTexture for g in trimesh.load(fin["glb"], force="scene").geometry.values()
           if hasattr(g.visual, "material") and getattr(g.visual.material, "baseColorTexture", None) is not None]
    checks.append(("decal baked into an exported texture", baked["texels"] > 1000 and len(tex) > 0))

    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"total {time.time() - t:.1f} s · model: {fin['glb']}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
