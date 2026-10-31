"""Command line: sketch/photo and/or prompt -> textured .glb (+ .obj zip, .stl, report).

  python -m src.cli --sketch examples/mug.png --prompt "a red ceramic mug" --out mug.glb
  python -m src.cli --prompt "a wooden cricket bat" --decal "BRAND NAME" --out bat.glb
  python -m src.cli --sketch car.png --candidates 4 --out car.glb     # rank 4, build the best
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sketch", help="sketch or photo (PNG/JPG); optional if --prompt is given")
    ap.add_argument("--prompt", default="", help="text prompt (optional with a sketch)")
    ap.add_argument("--out", default="model.glb", help="output .glb path (.obj zip / .stl written beside it)")
    ap.add_argument("--engine", default="auto",
                    help="auto | ai_cad | revolve | extrude | inflate | triposr_local | trellis | local")
    ap.add_argument("--material", default="auto")
    ap.add_argument("--candidates", type=int, default=0,
                    help="generate N ranked candidate images and build the best one with image-to-3D")
    ap.add_argument("--decal", default="", help="text to bake onto the model's front")
    ap.add_argument("--decal-color", default="#111111")
    ap.add_argument("--decal-size", type=float, default=0.5)
    ap.add_argument("--decal-y", type=float, default=0.5)
    ap.add_argument("--decal-rotation", type=float, default=0.0)
    ap.add_argument("--turntable", action="store_true", help="also write a 360° turntable GIF")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--offline", action="store_true", help="no network: local engines only")
    a = ap.parse_args(argv)

    from . import config
    if a.offline:
        config.set_offline()
    if config.use_gpu():
        from . import gpu_local
        gpu_local.register()
    sketch = None
    if a.sketch:
        from PIL import Image
        import numpy as np
        sketch = np.asarray(Image.open(a.sketch).convert("RGBA"))
    if sketch is None and not a.prompt.strip():
        ap.error("give --sketch, --prompt, or both")

    out = Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.candidates:
        from .pipeline import candidate_to_3d, make_candidates
        r = make_candidates(sketch, a.prompt, a.candidates, a.seed)
        for c in r["cands"]:
            print(c["label"])
        best = r["cands"][r["best"]]
        built = candidate_to_3d(best["rgba_path"], r["dir"] / "build", a.engine, a.seed)
        glb, report, timings, engine = built["glb"], built["report"], {**r["timings"], **built["timings"]}, \
            built["provider"]
    else:
        from .pipeline import StageError, run
        try:
            res = run(sketch, a.prompt, a.material, mesh_provider=a.engine, seed=a.seed)
        except StageError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        glb, report, timings, engine = res.glb, res.report, res.timings, res.mesh_provider
    if a.decal:
        from . import decal
        from .export import finalize
        d = Path(glb).parent / "decal"
        baked = decal.bake(glb, d / "baked.glb", decal.text_image(a.decal, a.decal_color, "Heavy"),
                           0.5, a.decal_y, a.decal_size, a.decal_rotation)
        fin = finalize(baked["glb"], d, name="model")
        glb, report = fin["glb"], fin["report"]
    src = Path(glb)
    shutil.copy(src, out)
    for ext, name in ((".stl", src.with_suffix(".stl")), ("_obj.zip", src.with_name(src.stem + "_obj.zip"))):
        if name.exists():
            shutil.copy(name, out.with_name(out.stem + ext))
    if a.turntable:
        from .render import turntable_gif
        turntable_gif(str(out), out.with_suffix(".gif"))
    # prove the written file opens
    import trimesh
    check = trimesh.load(out, force="scene")
    report = {**report, "reopened_faces": int(sum(len(g.faces) for g in check.geometry.values()))}
    print(json.dumps({"out": str(out), "engine": engine, "timings": timings, "validation": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
