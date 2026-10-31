"""Run the full pipeline over every sketch in examples/ (or given paths) and write a
benchmark table (timings, fidelity, mesh stats) to outputs/benchmark.md and .csv.

    python batch_run.py                       # all examples, auto providers
    python batch_run.py --offline             # force the key-free local path
    python batch_run.py examples/mug.png --prompt "a red mug"
"""
from __future__ import annotations

import argparse
import csv
import platform
from pathlib import Path

from src import config
from src.pipeline import run
from src.render import turntable

PROMPTS = {"mug": "a red ceramic mug with a black handle", "cup": "", "star": "a gold star", "chair": "a wooden chair", "car": "a red car", "teddy": "a brown teddy bear"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("-n", type=int, default=config.NUM_CANDIDATES)
    a = ap.parse_args()
    if a.offline:
        config.set_offline()
    paths = [Path(p) for p in a.paths] or sorted(config.EXAMPLES.glob("*.png"))
    rows = []
    for p in paths:
        prompt = a.prompt if a.prompt is not None else PROMPTS.get(p.stem, "")
        tag = "offline" if a.offline else "auto"
        print(f"=== {p.name} (prompt={prompt!r}, {tag})")
        try:
            r = run(str(p), prompt, n_candidates=a.n, run_name=f"batch_{tag}_{p.stem}")
            turntable(r.glb, r.run_dir / "turntable.png")
            best = r.scores[r.best]
            rows.append({"sketch": p.name, "prompt": prompt or "(auto-caption)", "image": r.image_provider,
                         "3d": r.mesh_provider, "fidelity": best["score"], "iou": best["iou"],
                         "candidates": len(r.scores), "parts": r.report.get("parts", 1), "verts": r.report["vertices"], "faces": r.report["faces"],
                         "watertight": r.report["watertight"], "colour": r.report["colour"],
                         "t_2d": r.timings.get("Sketch-to-image"), "t_3d": r.timings.get("Image-to-3D"),
                         "t_total": r.timings["total"], "status": "ok"})
        except Exception as e:
            print("FAILED:", e)
            rows.append({"sketch": p.name, "prompt": prompt, "status": f"FAILED: {str(e)[:80]}"})
    keys = list(dict.fromkeys(k for r in rows for k in r))
    out = config.OUTPUTS / f"benchmark_{'offline' if a.offline else 'auto'}"
    with open(f"{out}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    md = [f"Hardware: {platform.processor() or platform.machine()} · {platform.system()} {platform.release()} "
          f"· Python {platform.python_version()}", "", "| " + " | ".join(keys) + " |",
          "|" + "---|" * len(keys)]
    md += ["| " + " | ".join(str(r.get(k, "")) for k in keys) + " |" for r in rows]
    Path(f"{out}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
