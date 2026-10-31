"""Hard-case benchmark: every reconstruction backend on samples/hard_cases/, raw model vs
refined (carve + add + procedural strings), with silhouette IoU, thin-feature recall, hole match,
Accuracy and runtime. Writes samples/hard_cases/benchmark.{json,md} (the README table).

    python scripts/benchmark.py                          # backends available on this machine
    python scripts/benchmark.py --backends sketch_cad triposr_local --cases racket genie_lamp
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
HC = ROOT / "samples" / "hard_cases"


def main() -> int:
    from src import ai_assembly, config, image2mesh
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="*")
    ap.add_argument("--cases", nargs="*")
    a = ap.parse_args()
    config.IMAGE_PROVIDERS[:] = ["local"]       # candidates from the offline renderer: no quota spent
    config.ACCURACY_RETRY = False               # measure each backend on its own
    from src.pipeline import _run_once
    meta = json.loads((HC / "cases.json").read_text(encoding="utf-8"))
    backends = a.backends or (["sketch_cad", "triposr_local"]
                              + (["ai_cad"] if ai_assembly.available() else [])
                              + (["hunyuan3d"] if "hunyuan3d" in image2mesh.available() else []))
    rows = []
    for name in a.cases or list(meta):
        m = meta[name]
        for be in backends:
            t = time.time()
            try:
                r = _run_once(str(HC / f"{name}.png"), m["prompt"], mesh_provider=be,
                              run_name=f"bench/{name}_{be}")
                acc = r.accuracy or {}
                b = acc.get("before", {})
                row = {"case": name, "backend": be, "engine_used": r.mesh_provider,
                       "seconds": round(time.time() - t, 1), "expected_holes": m["expected_holes"],
                       "raw": {k: b.get(k) for k in ("accuracy", "iou", "thin_recall", "holes_open", "holes_extra")},
                       "refined": {k: acc.get(k) for k in ("accuracy", "iou", "thin_recall", "holes_open", "holes_extra")},
                       "warning": acc.get("warning", "")}
            except Exception as e:
                row = {"case": name, "backend": be, "error": str(e)[:200], "seconds": round(time.time() - t, 1)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    (HC / "benchmark.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    f = lambda v: "—" if v is None else f"{v:.2f}"
    md = ["| Case | Backend (used) | Raw acc | Refined acc | Raw IoU → refined | Thin recall | Holes open (expected) | Time (s) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            md.append(f"| {r['case']} | {r['backend']} | failed: {r['error'][:60]} | | | | | {r['seconds']} |")
            continue
        raw, ref = r["raw"], r["refined"]
        md.append(f"| {r['case']} | {r['backend']} ({r['engine_used']}) | {f(raw['accuracy'])} | **{f(ref['accuracy'])}** | "
                  f"{f(raw['iou'])} → {f(ref['iou'])} | {f(ref['thin_recall'])} | "
                  f"{raw['holes_open']} → {ref['holes_open']} ({r['expected_holes']}) | {r['seconds']} |")
    (HC / "benchmark.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
