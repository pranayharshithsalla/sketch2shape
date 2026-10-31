"""Regenerate the CACHED samples in samples/ (the live app works on fresh input too).

For each sketch: 4 ranked candidates, the final textured model (default engine), the best
candidate reconstructed with image-to-3D, a 360° turntable and a result.json with timings and
fidelity. Writes samples/benchmark.json + samples/benchmark.md (used in the README).

    python scripts/make_samples.py               # all five
    python scripts/make_samples.py bat lamp      # just these
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLES = {
    "bat": ("a wooden cricket bat", "BRAND NAME"),
    "chair": ("a wooden chair", None),
    "mug": ("a red ceramic mug", None),
    "car": ("a red toy car", None),
    "lamp": ("a table lamp with a cream shade and a brass stem", None),
}


def main(names: list[str]) -> int:
    import numpy as np
    from PIL import Image

    from src import config, decal
    from src.designer import sketch_accuracy
    from src.export import finalize
    from src.pipeline import candidate_to_3d, make_candidates, run
    from src.render import turntable_gif

    out_root = ROOT / "samples"
    rows = []
    for name in names or list(SAMPLES):
        prompt, text = SAMPLES[name]
        d = out_root / name
        shutil.rmtree(d, ignore_errors=True)
        (d / "candidates").mkdir(parents=True)
        src = config.EXAMPLES / f"{name}.png"
        shutil.copy(src, d / "input.png")
        sketch = np.asarray(Image.open(src).convert("RGBA"))
        ref = (d / "input.png").read_bytes()
        print(f"== {name}: {prompt}")

        c = make_candidates(sketch, prompt, 4, out_dir=config.OUTPUTS / f"sample_{name}" / "cands")
        for i, cand in enumerate(c["cands"]):
            shutil.copy(cand["path"], d / "candidates" / f"cand_{i + 1}{'_best' if i == c['best'] else ''}.png")

        t = time.time()
        r = run(sketch, prompt, run_name=f"sample_{name}/final")
        glb = r.glb
        if text:  # demo: lettering AI image models cannot render, baked into the texture
            b = decal.bake(glb, d / "decal_tmp.glb", decal.text_image(text, "#111111", "Heavy"),
                           x=0.5, y=0.45, size=0.7, rotation=90)
            glb = finalize(b["glb"], config.OUTPUTS / f"sample_{name}" / "decal", name="model")["glb"]
            (d / "decal_tmp.glb").unlink()
        final_s = round(time.time() - t, 2)
        shutil.copy(glb, d / "model.glb")
        zip_src = Path(glb).with_name(Path(glb).stem + "_obj.zip")
        if zip_src.exists():
            shutil.copy(zip_src, d / "model_obj.zip")
        turntable_gif(str(d / "model.glb"), d / "turntable.gif")

        t = time.time()
        try:
            img3d = candidate_to_3d(c["cands"][c["best"]]["rgba_path"],
                                    config.OUTPUTS / f"sample_{name}" / "img3d", name="model")
            shutil.copy(img3d["glb"], d / "model_from_best_candidate.glb")
            img3d_info = {"engine": img3d["provider"], "seconds": round(time.time() - t, 2),
                          "fidelity": sketch_accuracy(img3d["glb"], ref), "validation": img3d["report"]}
        except Exception as e:  # keep going: the image-to-3D path is the optional second result
            img3d_info = {"error": str(e)[:200]}

        fid = sketch_accuracy(str(d / "model.glb"), ref)
        res = {"sample": name, "cached": True, "prompt": prompt, "decal": text,
               "engine": r.mesh_provider, "final_seconds": final_s, "timings": r.timings,
               "fidelity": fid, "validation": r.report,
               "candidates": {"provider": c["provider"], "auto_pick": c["best"] + 1,
                              "scores": [{k: s[k] for k in ("score", "sketch", "ready", "prompt")} for s in c["cands"]],
                              "seconds": c["timings"]["total"]},
               "image_to_3d_from_best_candidate": img3d_info}
        (d / "result.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        rows.append(res)
        print(f"   engine {r.mesh_provider} · {final_s}s · fidelity {fid} · auto-pick #{c['best'] + 1}")

    bench = out_root / "benchmark.json"
    old = json.loads(bench.read_text(encoding="utf-8")) if bench.exists() else []
    keep = {r["sample"]: r for r in old}
    keep.update({r["sample"]: r for r in rows})
    allrows = [keep[k] for k in SAMPLES if k in keep]
    bench.write_text(json.dumps(allrows, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["| Sample | Engine | Final model (s) | Fidelity | Watertight | Candidates (s) | Auto-pick | "
          "Image→3D engine | Image→3D (s) | Image→3D fidelity |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in allrows:
        i3 = r["image_to_3d_from_best_candidate"]
        f = lambda x: "—" if x is None else f"{x:.0%}"
        md.append(f"| {r['sample']} | {r['engine']} | {r['final_seconds']} | {f(r['fidelity'])} | "
                  f"{'yes' if r['validation']['watertight'] else 'no'} | {r['candidates']['seconds']} | "
                  f"#{r['candidates']['auto_pick']} | {i3.get('engine', 'failed')} | {i3.get('seconds', '—')} | "
                  f"{f(i3.get('fidelity'))} |")
    (out_root / "benchmark.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
