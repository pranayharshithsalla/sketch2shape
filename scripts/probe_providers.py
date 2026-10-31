"""Smoke-test each image provider once on examples/mug.png (prints status + seconds)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PIL import Image
from src.preprocess import preprocess
from src import sketch2image
sk = preprocess("examples/mug.png")
out = Path("outputs/probe"); out.mkdir(parents=True, exist_ok=True)
for name in sys.argv[1:] or ["gemini", "replicate", "hf"]:
    t = time.time()
    try:
        img = sketch2image.PROVIDERS[name][0](sk, sketch2image.build_prompt("a red ceramic mug", ""), 42)
        Image.fromarray(img).save(out / f"{name}.png")
        print(f"{name}: OK {img.shape} {time.time()-t:.1f}s")
    except Exception as e:
        print(f"{name}: FAIL {time.time()-t:.1f}s {type(e).__name__}: {str(e)[:300]}")
