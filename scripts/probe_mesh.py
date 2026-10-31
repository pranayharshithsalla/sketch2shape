"""Smoke-test one mesh provider on a locally rendered mug (prints status + seconds)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preprocess import preprocess
from src.segment import remove_bg
from src import sketch2image, image2mesh
name = sys.argv[1]
sk = preprocess("examples/mug.png")
rgba = remove_bg(sketch2image.gen_local(sk, "a red ceramic mug. Single object", 0))
out = Path(f"outputs/probe/{name}"); out.mkdir(parents=True, exist_ok=True)
t = time.time()
try:
    p = image2mesh.PROVIDERS[name][0](rgba, out, 0)
    print(f"{name}: OK {p} {time.time()-t:.1f}s")
except Exception as e:
    print(f"{name}: FAIL {time.time()-t:.1f}s {type(e).__name__}: {str(e)[:400]}")
