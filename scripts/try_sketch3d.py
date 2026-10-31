"""Builds each example with the local sketch-modelling modes and renders a preview grid."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PIL import Image
from src.preprocess import preprocess
from src.sketch3d import build, symmetry_score
from src.export import finalize
from src.render import turntable
cases = [("mug", "a red ceramic mug", "auto"), ("car", "a blue car", "auto"), ("teddy", "a brown teddy bear", "auto"),
         ("chair", "a wooden chair", "auto"), ("star", "a gold star medal", "auto")]
rows = []
for name, prompt, mode in cases:
    src = f"examples/{name}.png"
    if not Path(src).exists():
        continue
    out = Path(f"outputs/s3d_{name}"); out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    sk = preprocess(src)
    glb, used, logs = build(sk.mask, out, mode, prompt)
    fin = finalize(glb, out)
    print(name, used, f"{time.time()-t:.1f}s", f"sym={symmetry_score(sk.mask):.2f}", fin["report"]["faces"], fin["report"]["watertight"], logs)
    rows.append(Image.open(turntable(fin["glb"], out / "turntable.png", views=4, size=240)))
grid = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)))
y = 0
for r in rows: grid.paste(r, (0, y)); y += r.height
grid.save(sys.argv[1] if len(sys.argv) > 1 else "outputs/s3d_grid.png")
