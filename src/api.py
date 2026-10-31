"""REST API + AR viewer, served by the same FastAPI app that hosts the Gradio UI.

  GET  /api/health                     providers, device, version
  POST /api/generate                   sketch/photo file and/or prompt -> textured .glb/.obj/.stl + report
  POST /api/candidates                 sketch/photo (+prompt) -> 4 ranked candidate images (no 3D yet)
  POST /api/candidates/{id}/build      build 3D from candidate `index` of a candidates run
  POST /api/decal                      bake text onto a generated model's texture
  GET  /models/{path}                  any generated file (under OUTPUT_DIR only)
  GET  /ar?m={path}                    <model-viewer> page with AR (phone: scan the QR in the UI)
"""
from __future__ import annotations

import html
import io
import json
import socket
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image

from . import config

MODEL_VIEWER = "https://cdn.jsdelivr.net/npm/@google/model-viewer@3.5.0/dist/model-viewer.min.js"


def lan_ip() -> str:
    """This machine's address on the local network (what a phone on the same Wi-Fi can reach)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))  # no packet is sent; just picks the outgoing interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def rel(path: str | Path) -> str:
    """Path of a generated file relative to OUTPUT_DIR, as used in /models/ and /ar URLs."""
    return Path(path).resolve().relative_to(config.OUTPUTS.resolve()).as_posix()


def _safe(path: str) -> Path:
    p = (config.OUTPUTS / path).resolve()
    if config.OUTPUTS.resolve() not in p.parents or not p.is_file():
        raise HTTPException(404, "not found")
    return p


def _links(paths: dict) -> dict:
    return {k: f"/models/{rel(v)}" for k, v in paths.items() if v}


async def _image(f: UploadFile | None):
    if f is None:
        return None
    data = await f.read()
    if not data:
        return None
    try:
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"))
    except Exception:
        raise HTTPException(400, "could not read the uploaded image")


def ar_page(model_rel: str, title: str = "Sketch2Shape model") -> str:
    src = f"/models/{html.escape(model_rel)}"
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<script type="module" src="{MODEL_VIEWER}"></script>
<style>html,body{{margin:0;height:100%;background:#070B14;color:#F0F6FC;font-family:system-ui,sans-serif}}
model-viewer{{width:100%;height:calc(100% - 64px);--poster-color:transparent}}
.bar{{height:64px;display:flex;align-items:center;justify-content:space-between;padding:0 16px;
border-bottom:1px solid #1C2B4E}} .bar b{{color:#00E5FF}} a{{color:#00E5FF}}
#ar{{background:#00E5FF;color:#05101E;border:0;border-radius:12px;padding:12px 18px;font-weight:800;
position:absolute;bottom:20px;left:50%;transform:translateX(-50%);box-shadow:0 0 20px rgba(0,229,255,.35)}}</style>
</head><body><div class="bar"><div>Sketch<b>2</b>Shape</div><a href="{src}" download>Download .glb</a></div>
<model-viewer src="{src}" ar ar-modes="scene-viewer webxr quick-look" camera-controls auto-rotate
 shadow-intensity="1" exposure="1.05" environment-image="neutral" alt="{html.escape(title)}">
<button slot="ar-button" id="ar">View in your room (AR)</button></model-viewer></body></html>"""


def qr_png(url: str) -> Image.Image:
    import qrcode
    q = qrcode.QRCode(border=2, box_size=8)
    q.add_data(url)
    q.make(fit=True)
    return q.make_image(fill_color="#05101E", back_color="#00E5FF").convert("RGB")


def register(app: FastAPI) -> None:
    @app.get("/api/health")
    def health():
        from . import image2mesh, sketch2image
        return {"ok": True, "device": "cuda" if config.use_gpu() else "cpu",
                "image_providers": sketch2image.available(), "mesh_providers": image2mesh.available()}

    @app.post("/api/generate")
    async def generate(sketch: UploadFile | None = File(None), prompt: str = Form(""),
                       material: str = Form("auto"), engine: str = Form("auto"), seed: int = Form(42)):
        """Full pipeline in one call. At least one of `sketch` or `prompt` is required."""
        from .pipeline import StageError, run
        img = await _image(sketch)
        if img is None and not prompt.strip():
            raise HTTPException(400, "send a sketch/photo, a prompt, or both")
        try:
            r = run(img, prompt, material, mesh_provider=engine, seed=seed)
        except StageError as e:
            raise HTTPException(422, f"stage '{e.stage}' failed: {str(e)[:300]}")
        return {"files": _links({"glb": r.glb, "obj_zip": r.obj_zip, "stl": r.stl}),
                "ar": f"/ar?m={rel(r.glb)}", "engine": r.mesh_provider, "image_provider": r.image_provider,
                "prompt": r.prompt, "caption": r.caption, "fidelity": r.scores[r.best] if r.scores else None,
                "validation": r.report, "timings": r.timings}

    @app.post("/api/candidates")
    async def candidates(sketch: UploadFile | None = File(None), prompt: str = Form(""),
                         n: int = Form(4), seed: int = Form(42)):
        from .pipeline import make_candidates
        img = await _image(sketch)
        if img is None and not prompt.strip():
            raise HTTPException(400, "send a sketch/photo, a prompt, or both")
        try:
            r = make_candidates(img, prompt, max(1, min(n, 8)), seed)
        except Exception as e:
            raise HTTPException(422, str(e)[:300])
        return {"id": r["dir"].name, "best": r["best"], "provider": r["provider"], "prompt": r["prompt"],
                "timings": r["timings"],
                "candidates": [{"image": f"/models/{rel(c['path'])}", "total": c["score"], "sketch": c["sketch"],
                                "ready": c["ready"], "prompt": c["prompt"]} for c in r["cands"]]}

    @app.post("/api/candidates/{run_id}/build")
    def build(run_id: str, index: int = Form(-1), engine: str = Form("auto")):
        from .pipeline import candidate_to_3d
        d = (config.OUTPUTS / run_id).resolve()
        if config.OUTPUTS.resolve() not in d.parents or not (d / "candidates.json").is_file():
            raise HTTPException(404, "unknown candidates run")
        meta = json.loads((d / "candidates.json").read_text(encoding="utf-8"))
        i = meta["best"] if index < 0 else index
        if not (d / f"cand_{i}_rgba.png").is_file():
            raise HTTPException(404, "no such candidate")
        out = candidate_to_3d(str(d / f"cand_{i}_rgba.png"), d / f"build_{i}", engine, name="model")
        return {"files": _links({"glb": out["glb"], "obj_zip": out["obj_zip"], "stl": out["stl"]}),
                "ar": f"/ar?m={rel(out['glb'])}", "engine": out["provider"], "validation": out["report"],
                "timings": out["timings"]}

    @app.post("/api/decal")
    def decal_api(model: str = Form(..., description="path from a /models/ link"), text: str = Form(...),
                  color: str = Form("#FFFFFF"), font: str = Form("Bold sans"), x: float = Form(0.5),
                  y: float = Form(0.5), size: float = Form(0.4), rotation: float = Form(0.0),
                  side: str = Form("front")):
        from . import decal
        from .export import finalize
        src = _safe(model.removeprefix("/models/"))
        out_dir = src.parent / "decal"
        baked = decal.bake(str(src), out_dir / "baked.glb", decal.text_image(text, color, font),
                           x, y, size, rotation, side)
        out = finalize(baked["glb"], out_dir, name="model_decal")
        return {"files": _links({"glb": out["glb"], "obj_zip": out["obj_zip"], "stl": out["stl"]}),
                "validation": out["report"]}

    @app.get("/models/{path:path}")
    def models(path: str):
        p = _safe(path)
        media = "model/gltf-binary" if p.suffix == ".glb" else None
        return FileResponse(p, media_type=media)

    @app.get("/ar", response_class=HTMLResponse)
    def ar(m: str):
        _safe(m)
        return ar_page(m)
