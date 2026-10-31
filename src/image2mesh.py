"""Stage 6: image-to-3D behind a resilient provider router. Always returns a .glb path.

Providers (config.MESH_PROVIDERS order):
  trellis           - microsoft TRELLIS via the public HF Space (textured GLB). Needs HF_TOKEN for ZeroGPU quota.
  sf3d              - Stability AI Stable Fast 3D via HF Space (textured GLB, ~10 s). Needs HF_TOKEN.
  replicate_trellis - TRELLIS on Replicate (paid, no queue). Needs REPLICATE_API_TOKEN.
  local             - offline silhouette inflation + marching cubes, vertex-coloured. CPU, <3 s, never fails.
"""
from __future__ import annotations

import logging
import os
import shutil
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image

from . import config
from .router import ProviderError, with_hf_tokens, with_retries

log = logging.getLogger(__name__)

TRELLIS_SPACE = os.getenv("TRELLIS_SPACE", "trellis-community/TRELLIS")
SF3D_SPACE = os.getenv("SF3D_SPACE", "stabilityai/stable-fast-3d")
REPLICATE_TRELLIS_MODEL = os.getenv("REPLICATE_TRELLIS_MODEL", "firtoz/trellis")


def _save_png(rgba: np.ndarray, out_dir: Path) -> str:
    p = out_dir / f"mesh_input_{uuid.uuid4().hex[:6]}.png"
    Image.fromarray(rgba).save(p)
    return str(p)


def _pick_glb(result) -> str:
    items = result if isinstance(result, (list, tuple)) else [result]
    for it in items:
        path = it.get("value") or it.get("path") if isinstance(it, dict) else it
        if isinstance(path, str) and path.lower().endswith(".glb") and os.path.exists(path):
            return path
    raise ProviderError(f"no .glb in provider output: {str(result)[:200]}")


def mesh_trellis(rgba: np.ndarray, out_dir: Path, seed: int = 0) -> str:
    from gradio_client import Client, handle_file
    img = _save_png(rgba, out_dir)

    def call(token):
        c = Client(TRELLIS_SPACE, token=token, verbose=False)
        c.predict(api_name="/start_session")
        pre = c.predict(handle_file(img), api_name="/preprocess_image")
        return c.predict(handle_file(pre), [], seed, 7.5, 12, 3.0, 12, "stochastic", 0.95, 1024,
                         api_name="/generate_and_extract_glb")
    return _pick_glb(with_hf_tokens(call))


def mesh_sf3d(rgba: np.ndarray, out_dir: Path, seed: int = 0) -> str:
    from gradio_client import Client, handle_file
    c = Client(SF3D_SPACE, token=config.HF_TOKEN, verbose=False)
    res = c.predict(handle_file(_save_png(rgba, out_dir)), 0.85, "None", -1, 1024, api_name="/run_button")
    return _pick_glb(res)


def mesh_replicate_trellis(rgba: np.ndarray, out_dir: Path, seed: int = 0) -> str:
    import replicate
    import requests
    client = replicate.Client(api_token=config.REPLICATE_API_TOKEN)
    version = client.models.get(REPLICATE_TRELLIS_MODEL).latest_version.id
    with open(_save_png(rgba, out_dir), "rb") as f:
        out = client.run(f"{REPLICATE_TRELLIS_MODEL}:{version}", input={
            "images": [f], "seed": seed, "texture_size": 1024, "mesh_simplify": 0.95,
            "generate_color": False, "generate_model": True, "save_gaussian_ply": False,
            "ss_sampling_steps": 12, "slat_sampling_steps": 12})
    url = str(getattr(out.get("model_file"), "url", out.get("model_file")))
    p = out_dir / f"replicate_{uuid.uuid4().hex[:6]}.glb"
    p.write_bytes(requests.get(url, timeout=120).content)
    return str(p)


def mesh_local(rgba: np.ndarray, out_dir: Path, seed: int = 0, res: int = 160,
               depth: float = 0.75) -> str:
    """Silhouette inflation: height at each pixel follows a circular profile of its
    distance to the outline (h = sqrt(d(2R-d))), mirrored front/back, meshed with
    marching cubes and coloured from the generated image."""
    from skimage.measure import marching_cubes
    # low threshold after area-downsampling keeps thin parts (chair legs, handles) connected
    alpha = cv2.resize(rgba[..., 3], (res, res), interpolation=cv2.INTER_AREA) > 40
    rgb = cv2.resize(rgba[..., :3], (res, res), interpolation=cv2.INTER_AREA)
    if alpha.sum() < 30:
        raise ProviderError("local: empty silhouette")
    a8 = alpha.astype(np.uint8)
    d = cv2.distanceTransform(a8, cv2.DIST_L2, 5)
    R = d.max()
    h = np.sqrt(np.clip(d * (2 * R - d), 0, None)) * depth
    h = cv2.GaussianBlur(h, (0, 0), max(1.0, res * 0.015)) + 1.5   # soften medial-axis creases
    # 2D signed distance to the outline gives clean side walls instead of zero-thickness rims
    sdf = d - cv2.distanceTransform(1 - a8, cv2.DIST_L2, 5)
    sdf = cv2.GaussianBlur(sdf, (0, 0), 0.8)
    zmax = int(np.ceil(h.max())) + 2
    z = np.arange(-zmax, zmax + 1, dtype=np.float32)
    vol = np.minimum(sdf[:, :, None], h[:, :, None] - np.abs(z)[None, None, :])  # >0 inside
    vol = np.pad(vol, 1, constant_values=-1)
    # Colour: replace dark outline strokes near the rim with neighbouring interior colour
    rim = (a8 > 0) & (cv2.erode(a8, np.ones((5, 5), np.uint8)) == 0)
    dark = (rgb.mean(-1) < 70) & (a8 > 0)
    rgb = cv2.inpaint(rgb, ((rim & dark) | (a8 == 0)).astype(np.uint8), 3, cv2.INPAINT_TELEA)
    verts, faces, _, _ = marching_cubes(vol, level=0.0)
    verts -= 1
    yi = np.clip(verts[:, 0].round().astype(int), 0, res - 1)
    xi = np.clip(verts[:, 1].round().astype(int), 0, res - 1)
    colours = np.c_[rgb[yi, xi], np.full(len(verts), 255)].astype(np.uint8)
    xyz = np.c_[verts[:, 1], res - verts[:, 0], verts[:, 2] - zmax]  # x right, y up, z to viewer
    mesh = trimesh.Trimesh(xyz, faces, vertex_colors=colours, process=True)
    trimesh.smoothing.filter_taubin(mesh, iterations=8)
    mesh.fix_normals()
    p = out_dir / f"local_{uuid.uuid4().hex[:6]}.glb"
    mesh.export(p)
    return str(p)


def mesh_triposr_local(rgba: np.ndarray, out_dir: Path, seed: int = 0) -> str:
    from .triposr_local import reconstruct
    return reconstruct(rgba, out_dir)


def _triposr_ok() -> bool:
    from .triposr_local import available as ok
    return ok()


PROVIDERS = {
    "trellis": (mesh_trellis, lambda: bool(config.HF_TOKEN)),
    "triposr_local": (mesh_triposr_local, _triposr_ok),
    "sf3d": (mesh_sf3d, lambda: bool(config.HF_TOKEN)),
    "replicate_trellis": (mesh_replicate_trellis, lambda: bool(config.REPLICATE_API_TOKEN)),
    "local": (mesh_local, lambda: True),
}


def available() -> list[str]:
    return [p for p in config.MESH_PROVIDERS if p in PROVIDERS and PROVIDERS[p][1]()]


def image_to_mesh(rgba: np.ndarray, out_dir: Path, provider: str = "auto",
                  seed: int = 0) -> tuple[str, str, list[str]]:
    """Returns (glb_path_in_out_dir, provider_used, log_lines)."""
    order = available() if provider == "auto" else [provider, "local"]
    logs = []
    for name in dict.fromkeys(order):
        t = time.time()
        try:
            glb = with_retries(PROVIDERS[name][0], rgba, out_dir, seed, name=f"mesh:{name}",
                               retries=0 if name == "local" else 1)
            dst = out_dir / f"raw_{name}.glb"
            if Path(glb).resolve() != dst.resolve():
                shutil.copy(glb, dst)
            logs.append(f"mesh provider '{name}' ok in {time.time() - t:.1f}s")
            return str(dst), name, logs
        except Exception as e:
            logs.append(f"mesh provider '{name}' failed after {time.time() - t:.1f}s: {str(e)[:200]}")
    raise ProviderError("All mesh providers failed:\n" + "\n".join(logs))
