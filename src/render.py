"""Dependency-free preview renderer (no OpenGL): painter's-algorithm rasterisation with
PIL, Lambert shading and vertex/texture colours. Used for turntable proofs and README images."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw


def _load(glb: str) -> trimesh.Trimesh:
    scene = trimesh.load(glb, force="scene")
    parts = []
    for g in scene.dump():
        if not isinstance(g, trimesh.Trimesh) or not len(g.faces):
            continue
        if isinstance(g.visual, trimesh.visual.TextureVisuals):
            mat = g.visual.material
            base = getattr(mat, "baseColorFactor", None)
            has_tex = getattr(mat, "baseColorTexture", None) is not None and g.visual.uv is not None
            if has_tex:
                g.visual = g.visual.to_color()
            else:  # flat PBR colour (sketch3d meshes): stored linear, shown as sRGB
                from .sketch3d import linear_to_srgb
                c = np.array([*linear_to_srgb(base[:3]), 255] if base is not None else [180, 180, 180, 255],
                             np.uint8)
                g.visual = trimesh.visual.ColorVisuals(g, vertex_colors=np.tile(c, (len(g.vertices), 1)))
        parts.append(g)
    return trimesh.util.concatenate(parts)


def render_view(mesh: trimesh.Trimesh, yaw_deg: float, size: int = 384, pitch_deg: float = 15) -> Image.Image:
    v = mesh.vertices - mesh.bounds.mean(0)
    v = v / max(np.abs(v).max(), 1e-9)
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    v = v @ ry.T @ rx.T
    f = mesh.faces
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    visible = n[:, 2] > 0
    light = np.array([0.35, 0.5, 0.8]) / np.linalg.norm([0.35, 0.5, 0.8])
    shade = np.clip(n @ light, 0, 1) * 0.7 + 0.3
    if mesh.visual.kind == "face":
        base = mesh.visual.face_colors[:, :3].astype(np.float32)
    elif mesh.visual.kind:
        base = mesh.visual.vertex_colors[:, :3].astype(np.float32)[f].mean(1)
    else:
        base = np.full((len(f), 3), 180.0)
    col = (base * shade[:, None]).clip(0, 255).astype(np.uint8)
    order = np.argsort(tri[:, :, 2].mean(1))
    img = Image.new("RGB", (size, size), (245, 245, 248))
    dr = ImageDraw.Draw(img)
    s = size * 0.45
    xy = np.c_[size / 2 + v[:, 0] * s, size / 2 - v[:, 1] * s]
    for i in order:
        if visible[i]:
            dr.polygon([tuple(xy[j]) for j in f[i]], fill=tuple(int(c) for c in col[i]))
    return img


def turntable(glb: str, out_png: str | Path, views: int = 4, size: int = 320) -> str:
    mesh = _load(glb)
    ims = [render_view(mesh, 360 * k / views + 20, size) for k in range(views)]
    strip = Image.new("RGB", (size * views, size))
    for k, im in enumerate(ims):
        strip.paste(im, (k * size, 0))
    strip.save(out_png)
    return str(out_png)


def turntable_gif(glb: str, out_gif: str | Path, frames: int = 12, size: int = 240) -> str:
    """Animated 360° spin: a quick visual proof that the output is real 3D (~3-5 s on CPU,
    so the UI renders it after showing the model)."""
    mesh = _load(glb)
    ims = [render_view(mesh, 360 * k / frames, size) for k in range(frames)]
    ims[0].save(out_gif, save_all=True, append_images=ims[1:], duration=90, loop=0, optimize=True)
    return str(out_gif)
