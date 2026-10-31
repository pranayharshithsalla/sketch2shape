"""Decal / text / logo layer, baked INTO the model's texture (AI image models cannot spell,
so lettering is drawn here, crisp, after the 3D step).

How it works, per mesh the decal touches:
  1. Give the mesh a unique texture atlas: tiled procedural textures are expanded so every
     surface point owns its texels, and back-facing faces get their own half of the atlas
     (otherwise box-mapped fronts and backs share texels and the decal would show twice).
     Flat-colour and vertex-colour meshes are baked into a new atlas first.
  2. Project the decal from the chosen side like a stencil: for every front-facing triangle
     inside the decal rectangle (and not hidden, checked with a ray-cast depth map), rasterise
     the triangle in UV space, map each texel back to its 3D position, and alpha-blend the
     decal pixel there.
The result is one ordinary textured GLB (no extra decal meshes), so the lettering survives
export to any viewer.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

from .sketch3d import linear_to_srgb

MAX_ATLAS = 4096
FONTS = {  # (Windows, Linux/Docker)
    "Bold sans": ("arialbd.ttf", "DejaVuSans-Bold.ttf"),
    "Heavy": ("impact.ttf", "DejaVuSans-Bold.ttf"),
    "Serif": ("georgiab.ttf", "DejaVuSerif-Bold.ttf"),
    "Mono": ("consolab.ttf", "DejaVuSansMono-Bold.ttf"),
}
SIDES = {"front": 0, "right": 90, "back": 180, "left": 270}


def _font(name: str, px: int) -> ImageFont.FreeTypeFont:
    for f in FONTS.get(name, FONTS["Bold sans"]):
        try:
            return ImageFont.truetype(f, px)
        except OSError:
            continue
    return ImageFont.load_default(px)


def text_image(text: str, color: str = "#FFFFFF", font: str = "Bold sans", px: int = 220,
               outline: str | None = None) -> Image.Image:
    """Render text large (so it stays sharp after projection) on a transparent canvas."""
    f = _font(font, px)
    w = max(int(f.getlength(text)), 1)
    pad = px // 6
    img = Image.new("RGBA", (w + 2 * pad, int(px * 1.3) + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((pad, pad), text, font=f, fill=color,
           stroke_width=max(px // 28, 2) if outline else 0, stroke_fill=outline)
    return img.crop(img.getbbox() or (0, 0, 1, 1))


def _view_rotation(side: str) -> np.ndarray:
    """World -> view frame where the viewer looks down -z from +z (glTF front = +z)."""
    a = math.radians(SIDES.get(side, 0))
    return np.array([[math.cos(a), 0, -math.sin(a)], [0, 1, 0], [math.sin(a), 0, math.cos(a)]])


# ---------------------------------------------------------------- unique atlas per mesh
def _base_image_and_corner_uv(g: trimesh.Trimesh):
    """-> (image RGB uint8 or None, per-corner UV (F,3,2) or None, flat sRGB colour, material)."""
    v = g.visual
    if isinstance(v, trimesh.visual.TextureVisuals):
        mat = v.material
        img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
        if img is not None and v.uv is not None:
            return np.asarray(img.convert("RGB")), v.uv[g.faces], None, mat
        base = getattr(mat, "baseColorFactor", None)
        rgb = linear_to_srgb(np.asarray(base)[:3]) if base is not None else [190, 190, 190]
        return None, None, np.array(rgb, np.uint8), mat
    if isinstance(v, trimesh.visual.ColorVisuals) and v.kind in ("vertex", "face"):
        return None, None, None, None  # vertex colours: baked per face below
    return None, None, np.array([190, 190, 190], np.uint8), None


def _with_atlas(g: trimesh.Trimesh, R: np.ndarray, min_res: int = 2048):
    """Return (unmerged mesh copy, atlas RGB, per-vertex UV) where no two faces share texels."""
    img, cuv, flat, mat = _base_image_and_corner_uv(g)
    F = len(g.faces)
    corners = g.vertices[g.faces]                              # (F,3,3)
    nz = (g.face_normals @ R.T)[:, 2]
    back = nz < 0
    if img is not None:
        lo = np.floor(cuv.reshape(-1, 2).min(0)).astype(int)
        span = np.maximum(np.ceil(cuv.reshape(-1, 2).max(0)).astype(int) - lo, 1)
        r = img.shape[1]
        scale = min(1.0, MAX_ATLAS / (2 * span[0] * r), MAX_ATLAS / (span[1] * r))
        tile = img if scale >= 1 else cv2.resize(img, (max(int(r * scale), 8),) * 2, interpolation=cv2.INTER_AREA)
        half = np.tile(tile, (span[1], span[0], 1))
        atlas = np.concatenate([half, half], 1)
        uv = (cuv - lo) / span                                 # 0..1 within a half
        uv[..., 0] = uv[..., 0] * 0.5 + back[:, None] * 0.5
    else:
        # planar projection from the decal side; fronts left half, backs right half
        p = corners @ R.T
        lo, hi = p.reshape(-1, 3).min(0), p.reshape(-1, 3).max(0)
        ext = np.maximum(hi - lo, 1e-9)
        uv = np.empty((F, 3, 2))
        uv[..., 0] = (p[..., 0] - lo[0]) / ext[0] * 0.49 + 0.005 + back[:, None] * 0.5
        uv[..., 1] = (p[..., 1] - lo[1]) / ext[1] * 0.98 + 0.01
        H = min_res // 2 if ext[1] < ext[0] * 0.6 else min_res
        atlas = np.empty((H, min_res, 3), np.uint8)
        if flat is not None:
            atlas[:] = flat
        else:  # vertex colours -> one colour per face, painted into its UV triangle
            fc = g.visual.face_colors[:, :3]
            atlas[:] = fc.mean(0).astype(np.uint8)
            px = np.stack([uv[..., 0] * min_res, (1 - uv[..., 1]) * H], -1).astype(np.int32)
            for f in range(F):
                cv2.fillConvexPoly(atlas, px[f], tuple(int(c) for c in fc[f]))
    m = trimesh.Trimesh(corners.reshape(-1, 3), np.arange(F * 3).reshape(-1, 3), process=False)
    m.metadata.update(g.metadata)
    return m, atlas, uv.reshape(-1, 2), mat, img is not None


def _material(old, atlas: np.ndarray, keep_factor: bool):
    kw = {}
    if old is not None:
        for k in ("metallicFactor", "roughnessFactor", "alphaMode", "doubleSided"):
            if getattr(old, k, None) is not None:
                kw[k] = getattr(old, k)
        if keep_factor and getattr(old, "baseColorFactor", None) is not None:
            kw["baseColorFactor"] = old.baseColorFactor
    kw.setdefault("metallicFactor", 0.0)
    kw.setdefault("roughnessFactor", 0.6)
    return trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(atlas), **kw)


# ---------------------------------------------------------------- projection + rasterising
def _depth_map(meshes, R, rect, n=96):
    """Nearest-surface view-z over the decal rectangle, from a grid of rays."""
    (x0, x1), (y0, y1) = rect
    allm = trimesh.util.concatenate([trimesh.Trimesh(m.vertices @ R.T, m.faces, process=False) for m in meshes])
    zmax = allm.bounds[1, 2] + 1.0
    xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
    gx, gy = np.meshgrid(xs, ys)
    origins = np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, zmax)]
    dirs = np.tile([0, 0, -1.0], (gx.size, 1))
    depth = np.full(gx.size, -np.inf)
    loc, ray_idx, _ = allm.ray.intersects_location(origins, dirs, multiple_hits=False)
    if len(ray_idx):
        np.maximum.at(depth, ray_idx, loc[:, 2])
    return depth.reshape(n, n), xs, ys


def bake(glb: str, out_glb: str | Path, decal: Image.Image, x: float = 0.5, y: float = 0.55,
         size: float = 0.4, rotation: float = 0.0, side: str = "front") -> dict:
    """Bake `decal` (RGBA) onto the model. x/y: decal centre as a fraction of the model's
    body width/height seen from `side`; size: decal length as a fraction of the body extent along
    the decal direction;
    rotation: degrees counter-clockwise. Returns {"glb", "meshes", "texels"}."""
    scene = trimesh.load(glb, force="scene")
    meshes = [g for g in scene.dump() if isinstance(g, trimesh.Trimesh) and len(g.faces)]
    R = _view_rotation(side)
    # x / y / size are measured on the main body (the largest part), so "centre" means the
    # middle of the mug, not of mug + handle
    body = max(meshes, key=lambda m: m.area)
    allv = body.vertices @ R.T
    lo, hi = allv.min(0), allv.max(0)
    lo[2] = min(lo[2], min((m.vertices @ R.T)[:, 2].min() for m in meshes))
    hi[2] = max(hi[2], max((m.vertices @ R.T)[:, 2].max() for m in meshes))
    th = math.radians(rotation)
    c, s = math.cos(th), math.sin(th)
    # size is measured along the decal's own direction: rotated 90° on a bat, it spans the blade
    W = size * (abs(c) * (hi[0] - lo[0]) + abs(s) * (hi[1] - lo[1]))
    dec = np.asarray(decal.convert("RGBA")).astype(np.float32) / 255
    dh, dw = dec.shape[:2]
    H = W * dh / dw
    cx, cy = lo[0] + x * (hi[0] - lo[0]), lo[1] + y * (hi[1] - lo[1])
    rad = 0.5 * math.hypot(W, H)
    rect = ((cx - rad, cx + rad), (cy - rad, cy + rad))
    depth, gxs, gys = _depth_map(meshes, R, rect)
    tol = 0.02 * (hi[2] - lo[2]) + 1e-4

    from .export import part_names
    names = part_names(meshes)
    out, touched, texels = trimesh.Scene(), 0, 0
    for i, g in enumerate(meshes):
        p = g.vertices[g.faces] @ R.T
        fn = (g.face_normals @ R.T)[:, 2]
        cand = (fn > 0.12) & (p[..., 0].max(1) >= rect[0][0]) & (p[..., 0].min(1) <= rect[0][1]) \
            & (p[..., 1].max(1) >= rect[1][0]) & (p[..., 1].min(1) <= rect[1][1])
        if not cand.any():
            out.add_geometry(g, geom_name=names[i])
            continue
        m, atlas, uv, mat, textured = _with_atlas(g, R)
        AH, AW = atlas.shape[:2]
        acc = atlas.astype(np.float32) / 255
        wrote = 0
        P = m.vertices.reshape(-1, 3, 3) @ R.T
        UVP = np.stack([uv[:, 0] * AW, (1 - uv[:, 1]) * AH], -1).reshape(-1, 3, 2)
        for f in np.nonzero(cand)[0]:
            t = UVP[f]
            bx0, by0 = np.floor(t.min(0)).astype(int)
            bx1, by1 = np.ceil(t.max(0)).astype(int)
            bx0, by0, bx1, by1 = max(bx0, 0), max(by0, 0), min(bx1, AW - 1), min(by1, AH - 1)
            if bx1 < bx0 or by1 < by0:
                continue
            gx, gy = np.meshgrid(np.arange(bx0, bx1 + 1) + 0.5, np.arange(by0, by1 + 1) + 0.5)
            T = np.array([[t[0, 0] - t[2, 0], t[1, 0] - t[2, 0]], [t[0, 1] - t[2, 1], t[1, 1] - t[2, 1]]])
            det = np.linalg.det(T)
            if abs(det) < 1e-12:
                continue
            Ti = np.linalg.inv(T)
            d = np.stack([gx - t[2, 0], gy - t[2, 1]], -1)
            l12 = d @ Ti.T
            l3 = 1 - l12.sum(-1)
            inside = (l12[..., 0] >= -0.02) & (l12[..., 1] >= -0.02) & (l3 >= -0.02)
            if not inside.any():
                continue
            bary = np.dstack([l12, l3])[inside]               # (k,3)
            pos = bary @ P[f]                                   # view-space points (k,3)
            dx, dy = pos[:, 0] - cx, pos[:, 1] - cy
            lx, ly = c * dx + s * dy, -s * dx + c * dy          # into the decal's rotated frame
            du, dv = lx / W + 0.5, 0.5 - ly / H
            ok = (du >= 0) & (du < 1) & (dv >= 0) & (dv < 1)
            if not ok.any():
                continue
            # occlusion: keep texels at (or in front of) the nearest surface along the view ray
            ix = np.clip(np.searchsorted(gxs, pos[:, 0]), 0, len(gxs) - 1)
            iy = np.clip(np.searchsorted(gys, pos[:, 1]), 0, len(gys) - 1)
            ok &= pos[:, 2] >= depth[iy, ix] - tol
            if not ok.any():
                continue
            src = dec[(dv[ok] * dh).astype(int), (du[ok] * dw).astype(int)]
            yy, xx = (gy[inside][ok]).astype(int), (gx[inside][ok]).astype(int)
            a = src[:, 3:4]
            acc[yy, xx] = acc[yy, xx] * (1 - a) + src[:, :3] * a
            wrote += int(ok.sum())
        if wrote:
            touched += 1
            texels += wrote
            m.visual = trimesh.visual.TextureVisuals(
                uv=uv, material=_material(mat, (acc * 255).round().astype(np.uint8), textured))
            out.add_geometry(m, geom_name=names[i])
        else:
            out.add_geometry(g, geom_name=names[i])
    if not touched:
        raise ValueError("The decal missed the model: move it onto the object or pick another side.")
    out_glb = Path(out_glb)
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    out.export(out_glb)
    return {"glb": str(out_glb), "meshes": touched, "texels": texels}
