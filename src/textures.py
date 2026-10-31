"""Procedural PBR textures for every material the prompt understands, plus UV mapping.

Textures are generated in numpy (no downloads, ~20 ms each, cached), tinted by the part's
colour, and embedded in the GLB/OBJ:
  wood grain · marble veins · brushed / polished / cast metal · mirror chrome · glazed ceramic
  · fabric weave · leather pebble · granite stone · porous concrete · grainy clay/terracotta
  · jade · ivory · rubber · plastic · transparent glass/crystal
UVs: cylindrical around the part's axis for round parts (with the wrap-around seam split so
the texture does not smear), planar per face for flat-faced parts (boxes, extrusions).
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
import trimesh
from PIL import Image

SIZE = 512


# ------------------------------------------------------------------ noise
def _noise(rng, cells_y: int, cells_x: int) -> np.ndarray:
    """Smooth value noise in [0,1] at SIZE x SIZE from a cells_y x cells_x random grid."""
    g = rng.random((cells_y + 1, cells_x + 1)).astype(np.float32)
    return cv2.resize(g, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC).clip(0, 1)


def _fbm(rng, base: int = 4, octaves: int = 5, stretch=(1, 1)) -> np.ndarray:
    out, amp, tot = np.zeros((SIZE, SIZE), np.float32), 1.0, 0.0
    for o in range(octaves):
        c = base * 2 ** o
        out += amp * _noise(rng, max(1, int(c * stretch[0])), max(1, int(c * stretch[1])))
        tot += amp
        amp *= 0.5
    return out / tot


def _norm(a):
    return (a - a.min()) / max(a.max() - a.min(), 1e-6)


# ------------------------------------------------------------------ patterns (luminance in [0,1])
def _wood(rng):
    """Plank grain: slowly wandering growth rings plus fine fibres running along v."""
    y, x = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32) / SIZE
    warp = _fbm(rng, 2, 3, stretch=(1.5, 0.5))
    rings = 0.5 + 0.5 * np.sin((x * 5.0 + warp * 1.6 + 0.15 * np.sin(y * 6.3)) * np.pi * 2)
    rings = rings ** 2.2  # thin dark late-wood lines, wide light early-wood
    fibres = _fbm(rng, 8, 4, stretch=(0.15, 6))  # streaks along the grain
    return _norm(0.6 * (1 - rings) + 0.4 * fibres)


def _marble(rng):
    y, x = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32) / SIZE
    turb = _fbm(rng, 2, 6)
    veins = np.abs(np.sin((x * 2 + y * 1.3 + turb * 4) * np.pi))
    soft = cv2.GaussianBlur((1 - veins) ** 4, (0, 0), 2.0)  # wider, softer main veins
    hair = (1 - veins) ** 18  # thin crisp hairline veins
    cloud = _fbm(rng, 6, 4)
    return _norm(1 - 0.55 * soft - 0.35 * hair - 0.12 * cloud)


def _brushed(rng, strength=1.0):
    streaks = _fbm(rng, 4, 5, stretch=(64, 0.25))
    return 0.82 + 0.18 * strength * _norm(streaks)


def _cast(rng):
    return 0.7 + 0.3 * _norm(_fbm(rng, 24, 4)) - 0.2 * (_noise(rng, 180, 180) > 0.93)


def _ceramic(rng):
    speck = (_noise(rng, 220, 220) > 0.96).astype(np.float32)
    return 0.95 + 0.05 * _fbm(rng, 6, 3) - 0.25 * cv2.GaussianBlur(speck, (3, 3), 0)


def _fabric(rng):
    y, x = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    k = 2 * np.pi / 8.0
    weave = 0.5 + 0.25 * np.sin(x * k) * np.sign(np.sin(y * k / 2)) + 0.25 * np.sin(y * k) * np.sign(np.sin(x * k / 2))
    return _norm(0.75 * weave + 0.25 * _fbm(rng, 32, 3))


def _leather(rng):
    n = _noise(rng, 90, 90)
    pebbles = _norm(cv2.GaussianBlur(n, (0, 0), 1.5) - cv2.GaussianBlur(n, (0, 0), 5))
    creases = (cv2.GaussianBlur(n, (0, 0), 3) > 0.62).astype(np.float32)
    return 0.6 + 0.4 * pebbles ** 1.5 - 0.12 * cv2.GaussianBlur(creases, (0, 0), 1) - 0.06 * _fbm(rng, 4, 3)


def _granite(rng):
    a = (_noise(rng, 160, 160) > 0.78).astype(np.float32)
    b = (_noise(rng, 160, 160) < 0.15).astype(np.float32)
    return _norm(0.6 * _fbm(rng, 10, 4) + 0.35 * a - 0.4 * b)


def _concrete(rng):
    pores = (_noise(rng, 120, 120) < 0.06).astype(np.float32)
    return 0.8 + 0.2 * _norm(_fbm(rng, 8, 5)) - 0.35 * cv2.GaussianBlur(pores, (3, 3), 0)


def _clay(rng):
    return 0.78 + 0.22 * _norm(_fbm(rng, 20, 5))


def _fine(rng, amount=0.08):
    return 1 - amount + amount * _norm(_fbm(rng, 40, 3))


# material -> (pattern fn, dark factor, light factor, metallic, roughness, alpha)
SPECS = {
    "wood": (_wood, 0.55, 1.15, 0.0, 0.6, 1.0),
    "marble": (_marble, 0.55, 1.05, 0.0, 0.2, 1.0),
    "metal": (lambda r: _brushed(r, 1.0), 0.85, 1.08, 0.95, 0.35, 1.0),
    "steel": (lambda r: _brushed(r, 1.0), 0.85, 1.08, 0.95, 0.32, 1.0),
    "aluminium": (lambda r: _brushed(r, 1.2), 0.85, 1.08, 0.9, 0.38, 1.0),
    "gold": (lambda r: _brushed(r, 0.5), 0.9, 1.06, 1.0, 0.22, 1.0),
    "silver": (lambda r: _brushed(r, 0.5), 0.9, 1.06, 1.0, 0.18, 1.0),
    "copper": (lambda r: _brushed(r, 0.6), 0.88, 1.06, 1.0, 0.28, 1.0),
    "bronze": (lambda r: _brushed(r, 0.7), 0.85, 1.05, 1.0, 0.35, 1.0),
    "brass": (lambda r: _brushed(r, 0.6), 0.88, 1.06, 1.0, 0.28, 1.0),
    "chrome": (lambda r: _fine(r, 0.02), 1.0, 1.0, 1.0, 0.05, 1.0),
    "iron": (_cast, 0.7, 1.05, 0.85, 0.6, 1.0),
    "ceramic": (_ceramic, 0.8, 1.02, 0.0, 0.12, 1.0),
    "porcelain": (_ceramic, 0.85, 1.02, 0.0, 0.08, 1.0),
    "plastic": (lambda r: _fine(r, 0.05), 0.95, 1.02, 0.0, 0.42, 1.0),
    "rubber": (lambda r: _fine(r, 0.15), 0.85, 1.0, 0.0, 0.92, 1.0),
    "glass": (lambda r: _fine(r, 0.02), 1.0, 1.0, 0.0, 0.03, 0.35),
    "crystal": (lambda r: _fine(r, 0.02), 1.0, 1.0, 0.0, 0.02, 0.3),
    "fabric": (_fabric, 0.7, 1.05, 0.0, 0.95, 1.0),
    "leather": (_leather, 0.7, 1.08, 0.0, 0.55, 1.0),
    "stone": (_granite, 0.6, 1.15, 0.0, 0.85, 1.0),
    "concrete": (_concrete, 0.75, 1.05, 0.0, 0.95, 1.0),
    "clay": (_clay, 0.8, 1.08, 0.0, 0.85, 1.0),
    "terracotta": (_clay, 0.8, 1.08, 0.0, 0.85, 1.0),
    "jade": (_marble, 0.75, 1.1, 0.0, 0.18, 0.92),
    "ivory": (lambda r: _fine(r, 0.1), 0.92, 1.02, 0.0, 0.3, 1.0),
    "paint": (lambda r: _fine(r, 0.03), 0.97, 1.0, 0.0, 0.4, 1.0),
}
ALIASES = {"wooden": "wood", "golden": "gold", "metallic": "metal", "aluminum": "aluminium",
           "stainless": "steel", "pottery": "clay", "stoneware": "ceramic", "granite": "stone"}


def canonical(name: str | None) -> str | None:
    if not name:
        return None
    name = ALIASES.get(name.lower(), name.lower())
    return name if name in SPECS else None


@lru_cache(maxsize=64)
def texture(name: str, rgb: tuple[int, int, int], seed: int = 7) -> Image.Image:
    """Tinted colour texture: the pattern modulates the part colour between dark and light."""
    fn, dark, light, *_ = SPECS[name]
    lum = np.clip(fn(np.random.default_rng(seed)), 0, 1)[..., None]
    col = np.asarray(rgb, np.float32)[None, None, :] * (dark + (light - dark) * lum)
    return Image.fromarray(np.clip(col, 0, 255).astype(np.uint8))


# ------------------------------------------------------------------ UV mapping
def _cylindrical_uv(mesh: trimesh.Trimesh, axis: np.ndarray, center: np.ndarray, tile: float):
    """Cylindrical UVs around `axis` (world units / tile). Faces straddling the wrap-around
    seam get duplicated vertices so the texture does not smear across them."""
    a = axis / np.linalg.norm(axis)
    ref = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 0, 1.0])
    e1 = np.cross(a, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    d = mesh.vertices - center
    h = d @ a
    radial = d - np.outer(h, a)
    theta = np.arctan2(radial @ e2, radial @ e1)  # -pi..pi
    r_mean = max(float(np.linalg.norm(radial, axis=1).mean()), 1e-6)
    U = 2 * np.pi * r_mean / tile  # u span for a full turn, in tiles
    u = (theta + np.pi) / (2 * np.pi) * U
    v = h / tile
    verts, normals = mesh.vertices.copy(), mesh.vertex_normals.copy()
    faces = mesh.faces.copy()
    uv = np.c_[u, v]
    fu = u[faces]
    bad = np.where(fu.max(1) - fu.min(1) > U / 2)[0]
    if len(bad):
        extra_v, extra_uv, extra_n = [], [], []
        nv = len(verts)
        for fi in bad:
            for c in range(3):
                vi = faces[fi, c]
                if u[vi] < U / 2:  # move the low side across the seam
                    extra_v.append(verts[vi])
                    extra_n.append(normals[vi])
                    extra_uv.append([u[vi] + U, v[vi]])
                    faces[fi, c] = nv
                    nv += 1
        verts = np.vstack([verts, extra_v])
        normals = np.vstack([normals, extra_n])
        uv = np.vstack([uv, extra_uv])
    out = trimesh.Trimesh(verts, faces, vertex_normals=normals, process=False)
    return out, uv


def _planar_uv(mesh: trimesh.Trimesh, tile: float):
    """Box mapping: each face is projected along its dominant normal axis (flat-faced parts)."""
    m = mesh.copy()
    m.unmerge_vertices()
    fn = m.face_normals
    dom = np.abs(fn).argmax(1)
    uv = np.zeros((len(m.vertices), 2))
    for f, k in enumerate(dom):
        ax = [(1, 2), (0, 2), (0, 1)][k]  # x->(y,z) y->(x,z) z->(x,y)
        idx = m.faces[f]
        p = m.vertices[idx]
        uv[idx] = np.c_[p[:, ax[0]], p[:, ax[1]]] / tile
    return m, uv


def apply(mesh: trimesh.Trimesh, material: str | None, rgb, metal: float, rough: float,
          tile: float, mapping: str = "auto", axis=None) -> trimesh.Trimesh:
    """Return `mesh` with a tinted procedural texture for `material`, or None if the material
    has no texture (the caller then keeps a flat PBR colour)."""
    name = canonical(material)
    if name is None:
        return None
    _, _, _, t_metal, t_rough, alpha = SPECS[name]
    if mapping == "auto":
        mapping = "planar" if len(mesh.faces) <= 64 else "cyl"
    if mapping == "planar":
        m, uv = _planar_uv(mesh, tile)
    else:
        if mapping == "cyl_y":
            axis = np.array([0, 1.0, 0])
        elif mapping == "axis" and axis is not None:
            axis = np.asarray(axis, float)
        else:  # principal axis of the part (tubes, handles, rotated cylinders)
            pts = mesh.vertices - mesh.vertices.mean(0)
            axis = np.linalg.svd(pts, full_matrices=False)[2][0]
        m, uv = _cylindrical_uv(mesh, axis, mesh.bounds.mean(0), tile)
    # the material decides how metallic/rough it is; texture images are sRGB (glTF expects that)
    mat = trimesh.visual.material.PBRMaterial(
        baseColorTexture=texture(name, tuple(int(c) for c in rgb)),
        baseColorFactor=[255, 255, 255, int(255 * alpha)],
        metallicFactor=t_metal, roughnessFactor=min(rough, t_rough) if rough < 0.15 else t_rough,
        alphaMode="BLEND" if alpha < 1 else "OPAQUE", doubleSided=alpha < 1)
    m.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
    return m
