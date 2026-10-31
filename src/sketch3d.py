"""Local, quota-free sketch-based modelling: turns the sketch silhouette directly into
clean geometry with classic CAD operations, so the result follows the drawn lines exactly.

  revolve  - lathe: spin the half-profile around the vertical axis (mugs, vases, bottles,
             lamps, glasses). Containers are hollowed. Parts that break the symmetry
             (a mug handle) are added as inflated tubes.
  extrude  - flat objects (signs, logos, gears, cookie cutters), holes preserved.
  inflate  - organic shapes (toys, animals): distance-field inflation + marching cubes.

Materials: PBR base colour / metallic / roughness per part (body, cap, rim, base, handle)
from the prompt (see prompt.py), so 'a red mug with a black handle' exports two materials.
"""
from __future__ import annotations

import re
import uuid
from pathlib import Path

import cv2
import numpy as np
import trimesh

from .prompt import parse as prompt_parse

LATHE_WORDS = {"mug", "cup", "vase", "bottle", "glass", "goblet", "bowl", "jar", "pot", "lamp",
               "candle", "pawn", "chess", "column", "pillar", "tower", "teapot", "kettle", "jug",
               "flask", "can", "bucket", "barrel", "lampshade", "trophy", "urn", "tumbler", "wine",
               "pitcher", "teacup", "beaker", "carafe", "chalice", "planter", "flowerpot", "basket",
               "bin", "hamper"}
CONTAINER_WORDS = {"mug", "cup", "vase", "bowl", "jar", "pot", "glass", "goblet", "bucket",
                   "tumbler", "jug", "urn", "wine", "flask", "pitcher", "teacup", "beaker", "carafe",
                   "chalice", "planter", "flowerpot", "basket", "bin", "hamper"}
SPOUT_WORDS = {"teapot", "kettle", "jug", "pitcher", "watering", "gravy", "carafe"}
FLAT_WORDS = {"sign", "logo", "letter", "coin", "medal", "badge", "gear", "cookie", "plate",
              "key", "star", "heart", "leaf", "card", "token", "emblem", "tile", "shield",
              # flat objects whose holes / strings / loops matter: extruded from the hole-aware mask
              "racket", "racquet", "scissors", "wheel", "hoop", "frame", "hanger", "spatula",
              "comb", "wrench", "sword", "paddle", "slingshot", "keyring"}

THIN_FLAT_WORDS = {"racket", "racquet", "scissors", "wheel", "hoop", "frame", "hanger", "spatula",
                   "comb", "wrench", "sword", "paddle", "slingshot", "keyring", "key"}


def words_of(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", (text or "").lower()))


def object_words(text: str) -> set[str]:
    """Words naming the object. 'glass' is a material in 'a glass bottle' but the object in
    'a wine glass', so it only counts when no other object word is present."""
    w = words_of(text)
    if "glass" in w and (w & (LATHE_WORDS | FLAT_WORDS)) - {"glass", "wine"}:
        w = w - {"glass"}
    return w


def srgb_to_linear(rgb) -> list[int]:
    """glTF baseColorFactor is linear; writing sRGB values directly makes viewers show
    washed-out colours (brown turns pinkish tan)."""
    c = np.asarray(rgb, np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return [int(round(x * 255)) for x in lin]


def linear_to_srgb(rgb) -> list[int]:
    c = np.asarray(rgb, np.float64) / 255.0
    s = np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)
    return [int(round(x * 255)) for x in s]


def apply_material(mesh: trimesh.Trimesh, rgb, metal: float, rough: float) -> trimesh.Trimesh:
    mat = trimesh.visual.material.PBRMaterial(
        baseColorFactor=[*srgb_to_linear(rgb), 255], metallicFactor=metal, roughnessFactor=rough,
        doubleSided=False)
    mesh.visual = trimesh.visual.TextureVisuals(material=mat)
    return mesh


# ------------------------------------------------------------------ analysis
def _rows(mask: np.ndarray):
    ys = np.where(mask.any(1))[0]
    xl = np.array([np.argmax(mask[y]) for y in ys])
    xr = np.array([mask.shape[1] - 1 - np.argmax(mask[y][::-1]) for y in ys])
    return ys, xl, xr


def best_axis(mask: np.ndarray) -> tuple[float, float]:
    """Vertical axis maximising mirror IoU -> (axis_x, iou). Robust to handles/spouts,
    which skew a simple per-row midpoint when they span many rows."""
    ys, xl, xr = _rows(mask)
    if len(ys) < 5:
        return mask.shape[1] / 2, 0.0
    m = mask[::2, ::2]
    W = m.shape[1]
    flipped = m[:, ::-1]
    best = (0.0, W / 2)
    for a2 in range(int(xl.min()), int(xr.max()) + 1):  # candidate axes at half-pixel steps (2a)
        shift = a2 - (W - 1)                              # mirror about x = a2/2 in half-res px
        mir = np.roll(flipped, shift, axis=1)
        if shift > 0:
            mir[:, :shift] = False
        elif shift < 0:
            mir[:, shift:] = False
        iou = (m & mir).sum() / max((m | mir).sum(), 1)
        if iou > best[0]:
            best = (float(iou), a2 / 2)
    return best[1] * 2, best[0]


def lathe_axis(mask: np.ndarray) -> float:
    """Axis for revolving: centre of the narrower rows. Rows crossing a handle or spout are
    wider and pull a whole-silhouette estimate sideways; cap, neck and plain body rows don't."""
    ys, xl, xr = _rows(mask)
    w = xr - xl
    sel = w <= np.percentile(w, 40) * 1.08
    return float(np.median(((xl + xr) / 2)[sel]))


def symmetry_score(mask: np.ndarray) -> float:
    """IoU of the silhouette with its mirror about the best vertical axis."""
    return best_axis(mask)[1]


def choose_mode(mask: np.ndarray, text: str) -> str | None:
    """Rule-based modes only where they are clearly right; None = use a learned 3D model.
    Mirror symmetry alone is NOT enough for revolve (a teddy or chair front view is
    mirror-symmetric but not rotationally symmetric), so revolve needs a lathe-type word."""
    w = words_of(text)
    if w & FLAT_WORDS:
        return "extrude"
    # handles/spouts lower mirror symmetry (a mug scores ~0.72); revolve adds them back as parts
    if w & LATHE_WORDS and symmetry_score(mask) > 0.55:
        return "revolve"
    return None


# ------------------------------------------------------------------ prompt shape modifiers
def parse_modifiers(text: str) -> dict:
    """Shape words in the prompt adjust the sketch-derived geometry (the drawing stays the
    base; the words refine it): e.g. 'hexagonal', 'cylindrical', 'domed', 'ribbed', 'tall',
    'slim', 'thick handle', 'no handle', 'solid', 'glossy'."""
    t = (text or "").lower()
    w = set(re.findall(r"[a-z]+", t))
    m: dict = {}
    for words, n in ((("triangular", "triangle"), 3), (("square", "cube", "cuboid", "box", "boxy"), 4),
                     (("pentagonal", "pentagon"), 5), (("hexagonal", "hexagon", "hex"), 6),
                     (("octagonal", "octagon"), 8), (("faceted", "lowpoly", "geometric"), 12)):
        if w & set(words):
            m["sections"] = n
    if "low poly" in t or "low-poly" in t:
        m["sections"] = 12
    if w & {"cylindrical", "cylinder", "straight", "tubular"}:
        m["cylindrical"] = True
    if w & {"dome", "domed", "rounded", "round", "spherical", "bulbous"}:
        m["top"] = "dome"
    if w & {"pointed", "conical", "cone", "spire", "tapered"}:
        m["top"] = "point"
    if w & {"ribbed", "ridged", "grooved", "fluted", "corrugated"}:
        m["ribs"] = True
    if w & {"tall", "taller", "elongated", "long"}:
        m["height"] = 1.3
    if w & {"short", "squat", "stubby", "stocky"}:
        m["height"] = 0.75
    if w & {"wide", "wider", "fat", "chubby", "bulky", "broad"}:
        m["width"] = 1.25
    if w & {"slim", "slender", "narrow", "skinny"}:
        m["width"] = 0.8
    if re.search(r"\b(no|without)\s+(a\s+)?(handle|holder)s?\b|handleless", t):
        m["handle"] = 0.0
    elif re.search(r"\b(thick|chunky|big|large|sturdy|nice|strong)\s+(handle|holder)", t):
        m["handle"] = 1.6
    elif re.search(r"\b(thin|slim|small|delicate)\s+(handle|holder)", t):
        m["handle"] = 0.6
    if w & {"solid", "closed", "sealed", "capped", "lid", "lidded"}:
        m["hollow"] = False
    elif w & {"hollow", "open"}:
        m["hollow"] = True
    if w & {"glossy", "shiny", "polished", "lacquered"}:
        m["finish"] = "glossy"
    elif w & {"matte", "rough", "brushed", "frosted"}:
        m["finish"] = "matte"
    if w & {"puffy", "inflated", "plump"}:
        m["depth"] = 1.2
    elif w & {"flat", "thin", "flattened"} and "handle" not in t:
        m["depth"] = 0.5
    return m


def _piecewise_constant(r: np.ndarray, tol: float) -> np.ndarray:
    """'cylindrical': each long section of the profile becomes a straight-sided cylinder;
    short sections between them (a shoulder) become straight slopes, not stair steps."""
    segs, s = [], 0
    for i in range(1, len(r) + 1):
        if i == len(r) or abs(r[i] - np.median(r[s:i])) > tol:
            segs.append((s, i))
            s = i
    min_len = max(4, int(0.08 * len(r)))
    out = r.astype(np.float64).copy()
    longs = [(a, b) for a, b in segs if b - a >= min_len]
    for a, b in longs:
        out[a:b] = np.median(r[a:b])
    for (a0, b0), (a1, b1) in zip(longs, longs[1:]):  # linear transition between cylinders
        if a1 > b0:
            out[b0:a1] = np.linspace(out[b0 - 1], out[a1], a1 - b0 + 2)[1:-1]
    return out.astype(r.dtype)


# ------------------------------------------------------------------ builders
def revolve(mask: np.ndarray, hollow: bool, sections: int = 96,
            mods: dict | None = None, strokes: np.ndarray | None = None) -> tuple[trimesh.Trimesh, np.ndarray, dict]:
    """Returns (mesh in pixel units, y-up), a mask of attached parts (handles) to add, and
    profile info (cap split height, top, bottom) for per-part materials."""
    H, W = mask.shape
    # join stacked parts drawn with small gaps (cap above body, base below)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(5, H // 30)))) > 0
    # A handle pulls the mirror axis towards itself, which pushes part of the cap/neck off-axis
    # (it then turns into lumps). Find the handle, drop it, and re-estimate the axis on the body.
    axis = lathe_axis(mask)
    if strokes is not None and strokes.any():
        mask, strokes, axis = _straighten(mask, strokes, axis)
    core = mask
    for _ in range(2):
        sym, res0 = _symmetric_part(core, axis, strokes)
        big = _large_parts(res0, core, sym)
        if not big.any():
            break
        core = mask & ~cv2.dilate(big.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        core = _largest_component(core)
        axis = lathe_axis(core)
    ys, r = _body_radius(core, axis, strokes)
    # A flat base drawn in perspective is a curve, which reads as a taper; if the profile
    # only narrows in the bottom ~12% of the height, hold the radius (flat base)
    n_base = max(2, int(0.12 * len(r)))
    j = len(r) - n_base + int(np.argmax(r[-n_base:]))
    if r[j] > 0 and np.all(np.diff(r[j:]) <= 1.0):
        r[j:] = r[j]
    # remove hand-drawn wobble but keep steps crisp (cap -> shoulder, rim, base): a median
    # filter preserves edges, a light Gaussian then removes pixel stair-stepping
    from scipy.ndimage import median_filter
    k = max(5, (len(r) // 15) | 1)
    r = median_filter(r, size=k, mode="nearest")
    # edge-preserving: smooth strongly except at genuine steps (cap/shoulder), so no ring banding
    mods = mods or {}
    rg = cv2.GaussianBlur(r.reshape(-1, 1), (1, max(5, (len(r) // 8) | 1)), 0).ravel()
    r = np.where(np.abs(r - rg) > 0.04 * r.max(), r, rg)
    r = cv2.GaussianBlur(r.reshape(-1, 1), (1, 3), 0).ravel()
    # The top of a container rim or a cap is an ellipse in perspective, i.e. a flat circle;
    # it reads as a dome. Hold the radius over the top ~8-10% (a requested dome/point is then
    # built on top of the full-width cap).
    n_top = max(2, int((0.10 if hollow else 0.08) * len(r)))
    peak = r[:n_top].max()
    if peak > 0:  # first row reaching ~full cap width (tolerant of hand-drawn wobble)
        j = int(np.argmax(r[:n_top] >= 0.92 * peak))
        r[:j] = r[j]
    if mods.get("cylindrical"):
        r = _piecewise_constant(r, 0.07 * r.max())
    if mods.get("ribs"):
        idx = np.arange(len(r))
        r = r + 0.025 * r.max() * np.sin(2 * np.pi * idx / max(6, len(r) / 14)) * (r > 0.3 * r.max())
    sections = mods.get("sections", sections)
    # cap = the narrow section at the top, for per-part colour: it ends where the radius jumps
    # well past the cap's own width (lid -> shoulder), and is never wider than 60% of the body
    n0 = max(2, len(r) // 50)
    cap_w = float(np.median(r[:n0]))
    j = 0
    while j < len(r) and r[j] < min(0.6 * r.max(), 1.5 * max(cap_w, 1.0)):
        j += 1
    cap_h = float(H - ys[j]) if 0.03 * len(r) <= j <= 0.4 * len(r) else None
    h = (H - ys).astype(np.float32)             # y-up height in px
    keep = r > 0.5
    h, r = h[keep], r[keep]
    step = max(1, len(h) // 120)
    h, r = h[::step], r[::step]
    if len(h) < 3:
        raise ValueError("profile too short to revolve")
    top, bot = h.max(), h.min()
    order = np.argsort(h)
    h, r = h[order], r[order]
    if hollow:
        wall = max(2.0, 0.06 * r.max())
        base = max(3.0, 0.06 * (top - bot))
        inner_h = h[h >= bot + base]
        inner_r = np.clip(np.interp(inner_h, h, r) - wall, 0.5, None)
        prof = ([(0.0, bot)] + list(zip(r, h)) + list(zip(inner_r[::-1], inner_h[::-1])) +
                [(0.0, bot + base)])
    else:
        cap = []
        r_top = float(r[-1])
        if mods.get("top") == "dome":
            a = np.linspace(0, np.pi / 2, 12)[1:]
            cap = list(zip(r_top * np.cos(a), top + 0.8 * r_top * np.sin(a)))
        elif mods.get("top") == "point":
            cap = [(r_top * 0.5, top + 0.6 * r_top), (0.0, top + 1.2 * r_top)]
        prof = [(0.0, bot)] + list(zip(r, h)) + cap + ([] if cap and cap[-1][0] == 0.0 else [(0.0, top)])
    prof = np.array(prof, np.float64)
    # trimesh.creation.revolve spins (radius, z) around Z; we then rotate Z-up -> Y-up
    mesh = trimesh.creation.revolve(prof, sections=sections)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    mesh.fix_normals()

    # residual = silhouette parts outside the revolved body's silhouette (e.g. a handle)
    body, _ = _symmetric_part(core, axis, strokes)
    residual = mask & ~cv2.dilate(body.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    residual = cv2.morphologyEx(residual.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    parts = _large_parts(residual, mask, body)
    # translate mesh so its axis sits at image column `axis`
    mesh.apply_translation([axis, 0, 0])
    info = {"cap_h": cap_h, "top": float(top), "bot": float(bot), "hollow": hollow,
            "strokes": strokes}  # straightened strokes, same frame as the returned parts
    return mesh, parts, info


def _wall_positions(strokes: np.ndarray, ys: np.ndarray, axis: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-row x of the outer edge of the FIRST steep drawn line left and right of the axis
    (the body walls). Handles drawn outside the walls don't count, even on both sides
    (trophy, kettle). Only steep lines are walls: rim ellipses, rim lines and lids inside
    the body are mostly horizontal. NaN where a row is unusable."""
    a = int(round(axis))
    W = strokes.shape[1]
    left = np.full(len(ys), np.nan, np.float32)
    right = np.full(len(ys), np.nan, np.float32)
    s8 = strokes.astype(np.uint8) * 255
    gx = np.abs(cv2.Sobel(s8, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(s8, cv2.CV_32F, 0, 1, ksize=3))
    steep = ((gx > 0) & (gx >= 0.7 * gy)).astype(np.uint8)
    steep = (cv2.dilate(steep, np.ones((5, 5), np.uint8)) > 0) & strokes
    for i, y in enumerate(ys):
        row, wall_row = strokes[y], steep[y]
        if a < 0 or a >= W or row[a]:
            continue
        found = []
        for step in (-1, 1):
            x = a
            while 0 <= x < W and not wall_row[x]:
                x += step
            if not (0 <= x < W):
                break
            while 0 <= x < W and row[x]:  # cross the wall to its outer edge
                x += step
            found.append(x)
        if len(found) == 2:
            left[i], right[i] = found
    return left, right


def _straighten(mask: np.ndarray, strokes: np.ndarray, axis: float):
    """Hand-drawn lathe objects lean (a trophy cup drawn 20 px left of its stem). Shift each
    row so the midpoint between its walls sits on one vertical axis; handles move with
    their rows. Returns (mask, strokes, axis)."""
    ys = np.where(mask.any(1))[0]
    left, right = _wall_positions(strokes, ys, axis)
    ok = ~np.isnan(left)
    if ok.sum() < 0.4 * len(ys):
        return mask, strokes, axis
    mid = (left + right) / 2
    idx = np.arange(len(ys))
    mid = np.interp(idx, idx[ok], mid[ok])
    from scipy.ndimage import median_filter
    mid = median_filter(mid, size=max(5, (len(ys) // 20) | 1), mode="nearest")
    target = float(np.median(mid))
    W = mask.shape[1]
    shifts = np.clip(np.round(target - mid), -0.15 * W, 0.15 * W).astype(int)
    if np.abs(shifts).max() < 2:
        return mask, strokes, axis
    m2, s2 = np.zeros_like(mask), np.zeros_like(strokes)
    for y, d in zip(ys, shifts):
        if d >= 0:
            m2[y, d:], s2[y, d:] = mask[y, :W - d], strokes[y, :W - d]
        else:
            m2[y, :W + d], s2[y, :W + d] = mask[y, -d:], strokes[y, -d:]
    return m2, s2, target


def _body_radius(mask: np.ndarray, axis: float,
                 strokes: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-row body radius. From the silhouette: the nearer edge, which ignores a handle on
    one side. With strokes, also capped by the drawn wall, which handles protrusions on
    BOTH sides. Falls back to the silhouette for filled (non-line) drawings."""
    ys, xl, xr = _rows(mask)
    r = np.clip(np.minimum(axis - xl, xr - axis), 0, None).astype(np.float32)
    if strokes is None or not strokes.any():
        return ys, r
    left, right = _wall_positions(strokes, ys, axis)
    wall = (right - left) / 2  # half the wall-to-wall width: robust to a slightly-off axis
    ok = ~np.isnan(wall)
    if ok.sum() < 0.4 * len(ys):  # mostly solid drawing: no usable walls
        return ys, r
    idx = np.arange(len(ys))
    wall = np.interp(idx, idx[ok], wall[ok]).astype(np.float32)
    from scipy.ndimage import median_filter
    wall = median_filter(wall, size=max(3, (len(ys) // 25) | 1), mode="nearest")
    return ys, np.minimum(r, wall)


def _symmetric_part(mask: np.ndarray, axis: float,
                    strokes: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(mirror-symmetric body about `axis`, leftover parts) for a silhouette."""
    body = np.zeros_like(mask)
    ys, radii = _body_radius(mask, axis, strokes)
    for y, rr in zip(ys, radii):
        if rr > 0:
            body[y, int(np.floor(axis - rr)):int(np.ceil(axis + rr)) + 1] = True
    residual = mask & ~cv2.dilate(body.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    residual = cv2.morphologyEx(residual.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    return body, residual


def _large_parts(residual: np.ndarray, mask: np.ndarray, body: np.ndarray | None = None) -> np.ndarray:
    """Residual components that are real parts (handles, spouts): long, substantial and
    sticking out from the body. Wobble slivers along the outline and stroke hooks are not."""
    out = np.zeros_like(residual)
    if not residual.any():
        return out
    rows = np.where(mask.any(1))[0]
    body_h = np.ptp(rows) + 1
    body_w = np.median(mask.sum(1)[rows])
    dist = None
    if body is not None and body.any():
        dist = cv2.distanceTransform((~body).astype(np.uint8), cv2.DIST_L2, 5)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(residual.astype(np.uint8))
    for i in range(1, n):
        comp = lab == i
        extent = max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        area = stats[i, cv2.CC_STAT_AREA]
        big = (extent > 0.2 * body_h and area > 0.008 * mask.sum()) or area > 0.04 * mask.sum()
        sticks_out = dist is None or dist[comp].max() > 0.12 * body_w
        if big and sticks_out:
            out |= comp
    return out


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if n <= 2:
        return mask
    return lab == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))


def inflate(mask: np.ndarray, depth: float = 0.75, res: int | None = None,
            min_thick: float = 1.5) -> trimesh.Trimesh:
    """Distance-field inflation in pixel units (x right, y up, z towards viewer)."""
    from skimage.measure import marching_cubes
    H, W = mask.shape
    scale = 1.0
    if res and max(H, W) > res:
        scale = max(H, W) / res
        mask = cv2.resize(mask.astype(np.uint8) * 255, (int(W / scale), int(H / scale)),
                          interpolation=cv2.INTER_AREA) > 40
    a8 = mask.astype(np.uint8)
    d = cv2.distanceTransform(a8, cv2.DIST_L2, 5)
    R = max(d.max(), 1)
    hgt = np.sqrt(np.clip(d * (2 * R - d), 0, None)) * depth
    hgt = cv2.GaussianBlur(hgt, (0, 0), max(1.0, mask.shape[0] * 0.012)) + min_thick
    sdf = d - cv2.distanceTransform(1 - a8, cv2.DIST_L2, 5)
    sdf = cv2.GaussianBlur(sdf, (0, 0), 0.8)
    zmax = int(np.ceil(hgt.max())) + 2
    z = np.arange(-zmax, zmax + 1, dtype=np.float32)
    vol = np.minimum(sdf[:, :, None], hgt[:, :, None] - np.abs(z)[None, None, :])
    vol = np.pad(vol, 1, constant_values=-1)
    verts, faces, _, _ = marching_cubes(vol, level=0.0)
    verts -= 1
    h = mask.shape[0]
    xyz = np.c_[verts[:, 1], h - verts[:, 0], verts[:, 2] - zmax] * scale
    mesh = trimesh.Trimesh(xyz, faces, process=True)
    trimesh.smoothing.filter_taubin(mesh, iterations=8)
    mesh.fix_normals()
    return mesh


def extrude(mask: np.ndarray, depth_frac: float = 0.15) -> trimesh.Trimesh:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    H, W = mask.shape
    m = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for i, c in enumerate(contours):
        if hier[0][i][3] != -1 or cv2.contourArea(c) < 30:
            continue  # holes are handled with their parent
        c = cv2.approxPolyDP(c, 1.0, True)[:, 0, :]
        holes = [cv2.approxPolyDP(contours[j], 1.0, True)[:, 0, :] for j in range(len(contours))
                 if hier[0][j][3] == i and cv2.contourArea(contours[j]) > 20]
        flip = lambda p: [(x, H - y) for x, y in p]
        poly = Polygon(flip(c), [flip(hh) for hh in holes if len(hh) >= 3]).buffer(0)
        if not poly.is_empty:
            polys.append(poly)
    if not polys:
        raise ValueError("no closed outline to extrude")
    shape = unary_union(polys)
    depth = depth_frac * max(H, W) * (shape.area / (H * W)) ** 0.5 * 2
    geoms = shape.geoms if hasattr(shape, "geoms") else [shape]
    parts = [trimesh.creation.extrude_polygon(g, height=max(depth, 4.0)) for g in geoms]
    mesh = trimesh.util.concatenate(parts)
    mesh.apply_translation([0, 0, -mesh.bounds[:, 2].mean()])
    return mesh


def _tube_mask(comp: np.ndarray, strokes: np.ndarray | None, body_width: float) -> np.ndarray:
    """Attached parts (handles) as tubes following the drawn strokes, so a handle keeps
    its hole instead of becoming a solid 'D'."""
    if strokes is None:
        return comp
    region = cv2.dilate(comp.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    s = (strokes & region).astype(np.uint8)
    if s.sum() < 20:
        return comp
    thick = max(5, int(0.07 * body_width)) | 1
    tube = cv2.dilate(s, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thick, thick))) > 0
    # double-line handle (outer + inner arc): the handle is the band between the arcs
    n, lab = cv2.connectedComponents(s)
    hulls = []
    for i in range(1, n):
        pts = cv2.findNonZero((lab == i).astype(np.uint8))
        if pts is not None and len(pts) > 15:
            h = np.zeros_like(s)
            cv2.drawContours(h, [cv2.convexHull(pts)], -1, 1, thickness=-1)
            hulls.append(h)
    if len(hulls) >= 2:
        hulls.sort(key=lambda h: h.sum(), reverse=True)
        band = (hulls[0] > 0) & ~(hulls[1] > 0)
        band &= comp | tube
        if band.sum() > 0.3 * tube.sum():
            return band | tube
    return tube


def build(mask: np.ndarray, out_dir: Path, mode: str = "auto", text: str = "",
          material: str = "auto", strokes: np.ndarray | None = None,
          known_parts: list[tuple[str, np.ndarray]] | None = None) -> tuple[str, str, list[str]]:
    """Returns (glb_path, mode_used, log_lines). `known_parts` (prompt-only templates) are
    exact (name, mask) attachments such as a handle or spout, used instead of detecting
    them from the silhouette."""
    logs = []
    if mode == "auto":
        mode = choose_mode(mask, text) or "inflate"
        logs.append(f"auto geometry mode -> {mode} (symmetry {symmetry_score(mask):.2f})")
    look = prompt_parse(text, material)
    mods = parse_modifiers(text)
    if mods:
        logs.append("prompt shape modifiers: " + ", ".join(f"{k}={v}" for k, v in mods.items()))
    named: list[tuple[str, trimesh.Trimesh]] = []
    if mode == "revolve":
        hollow = mods.get("hollow", bool(object_words(text) & CONTAINER_WORDS))
        body, residual, info = revolve(mask, hollow, mods=mods,
                                       strokes=strokes if known_parts is None else None)
        named += _split_body(body, info, set(look["parts"]))
        # attached parts (handles): long and substantial; wobble slivers and stroke hooks are not
        handle = mods.get("handle", 1.0)
        if known_parts is not None:
            for part, pm in known_parts:
                if handle and handle != 1.0:  # 'thick' / 'thin handle' on an exact template tube
                    k = max(1, int(abs(handle - 1) * 10)) | 1
                    op = cv2.dilate if handle > 1 else cv2.erode
                    pm = op(pm.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
                if handle and pm.any():
                    named.append((part, inflate(pm, depth=0.9, res=200, min_thick=2.0)))
            n = 1 + sum(k in ("handle", "spout") for k, _ in named)
        else:
            n, lab, st, _ = (cv2.connectedComponentsWithStats(residual.astype(np.uint8)) if handle
                             else (1, None, None, None))
            # Attached parts are handles (a trophy has two). Only spouted vessels get a spout:
            # there the smallest attached part is the spout.
            spouted = bool(object_words(text) & SPOUT_WORDS) or "spout" in text.lower()
            spout_i = 1 + int(np.argmin(st[1:, cv2.CC_STAT_AREA])) if (spouted and n > 2) else -1
            for i in range(1, n):
                named.append(("spout" if i == spout_i else "handle",
                              inflate(_tube_mask(lab == i, info.get("strokes", strokes),
                                                 body.extents[0] * handle),
                                      depth=0.9, res=200, min_thick=2.0)))
        logs.append(f"revolved body (hollow={hollow}, sections={mods.get('sections', 96)}) + "
                    f"{n - 1} attached part(s)")
    elif mode == "extrude":
        thin_obj = bool(words_of(text) & THIN_FLAT_WORDS)  # rackets, scissors, wheels are slim
        named.append(("body", extrude(mask, depth_frac=(0.045 if thin_obj else 0.15) * mods.get("depth", 1.0))))
        logs.append("extruded flat outline")
    else:
        named.append(("body", inflate(mask, depth=0.75 * mods.get("depth", 1.0), res=200)))
        logs.append("inflated silhouette")

    wf, hf = mods.get("width", 1.0), mods.get("height", 1.0)
    if wf != 1.0 or hf != 1.0:  # 'tall' / 'slim' etc. scale about the base centre (before UVs)
        lo = np.min([m.bounds[0] for _, m in named], axis=0)
        hi = np.max([m.bounds[1] for _, m in named], axis=0)
        c = (lo + hi) / 2
        c[1] = lo[1]
        T = np.diag([wf, hf, wf, 1.0])
        T[:3, 3] = c - c * np.array([wf, hf, wf])
        for _, m in named:
            m.apply_transform(T)
    from . import textures
    lo = np.min([m.bounds[0] for _, m in named], axis=0)
    hi = np.max([m.bounds[1] for _, m in named], axis=0)
    tile = float((hi - lo).max()) / 3.0  # same texture density across all parts
    scene = trimesh.Scene()
    textured = []
    for i, (part, m) in enumerate(named):
        rgb, metal, rough = look["parts"].get(part, look["body"])
        if mods.get("finish") == "glossy":
            rough = min(rough, 0.12)
        elif mods.get("finish") == "matte":
            rough = max(rough, 0.85)
        mat_name = look["mat_names"].get(part, look["mat_names"]["body"])
        mapping = ("planar" if mode == "extrude" else
                   "cyl" if part in ("handle", "spout") else "cyl_y")
        tm = textures.apply(m, mat_name, rgb, metal, rough, tile, mapping)
        if tm is not None:
            textured.append(f"{part}:{textures.canonical(mat_name)}")
        scene.add_geometry(tm if tm is not None else apply_material(m, rgb, metal, rough),
                           geom_name=f"{part}_{i}")
    if textured:
        logs.append("textures: " + ", ".join(textured))
    p = out_dir / f"sketch3d_{mode}_{uuid.uuid4().hex[:6]}.glb"
    scene.export(p)
    logs.append(f"materials: {look['summary']}")
    return str(p), mode, logs


def _split_body(body: trimesh.Trimesh, info: dict, wanted: set[str]) -> list[tuple[str, trimesh.Trimesh]]:
    """Cut the revolved body into cap / rim / base / body pieces when the prompt colours them
    separately ('a bottle with a red cap', 'a vase with a gold rim')."""
    top, bot = info["top"], info["bot"]
    cuts = []  # (name, y_low, y_high)
    if "cap" in wanted and info.get("cap_h"):
        cuts.append(("cap", info["cap_h"], np.inf))
    elif "rim" in wanted:
        cuts.append(("rim", top - 0.06 * (top - bot), np.inf))
    if "base" in wanted:
        cuts.append(("base", -np.inf, bot + 0.08 * (top - bot)))
    if not cuts:
        return [("body", body)]
    pieces, rest = [], body
    for name, lo, hi in cuts:
        try:
            y = lo if np.isfinite(lo) else hi
            keep_above = np.isfinite(lo)
            normal = np.array([0, 1.0, 0]) if keep_above else np.array([0, -1.0, 0])
            part = trimesh.intersections.slice_mesh_plane(rest, normal, [0, y, 0], cap=True)
            rest = trimesh.intersections.slice_mesh_plane(rest, -normal, [0, y, 0], cap=True)
            if len(part.faces) and len(rest.faces):
                pieces.append((name, part))
        except Exception:  # slicing is cosmetic; never fail the model over it
            continue
    return [("body", rest)] + pieces
