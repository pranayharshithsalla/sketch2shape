"""Silhouette carving and adding (Phase D): make the model's view from the sketch's side match
the hole-aware input mask.

  1 align    best of 4 side views + box fit (accuracy.align); geometry moves into that frame
  2 carve    voxelise each offending part and delete every voxel whose projection falls on
             background or a through-hole of the mask (opens racket heads, handle loops)
  3 add      where the mask is solid but the model is empty (missing handle, thin wire), extrude
             the missing pixels along the view axis: depth from the neighbouring model columns,
             or the measured stroke width for thin parts, centred on the neighbours' mid-plane
  4 re-mesh  marching cubes + Taubin smoothing (no shrinkage); tiny specks dropped
  5 colour   new vertices take the colour of the nearest original surface point; parts that
             needed no carving keep their textures untouched

Carving works along the view axis only: a hole that is not visible from the sketch's side
cannot be opened, and hidden-side geometry stays inferred.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import trimesh

from .accuracy import Alignment, align, load_parts, measure, silhouette

log = logging.getLogger(__name__)


def _colours(m: trimesh.Trimesh) -> np.ndarray:
    """Per-vertex RGB of any mesh (texture sampled, PBR factor, or vertex colours)."""
    v = m.visual
    try:
        if isinstance(v, trimesh.visual.TextureVisuals):
            mat = v.material
            tex = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
            if tex is not None and v.uv is not None:
                t = np.asarray(tex.convert("RGB"))
                uv = v.uv % 1.0
                return t[((1 - uv[:, 1]) * (t.shape[0] - 1)).astype(int), (uv[:, 0] * (t.shape[1] - 1)).astype(int)]
            base = getattr(mat, "baseColorFactor", None)
            if base is not None:
                from .sketch3d import linear_to_srgb
                return np.tile(linear_to_srgb(np.asarray(base)[:3]), (len(m.vertices), 1))
        if v.kind in ("vertex", "face"):
            return np.asarray(v.vertex_colors)[:, :3]
    except Exception:
        pass
    return np.full((len(m.vertices), 3), 180)


def _voxelise(m: trimesh.Trimesh, pitch: float, origin: np.ndarray, shape) -> np.ndarray:
    """Solid occupancy of mesh `m` (already in pixel space) on the shared grid.
    Column parity: every triangle is rasterised at the column centres it covers, the depth
    hits of each column are sorted, and the space between entry/exit pairs is filled. Fully
    vectorised (trimesh's subdivision voxeliser takes minutes on a large mug)."""
    occ = _voxelise_parity(m, pitch, origin, shape)
    if occ is not None:
        return occ
    return _voxelise_trimesh(m, pitch, origin, shape)


def _voxelise_parity(m, pitch, origin, shape):
    tri = (m.vertices[m.faces] - origin) / pitch          # grid units, (F, 3, 3)
    xy = tri[:, :, :2] - 0.5                                # column centres at integer coords
    lo = np.ceil(xy.min(1)).astype(int)
    hi = np.floor(xy.max(1)).astype(int)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.array(shape[:2]) - 1)
    w, h = hi[:, 0] - lo[:, 0] + 1, hi[:, 1] - lo[:, 1] + 1
    ok = (w > 0) & (h > 0)
    if not ok.any():
        return np.zeros(shape, bool)
    f = np.nonzero(ok)[0]
    n = (w[f] * h[f])
    if n.sum() > 60_000_000:
        return None
    fi = np.repeat(f, n)
    k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    px = lo[fi, 0] + k % w[fi]
    py = lo[fi, 1] + k // w[fi]
    a, b, c = xy[fi, 0], xy[fi, 1], xy[fi, 2]
    d = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    good = np.abs(d) > 1e-12
    d = np.where(good, d, 1.0)
    l1 = ((b[:, 1] - c[:, 1]) * (px - c[:, 0]) + (c[:, 0] - b[:, 0]) * (py - c[:, 1])) / d
    l2 = ((c[:, 1] - a[:, 1]) * (px - c[:, 0]) + (a[:, 0] - c[:, 0]) * (py - c[:, 1])) / d
    l3 = 1 - l1 - l2
    inside = good & (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
    # half-open rule on shared edges would be exact; tiny duplicates are removed below instead
    z = l1 * tri[fi, 0, 2] + l2 * tri[fi, 1, 2] + l3 * tri[fi, 2, 2]
    px, py, z = px[inside], py[inside], z[inside]
    col = px * shape[1] + py
    order = np.lexsort((z, col))
    col, z, px, py = col[order], z[order], px[order], py[order]
    # merge hits that coincide (a column passing exactly through a shared edge/vertex)
    dup = np.r_[False, (col[1:] == col[:-1]) & (np.abs(z[1:] - z[:-1]) < 1e-4)]
    col, z, px, py = col[~dup], z[~dup], px[~dup], py[~dup]
    starts = np.r_[True, col[1:] != col[:-1]]
    rank = np.arange(len(col)) - np.maximum.accumulate(np.where(starts, np.arange(len(col)), 0))
    counts = np.bincount(np.cumsum(starts) - 1)
    odd = (counts % 2 == 1).mean() if len(counts) else 0
    if odd > 0.2:  # mesh not closed: parity is meaningless
        return None
    entry = (rank % 2 == 0)
    nxt = np.r_[np.arange(1, len(col)), 0]
    pair = entry & np.r_[col[1:] == col[:-1], False]
    zi0 = np.clip(np.ceil(z[pair] - 0.5).astype(int), 0, shape[2] - 1)
    zi1 = np.clip(np.floor(z[nxt[pair]] - 0.5).astype(int), -1, shape[2] - 1)
    keep = zi1 >= zi0
    diff = np.zeros((shape[0], shape[1], shape[2] + 1), np.int16)
    np.add.at(diff, (px[pair][keep], py[pair][keep], zi0[keep]), 1)
    np.add.at(diff, (px[pair][keep], py[pair][keep], zi1[keep] + 1), -1)
    return np.cumsum(diff, axis=2)[:, :, :-1] > 0


def _voxelise_trimesh(m, pitch, origin, shape) -> np.ndarray:
    occ = np.zeros(shape, bool)
    try:
        vg = m.voxelized(pitch).fill()
        pts = vg.points
    except Exception:
        pts = m.sample(20000)
    idx = np.floor((pts - origin) / pitch).astype(int)
    ok = np.all((idx >= 0) & (idx < np.array(shape)), axis=1)
    occ[tuple(idx[ok].T)] = True
    return occ


def _remesh(occ: np.ndarray, pitch: float, origin: np.ndarray, min_faces: int = 40) -> trimesh.Trimesh | None:
    from skimage.measure import marching_cubes
    if occ.sum() < 8:
        return None
    pad = np.pad(occ.astype(np.float32), 1)
    v, f, _, _ = marching_cubes(pad, 0.5)
    mesh = trimesh.Trimesh((v - 1) * pitch + origin, f[:, ::-1], process=True)
    trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=8)
    parts = [p for p in mesh.split(only_watertight=False) if len(p.faces) >= min_faces]
    return trimesh.util.concatenate(parts) if parts else None


def refine(glb: str, analysis, out_glb: str | Path, pitch_px: float = 1.0, strict: bool = True,
           procedural: bool = True, debug_dir: Path | None = None) -> dict:
    """Carve + add the model at `glb` against `analysis` (preprocess's InputAnalysis).
    strict: carve exactly to the mask (default); False keeps a 1-voxel tolerance band.
    procedural: rebuild detected strings / wires / spokes as clean cylinders (Phase E) instead
    of whatever the reconstructor made of them."""
    parts = load_parts(glb)
    mask = analysis.mask
    al, iou0 = align(parts, mask)
    before = measure(parts, analysis, al)

    # geometry into pixel space (u right, v up, z toward the viewer), so the grid matches the mask
    H = mask.shape[0]

    def to_px(v):
        p = al.to_pixels(v)
        return np.c_[p[:, 0], H - p[:, 1], p[:, 2]]

    px_parts = []
    for m in parts:
        c = m.copy()
        c.vertices = to_px(m.vertices)
        px_parts.append(c)
    # tubes on one mid-plane suit flat things (racket, wheel); on a deep object (a wire basket)
    # the strings belong to both walls, so they are kept as carved geometry instead
    ext = np.ptp(np.vstack([q.vertices for q in px_parts]), axis=0)
    flat_obj = ext[2] < 0.35 * max(ext[0], ext[1])
    fam_px = _family_pixels(analysis) if procedural and analysis.families and flat_obj else None
    allv = np.vstack([m.vertices for m in px_parts])
    lo, hi = allv.min(0) - 3 * pitch_px, allv.max(0) + 3 * pitch_px
    origin = np.array([0.0, 0.0, lo[2]])
    shape = (int(np.ceil(mask.shape[1] / pitch_px)), int(np.ceil(mask.shape[0] / pitch_px)),
             int(np.ceil((hi[2] - lo[2]) / pitch_px)) + 1)
    # mask on the grid: column (i, j) <- pixel (x = i*pitch, y = H - j*pitch)
    gx = ((np.arange(shape[0]) + 0.5) * pitch_px).astype(int).clip(0, mask.shape[1] - 1)
    gy = (H - (np.arange(shape[1]) + 0.5) * pitch_px).astype(int).clip(0, mask.shape[0] - 1)

    if fam_px is not None:  # strings are rebuilt as tubes: drop them from the solid, inside the hole only
        near_hole = cv2.dilate(analysis.hole_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        fam_px &= near_hole
    solid = mask & ~fam_px if fam_px is not None else mask
    keep_mask = solid[np.ix_(gy, gx)].T                                 # (nx, ny) solid columns
    tol = keep_mask if strict else cv2.dilate(keep_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0

    out, carved_parts, carved_vox = trimesh.Scene(), 0, 0
    occ_all = np.zeros(shape, bool)
    kept = []
    for k, (m, mp) in enumerate(zip(parts, px_parts)):
        occ = _voxelise(mp, pitch_px, origin, shape)
        cols = occ.any(2)
        bad = int((cols & ~tol).sum())
        if bad > max(6, 0.01 * cols.sum()):
            n0 = int(occ.sum())
            occ &= tol[:, :, None]
            carved_vox += n0 - int(occ.sum())
            new = _remesh(occ, pitch_px, origin)
            if new is not None:
                from scipy.spatial import cKDTree
                cols_src = _colours(mp)
                _, nn = cKDTree(mp.vertices).query(new.vertices)
                new.visual = trimesh.visual.ColorVisuals(
                    new, vertex_colors=np.c_[cols_src[nn], np.full(len(nn), 255)].astype(np.uint8))
                new.metadata["name"] = mp.metadata.get("name") or f"part_{k}"
                kept.append(new)
            carved_parts += 1
        else:
            kept.append(mp)
        occ_all |= occ

    # ADD: solid in the mask, empty in the model
    cover = occ_all.any(2)
    missing = keep_mask & ~cover
    missing = cv2.morphologyEx(missing.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    added_vox = 0
    if missing.sum() > 4:
        zs = np.arange(shape[2])
        occ_cols = occ_all.any(2)
        zmin = np.where(occ_cols, occ_all.argmax(2), 0)
        zmax = np.where(occ_cols, shape[2] - 1 - occ_all[:, :, ::-1].argmax(2), 0)
        mid_all = np.median((zmin + zmax)[occ_cols]) / 2 if occ_cols.any() else shape[2] / 2
        thin_d = max(2, int(round(1.6 * analysis.stroke_px / pitch_px)))
        n, lab = cv2.connectedComponents(missing.astype(np.uint8), connectivity=8)
        add = np.zeros(shape, bool)
        dist = cv2.distanceTransform(keep_mask.astype(np.uint8), cv2.DIST_L2, 3)
        for i in range(1, n):
            comp = lab == i
            ring = (cv2.dilate(comp.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & occ_cols
            if ring.any():
                zc = float(np.median((zmin[ring] + zmax[ring]) / 2))
                depth = float(np.median(zmax[ring] - zmin[ring] + 1))
            else:
                zc, depth = mid_all, thin_d
            width = 2 * float(dist[comp].max())
            depth = min(depth, max(thin_d, width * 1.2))   # thin parts get round-ish sections
            z0, z1 = int(max(0, zc - depth / 2)), int(min(shape[2] - 1, zc + depth / 2))
            add[comp, z0:z1 + 1] = True
        added_vox = int(add.sum())
        new = _remesh(add, pitch_px, origin, min_faces=12)
        if new is not None:
            from scipy.spatial import cKDTree
            src = trimesh.util.concatenate(px_parts)
            cols_src = _colours(src) if len(px_parts) == 1 else np.vstack([_colours(p) for p in px_parts])
            _, nn = cKDTree(src.vertices).query(new.vertices)
            new.visual = trimesh.visual.ColorVisuals(
                new, vertex_colors=np.c_[cols_src[nn], np.full(len(nn), 255)].astype(np.uint8))
            new.metadata["name"] = "added_detail"
            kept.append(new)

    strings = 0
    if fam_px is not None:
        tubes = _string_tubes(analysis, occ_all, origin, pitch_px, H, solid)
        if tubes is not None:
            tubes.metadata["name"] = "strings"
            kept.append(tubes)
            strings = sum(len(f["lines"]) for f in analysis.families)
    from .export import part_names
    for m, name in zip(kept, part_names(kept)):
        out.add_geometry(m, geom_name=name)
    out_glb = Path(out_glb)
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    out.export(out_glb)
    ident = Alignment(0, 1, 1, 0, 0)  # refined geometry already lives in the mask frame
    after = measure(_flip_back(kept, H), analysis, None)
    rep = {"glb": str(out_glb), "view_yaw": al.yaw, "parts_carved": carved_parts, "voxels_carved": carved_vox,
           "voxels_added": added_vox, "procedural_strings": strings, "before": _public(before), "after": _public(after)}
    if debug_dir is not None:
        from .accuracy import report_image
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(Path(debug_dir) / "refine_before.png"),
                    cv2.cvtColor(report_image(analysis, before["_silhouette"]), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(Path(debug_dir) / "refine_after.png"),
                    cv2.cvtColor(report_image(analysis, after["_silhouette"]), cv2.COLOR_RGB2BGR))
    log.info("refine: %s -> %s (carved %d vox in %d part(s), added %d vox)",
             before["accuracy"], after["accuracy"], carved_vox, carved_parts, added_vox)
    return rep


def _family_pixels(analysis) -> np.ndarray:
    m = np.zeros(analysis.mask.shape, np.uint8)
    w = max(2, int(round(analysis.stroke_px))) + 2
    for f in analysis.families:
        for x0, y0, x1, y1 in f["lines"]:
            cv2.line(m, (int(x0), int(y0)), (int(x1), int(y1)), 1, w)
    return m > 0


def _extend_to_solid(p, d, solid, limit):
    """Walk from p along d until the line meets solid (the frame) so the string attaches."""
    h, w = solid.shape
    for t in range(int(limit)):
        q = np.round(p + d * t).astype(int)
        if not (0 <= q[0] < w and 0 <= q[1] < h):
            break
        if solid[q[1], q[0]]:
            return p + d * (t + 1.5)
    return p


def _string_tubes(analysis, occ, origin, pitch, H, solid):
    """Phase E: each string / wire / spoke as a thin cylinder from frame to frame, on the
    mid-plane of the solid around its ends."""
    cols = occ.any(2)
    zmin = np.where(cols, occ.argmax(2), 0)
    zmax = np.where(cols, occ.shape[2] - 1 - occ[:, :, ::-1].argmax(2), 0)
    mid = (zmin + zmax) / 2.0
    zc_all = float(np.median(mid[cols])) if cols.any() else occ.shape[2] / 2
    r = max(0.9, 0.55 * analysis.stroke_px)  # match the drawn line width
    tubes = []
    for f in analysis.families:
        for x0, y0, x1, y1 in f["lines"]:
            p0, p1 = np.array([x0, y0], float), np.array([x1, y1], float)
            d = (p1 - p0) / (np.linalg.norm(p1 - p0) + 1e-9)
            lim = 6 * analysis.stroke_px + 8
            p0, p1 = _extend_to_solid(p0, -d, solid, lim), _extend_to_solid(p1, d, solid, lim)
            zs = []
            for q in (p0, p1):
                i, j = int(q[0] / pitch), int((H - q[1]) / pitch)
                win = cols[max(i - 4, 0):i + 5, max(j - 4, 0):j + 5]
                if win.any():
                    zs.append(float(np.median(mid[max(i - 4, 0):i + 5, max(j - 4, 0):j + 5][win])))
            z = (np.mean(zs) if zs else zc_all) * pitch + origin[2]
            a3, b3 = np.array([p0[0], H - p0[1], z]), np.array([p1[0], H - p1[1], z])
            if np.linalg.norm(b3 - a3) < 2:
                continue
            tubes.append(trimesh.creation.cylinder(radius=r, segment=[a3, b3], sections=10))
    if not tubes:
        return None
    t = trimesh.util.concatenate(tubes)
    t.visual = trimesh.visual.ColorVisuals(t, vertex_colors=np.tile([235, 235, 228, 255], (len(t.vertices), 1)))
    return t


def _flip_back(meshes, H):
    return meshes  # measure() re-aligns; the geometry is already upright in pixel units


def _public(m: dict) -> dict:
    return {k: v for k, v in m.items() if not k.startswith("_")}


def ensure_accuracy(raw_glb: str, analysis, out_dir: Path, target: float | None = None,
                    procedural: bool = True) -> tuple[str, dict]:
    """Measure, then refine with escalating strictness until Accuracy >= target (max 2 refine
    passes here; the caller may switch backend as a 3rd escalation). Keeps the best result and
    the full history. -> (glb to export, report)."""
    from . import config
    target = config.ACCURACY_TARGET if target is None else target
    before = measure(raw_glb, analysis)
    history = [{"step": "raw model", "accuracy": before["accuracy"], "iou": before["iou"],
                "thin_recall": before["thin_recall"], "holes": f"{before['holes_open']}/{before['holes_expected']}"}]
    best_glb, best = raw_glb, _public(before)
    need = before["accuracy"] < target or (procedural and bool(analysis.families))
    if need:
        for k, pitch in enumerate((1.0,)):  # one fast pass; a finer 2nd pass was too slow on CPU
            try:
                r = refine(raw_glb, analysis, Path(out_dir) / f"refined_{k + 1}.glb", pitch_px=pitch,
                           procedural=procedural, debug_dir=Path(out_dir) / "debug")
            except Exception as e:  # never lose the raw model because a refinement failed
                history.append({"step": f"refine pass {k + 1}", "error": str(e)[:160]})
                break
            a = r["after"]
            history.append({"step": f"refine pass {k + 1} (carve+add, {pitch}px voxels"
                                    f"{', procedural strings' if r['procedural_strings'] else ''})",
                            "accuracy": a["accuracy"], "iou": a["iou"], "thin_recall": a["thin_recall"],
                            "holes": f"{a['holes_open']}/{a['holes_expected']}"})
            if a["accuracy"] > best["accuracy"] or (r["procedural_strings"] and a["accuracy"] >= best["accuracy"] - 0.02):
                best_glb, best = r["glb"], a
            if best["accuracy"] >= target:
                break
    rep = {**best, "before": _public(before), "refined": best_glb != raw_glb, "target": target,
           "history": history, "raw_glb": raw_glb,
           "summary": analysis.summary()}
    if best["lost"]:
        rep["warning"] = "Not fully reproduced: " + "; ".join(best["lost"])
    return best_glb, rep
