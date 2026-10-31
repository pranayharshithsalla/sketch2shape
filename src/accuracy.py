"""Structural accuracy of a 3D model against the analysed input (Phase F metrics).

  align()   best of the 4 side views + fit to the mask's box: model view coords -> mask pixels
  measure() silhouette IoU, thin-feature recall, expected vs found through-holes, and
            Accuracy = 0.4 IoU + 0.3 thin recall + 0.3 hole match
Silhouettes are rasterised orthographically with one cv2.fillPoly call (fast, no OpenGL).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import trimesh

W_IOU, W_THIN, W_HOLES = 0.4, 0.3, 0.3


@dataclass
class Alignment:
    yaw: float           # degrees about +y: which side of the model faces the viewer
    sx: float            # model units -> mask pixels
    sy: float
    tx: float
    ty: float

    def rot(self) -> np.ndarray:
        a = math.radians(self.yaw)
        return np.array([[math.cos(a), 0, -math.sin(a)], [0, 1, 0], [math.sin(a), 0, math.cos(a)]])

    def to_pixels(self, v: np.ndarray) -> np.ndarray:
        """World vertices -> (u, v_img, depth) in mask pixel units (image rows grow downward)."""
        p = v @ self.rot().T
        sz = 0.5 * (self.sx + self.sy)
        return np.c_[p[:, 0] * self.sx + self.tx, -p[:, 1] * self.sy + self.ty, p[:, 2] * sz]


def load_parts(glb: str) -> list[trimesh.Trimesh]:
    scene = trimesh.load(glb, force="scene")
    return [g for g in scene.dump() if isinstance(g, trimesh.Trimesh) and len(g.faces)]


def silhouette(meshes, al: Alignment, shape) -> np.ndarray:
    """Union of all projected triangles (fillPoly on many polygons XORs overlaps, so each
    triangle is filled on its own)."""
    img = np.zeros(shape, np.uint8)
    h, w = shape
    for m in meshes:
        uv = al.to_pixels(m.vertices)[:, :2]
        tri = np.round(uv[m.faces] * 4).astype(np.int32)  # 2 bits sub-pixel precision
        lo, hi = tri.min(1), tri.max(1)
        keep = (hi[:, 0] >= 0) & (hi[:, 1] >= 0) & (lo[:, 0] < w * 4) & (lo[:, 1] < h * 4)
        for t in tri[keep]:
            cv2.fillConvexPoly(img, t, 1, lineType=cv2.LINE_8, shift=2)
    return img > 0


def _shift(m: np.ndarray, dx: int, dy: int) -> np.ndarray:
    out = np.zeros_like(m)
    h, w = m.shape
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = m[max(-dy, 0):h - max(dy, 0),
                                                                  max(-dx, 0):w - max(dx, 0)]
    return out


def _box(mask):
    ys, xs = np.nonzero(mask)
    return xs.min(), xs.max(), ys.min(), ys.max()


def align(meshes, mask: np.ndarray) -> tuple[Alignment, float]:
    """Pick the side view and box-fit that best overlays the model on the mask (IoU)."""
    bx0, bx1, by0, by1 = _box(mask)
    best, best_iou = None, -1.0
    allv = np.vstack([m.vertices for m in meshes])
    for yaw in (0, 90, 180, 270):
        al = Alignment(yaw, 1, 1, 0, 0)
        p = allv @ al.rot().T
        x0, x1, y0, y1 = p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max()
        sx = (bx1 - bx0 + 1) / max(x1 - x0, 1e-9)
        sy = (by1 - by0 + 1) / max(y1 - y0, 1e-9)
        if not 0.6 <= sx / sy <= 1.67:  # that side would need a big stretch to fit: wrong view
            continue
        cand = Alignment(yaw, sx, sy, bx0 - x0 * sx, by0 + y1 * sy)
        sil0 = silhouette(meshes, cand, mask.shape)
        for dx in (-2, -1, 0, 1, 2):  # small shift search around the box fit
            for dy in (-2, -1, 0, 1, 2):
                sil = _shift(sil0, dx, dy)
                iou = float((sil & mask).sum() / max((sil | mask).sum(), 1))
                if iou > best_iou:
                    best, best_iou = Alignment(yaw, sx, sy, cand.tx + dx, cand.ty + dy), iou
    if best is None:  # every side needs a stretch: fall back to a uniform fit of the front
        p = allv
        s_ = min((bx1 - bx0 + 1) / max(np.ptp(p[:, 0]), 1e-9), (by1 - by0 + 1) / max(np.ptp(p[:, 1]), 1e-9))
        best = Alignment(0, s_, s_, (bx0 + bx1) / 2 - p[:, 0].mean() * s_, (by0 + by1) / 2 + p[:, 1].mean() * s_)
        sil = silhouette(meshes, best, mask.shape)
        best_iou = float((sil & mask).sum() / max((sil | mask).sum(), 1))
    return best, best_iou


def _hole_status(sil: np.ndarray, analysis) -> tuple[int, int, list[dict]]:
    """-> (expected holes found open, extra holes, per-hole detail)."""
    lines = np.zeros(sil.shape, np.uint8)
    for f in analysis.families:
        for x0, y0, x1, y1 in f["lines"]:
            cv2.line(lines, (int(x0), int(y0)), (int(x1), int(y1)), 1, int(analysis.stroke_px * 2) + 3)
    detail, found = [], 0
    for h in analysis.holes:
        region = (analysis.labels == h["id"]) & (lines == 0)
        region = cv2.erode(region.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        open_frac = float((~sil[region]).mean()) if region.any() else 0.0
        ok = open_frac >= 0.6
        found += ok
        detail.append({"kind": h["kind"], "open_fraction": round(open_frac, 2), "open": bool(ok)})
    # background pockets inside the silhouette that the input does not have
    outer_sil = _fill(sil)
    pockets = outer_sil & ~sil & ~cv2.dilate(analysis.hole_mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    pockets &= ~(lines > 0)
    n, _, stats, _ = cv2.connectedComponentsWithStats(pockets.astype(np.uint8), 4)
    min_area = max(40, int(0.002 * analysis.mask.sum()))
    extra = int(sum(stats[i, cv2.CC_STAT_AREA] >= min_area for i in range(1, n)))
    return found, extra, detail


def _fill(m: np.ndarray) -> np.ndarray:
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(m.shape, np.uint8)
    cv2.drawContours(out, cs, -1, 1, -1)
    return out > 0


def measure(glb_or_meshes, analysis, al: Alignment | None = None) -> dict:
    meshes = load_parts(glb_or_meshes) if isinstance(glb_or_meshes, str) else glb_or_meshes
    mask = analysis.mask
    if al is None:
        al, _ = align(meshes, mask)
    sil = silhouette(meshes, al, mask.shape)
    iou = float((sil & mask).sum() / max((sil | mask).sum(), 1))
    thin = analysis.thin if analysis.thin is not None else np.zeros_like(mask)
    if thin.sum() >= 10:
        cover = cv2.dilate(sil.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        recall = float((cover & thin).sum() / thin.sum())
    else:
        recall = 1.0
    found, extra, detail = _hole_status(sil, analysis)
    exp = analysis.hole_count
    holes = (found / exp if exp else 1.0) - 0.25 * extra
    holes = float(np.clip(holes, 0, 1))
    score = W_IOU * iou + W_THIN * recall + W_HOLES * holes
    lost = []
    if exp and found < exp:
        lost.append(f"{exp - found} of {exp} through-hole(s) filled in")
    if extra:
        lost.append(f"{extra} extra hole(s) not in the input")
    if recall < 0.85:
        lost.append(f"thin parts only {recall:.0%} reproduced")
    return {"accuracy": round(score, 3), "iou": round(iou, 3), "thin_recall": round(recall, 3),
            "holes_expected": exp, "holes_found": found + extra, "holes_open": found, "holes_extra": extra,
            "hole_match": round(holes, 3), "holes": detail, "view_yaw": al.yaw, "lost": lost,
            "_alignment": al, "_silhouette": sil}


def report_image(analysis, sil: np.ndarray) -> np.ndarray:
    """Overlay for the UI: green = matched, red = input not covered, blue = extra model area."""
    m = analysis.mask
    img = np.full(m.shape + (3,), 255, np.uint8)
    img[m & sil] = (60, 200, 120)
    img[m & ~sil] = (230, 50, 60)
    img[~m & sil] = (70, 120, 230)
    return img
