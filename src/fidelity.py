"""Stage 5: Sketch Fidelity Score and best-of-N selection.

score = 0.6 * silhouette IoU + 0.4 * edge agreement (chamfer-based), computed after
cropping both to their bounding boxes so scale/offset differences do not count.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import FIDELITY_CHAMFER_DIV, FIDELITY_W_EDGE, FIDELITY_W_SIL

_S = 256


def _crop_resize(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    c = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8) * 255
    h, w = c.shape
    side = max(h, w)
    sq = np.zeros((side, side), np.uint8)
    sq[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = c
    return cv2.resize(sq, (_S, _S), interpolation=cv2.INTER_NEAREST) > 127


def _chamfer(a_edges: np.ndarray, b_edges: np.ndarray) -> float:
    """Symmetric mean distance (px at 256 res) between two edge maps."""
    if a_edges.sum() == 0 or b_edges.sum() == 0:
        return 1e3
    da = cv2.distanceTransform((~a_edges).astype(np.uint8), cv2.DIST_L2, 3)
    db = cv2.distanceTransform((~b_edges).astype(np.uint8), cv2.DIST_L2, 3)
    return 0.5 * (db[a_edges].mean() + da[b_edges].mean())


def fidelity(sketch_mask: np.ndarray, sketch_strokes: np.ndarray, cand_rgba: np.ndarray) -> dict:
    """sketch_mask: filled silhouette (bool); sketch_strokes: stroke map (uint8, white=stroke);
    cand_rgba: background-removed candidate."""
    alpha = cand_rgba[..., 3] > 127
    if alpha.sum() < 0.005 * alpha.size:
        return {"score": 0.0, "iou": 0.0, "edge": 0.0}
    s, g = _crop_resize(sketch_mask), _crop_resize(alpha)
    iou = float((s & g).sum() / max((s | g).sum(), 1))

    # Edge term: sketch strokes vs candidate's Canny edges, in the same crop frame
    ys, xs = np.nonzero(alpha)
    gray = cv2.cvtColor(cand_rgba[..., :3], cv2.COLOR_RGB2GRAY)
    gray = np.where(alpha, gray, 255).astype(np.uint8)
    crop = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    cand_edges = _crop_resize_gray_edges(crop)
    ys2, xs2 = np.nonzero(sketch_mask)
    sk = sketch_strokes[ys2.min():ys2.max() + 1, xs2.min():xs2.max() + 1] > 127
    sk_edges = _crop_resize_bool(sk)
    edge = 1.0 - min(_chamfer(sk_edges, cand_edges) / FIDELITY_CHAMFER_DIV, 1.0)
    score = FIDELITY_W_SIL * iou + FIDELITY_W_EDGE * edge
    return {"score": round(float(score), 3), "iou": round(iou, 3), "edge": round(float(edge), 3)}


def _square(img: np.ndarray, fill) -> np.ndarray:
    h, w = img.shape[:2]
    side = max(h, w)
    sq = np.full((side, side), fill, img.dtype)
    sq[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = img
    return sq


def _crop_resize_gray_edges(gray_crop: np.ndarray) -> np.ndarray:
    sq = cv2.resize(_square(gray_crop, 255), (_S, _S), interpolation=cv2.INTER_AREA)
    sq = cv2.GaussianBlur(sq, (3, 3), 0)
    return cv2.Canny(sq, 50, 150) > 0


def _crop_resize_bool(b: np.ndarray) -> np.ndarray:
    sq = _square(b.astype(np.uint8) * 255, 0)
    return cv2.resize(sq, (_S, _S), interpolation=cv2.INTER_AREA) > 40


def best_of(sketch_mask, sketch_strokes, candidates_rgba: list[np.ndarray]) -> tuple[int, list[dict]]:
    scores = [fidelity(sketch_mask, sketch_strokes, c) for c in candidates_rgba]
    best = int(np.argmax([s["score"] for s in scores])) if scores else -1
    return best, scores
