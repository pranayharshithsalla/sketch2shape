"""Stage 5: auto-ranking of candidate images before the expensive 3D step.

total = 0.5 * sketch match   (silhouette IoU + edge chamfer vs the input sketch, see fidelity.py)
      + 0.3 * 3D-readiness   (one connected object, centred, not cropped, clean background)
      + 0.2 * prompt match   (CLIP image-text similarity; skipped + renormalised with no prompt)
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from .fidelity import fidelity

log = logging.getLogger(__name__)

W_SKETCH, W_READY, W_PROMPT = 0.5, 0.3, 0.2
# when the sketch has through-holes or thin parts, structure match carries high weight
W_STRUCT = {"sketch": 0.35, "structure": 0.35, "ready": 0.2, "prompt": 0.1}


def structure_match(rgba: np.ndarray, analysis) -> float | None:
    """Does the candidate keep the sketch's holes and thin parts? Compares the open (hole) area
    fraction inside the outline and the coverage of the sketch's thin skeleton, both in the
    candidate's own box. None when the sketch has neither."""
    if analysis is None or (not analysis.holes and (analysis.thin is None or analysis.thin.sum() < 10)):
        return None
    alpha = rgba[..., 3] > 127
    ys, xs = np.nonzero(alpha)
    oy, ox = np.nonzero(analysis.outer)
    if len(xs) < 20 or len(ox) < 20:
        return 0.0
    cs, _ = cv2.findContours(alpha.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(alpha.shape, np.uint8)
    cv2.drawContours(filled, cs, -1, 1, -1)
    r_c = float((filled.astype(bool) & ~alpha).sum() / max(filled.sum(), 1))
    r_e = float((analysis.outer & ~analysis.mask).sum() / max(analysis.outer.sum(), 1))
    hole = 1.0 - min(abs(r_c - r_e) / max(r_e, 0.05), 1.0)
    thin = 1.0
    if analysis.thin is not None and analysis.thin.sum() >= 10:
        crop = analysis.thin[oy.min():oy.max() + 1, ox.min():ox.max() + 1].astype(np.uint8)
        h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        t = cv2.resize(crop, (w, h), interpolation=cv2.INTER_NEAREST) > 0
        cover = cv2.dilate(alpha[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8),
                           np.ones((5, 5), np.uint8)) > 0
        thin = float((t & cover).sum() / max(t.sum(), 1))
    return round(0.5 * hole + 0.5 * thin, 3)


def readiness(rgba: np.ndarray) -> dict:
    """How well a background-removed candidate will reconstruct: 0-1 plus its parts."""
    alpha = (rgba[..., 3] > 127).astype(np.uint8)
    h, w = alpha.shape
    area = alpha.sum()
    if area < 0.005 * alpha.size:
        return {"ready": 0.0, "single": 0.0, "centred": 0.0, "uncropped": 0.0, "clean": 0.0}
    n, lab, stats, _ = cv2.connectedComponentsWithStats(alpha, 8)
    biggest = stats[1:, cv2.CC_STAT_AREA].max()
    single = float(biggest / area)                       # 1.0 = all foreground in one piece
    ys, xs = np.nonzero(alpha)
    cx, cy = xs.mean() / w, ys.mean() / h
    centred = float(max(0.0, 1 - 2.5 * np.hypot(cx - 0.5, cy - 0.5)))
    edge = np.concatenate([alpha[0], alpha[-1], alpha[:, 0], alpha[:, -1]])
    uncropped = float(1 - min(edge.mean() * 8, 1))       # object touching the frame = cut off
    # soft alpha (semi-transparent fringe / leftover background) is unclean
    soft = ((rgba[..., 3] > 20) & (rgba[..., 3] < 235)).sum() / max(area, 1)
    clean = float(max(0.0, 1 - soft * 4))
    ready = 0.35 * single + 0.2 * centred + 0.3 * uncropped + 0.15 * clean
    return {k: round(v, 3) for k, v in
            dict(ready=ready, single=single, centred=centred, uncropped=uncropped, clean=clean).items()}


def prompt_match(images_rgb: list[np.ndarray], prompt: str) -> list[float] | None:
    """CLIP similarity of each image to the prompt, rescaled to 0-1 (None if CLIP is unavailable)."""
    if not prompt.strip():
        return None
    try:
        import torch
        from PIL import Image
        from .caption_local import _image_embedding, _load
        model, proc, _ = _load()
        tok = proc.tokenizer([prompt[:300]], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            t = model.text_projection(model.text_model(**tok).pooler_output)
            t = t / t.norm(dim=-1, keepdim=True)
            sims = [float((_image_embedding(model, proc, Image.fromarray(im)) @ t.T)[0, 0]) for im in images_rgb]
        # raw CLIP cosine for a matching photo is ~0.25-0.35, unrelated ~0.1-0.18
        return [round(float(np.clip((s - 0.15) / 0.17, 0, 1)), 3) for s in sims]
    except Exception as e:
        log.warning("CLIP prompt match unavailable: %s", e)
        return None


def rank(sketch_mask: np.ndarray, sketch_strokes: np.ndarray, candidates_rgba: list[np.ndarray],
         prompt: str = "", analysis=None) -> tuple[int, list[dict]]:
    """-> (index of best, per-candidate scores). Each score dict keeps the old `score` key
    (= total) so callers that only want one number still work."""
    from .segment import rgba_on_white
    pm = prompt_match([rgba_on_white(c) for c in candidates_rgba], prompt)
    out = []
    for i, c in enumerate(candidates_rgba):
        f = fidelity(sketch_mask, sketch_strokes, c)
        r = readiness(c)
        if pm is None:
            total = (W_SKETCH * f["score"] + W_READY * r["ready"]) / (W_SKETCH + W_READY)
        else:
            total = W_SKETCH * f["score"] + W_READY * r["ready"] + W_PROMPT * pm[i]
        st = structure_match(c, analysis)
        if st is not None:
            w = dict(W_STRUCT)
            if pm is None:
                w.pop("prompt")
            parts = {"sketch": f["score"], "structure": st, "ready": r["ready"],
                     "prompt": pm[i] if pm is not None else 0.0}
            total = sum(w[k] * parts[k] for k in w) / sum(w.values())
            if st < 0.4:  # lost the handle / filled the frame: reject
                total *= 0.5
        if f["score"] == 0.0:
            total = 0.0  # the object was lost entirely
        out.append({"score": round(float(total), 3), "sketch": f["score"], "iou": f["iou"], "edge": f["edge"],
                    "ready": r["ready"], "prompt": None if pm is None else pm[i], "readiness": r,
                    "structure": st})
    best = int(np.argmax([s["score"] for s in out])) if out else -1
    return best, out


def card_label(i: int, s: dict, best: int) -> str:
    """Short card caption: rank badge, total, then sketch / 3D-ready / prompt components."""
    p = "–" if s["prompt"] is None else f"{s['prompt']:.2f}"
    st = "" if s.get("structure") is None else f" · structure {s['structure']:.2f}"
    return (f"{'★ BEST ' if i == best else ''}#{i + 1} · {s['score']:.2f} | "
            f"sketch {s['sketch']:.2f}{st} · ready {s['ready']:.2f} · prompt {p}")
