"""Stage 4: foreground isolation with rembg (U^2-Net family, runs locally on CPU)."""
from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)
_session = None


def _get_session():
    global _session
    if _session is None:
        from rembg import new_session
        _session = new_session("isnet-general-use")
    return _session


def _fallback_alpha(rgb: np.ndarray) -> np.ndarray:
    """If rembg is unavailable: treat near-white, low-saturation pixels as background."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    fg = ~((hsv[..., 2] > 235) & (hsv[..., 1] < 25))
    fg = cv2.morphologyEx(fg.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return fg


def remove_bg(img) -> np.ndarray:
    """Return RGBA uint8 with background alpha=0, object centred with a 10% margin."""
    rgb = np.array(Image.fromarray(np.asarray(img)).convert("RGB"))
    try:
        from rembg import remove
        rgba = np.array(remove(Image.fromarray(rgb), session=_get_session()))
    except Exception as e:  # missing model download, onnx issue, ...
        log.warning("rembg failed (%s); using white-background threshold", e)
        rgba = np.dstack([rgb, _fallback_alpha(rgb)])
    return recentre(rgba)


def recentre(rgba: np.ndarray, margin: float = 0.06) -> np.ndarray:  # object fills ~88%
    a = rgba[..., 3]
    ys, xs = np.nonzero(a > 127)
    if len(xs) == 0:
        return rgba
    crop = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape[:2]
    side = int(max(h, w) / (1 - 2 * margin))
    out = np.zeros((side, side, 4), np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    out[y0:y0 + h, x0:x0 + w] = crop
    return cv2.resize(out, (512, 512), interpolation=cv2.INTER_AREA)


def rgba_on_white(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3:4].astype(np.float32) / 255
    return (rgba[..., :3] * a + 255 * (1 - a)).astype(np.uint8)


def merge_holes(rgba: np.ndarray, analysis) -> np.ndarray:
    """rembg happily fills a handle loop or a racket head; the analysed sketch knows those are
    through-holes. Map the sketch's hole mask onto the object's box and keep it transparent
    (strings drawn across a hole stay)."""
    import cv2
    if analysis is None or not analysis.holes:
        return rgba
    alpha = rgba[..., 3] > 127
    ys, xs = np.nonzero(alpha)
    oy, ox = np.nonzero(analysis.outer)
    if len(xs) < 20 or len(ox) < 20:
        return rgba
    hole = analysis.hole_mask.copy()
    if analysis.families:
        lines = np.zeros(hole.shape, np.uint8)
        for f in analysis.families:
            for x0, y0, x1, y1 in f["lines"]:
                cv2.line(lines, (int(x0), int(y0)), (int(x1), int(y1)), 1, max(2, int(analysis.stroke_px)))
        hole &= lines == 0
    crop = hole[oy.min():oy.max() + 1, ox.min():ox.max() + 1].astype(np.uint8)
    h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    hm = cv2.resize(crop, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    out = rgba.copy()
    region = out[ys.min():ys.max() + 1, xs.min():xs.max() + 1, 3]
    region[hm] = 0
    return out


def is_clean_illustration(img) -> bool:
    """Flat colours on a white/transparent background with sharp edges (a clip-art style
    render, not a line sketch or a photo): safe to reconstruct directly, no diffusion redraw."""
    import cv2
    a = np.asarray(img)
    if a.ndim != 3:
        return False
    rgb = a[..., :3].astype(np.int16)
    bg = (a[..., 3] < 20) if a.shape[-1] == 4 else np.zeros(a.shape[:2], bool)
    bg |= (rgb.min(-1) > 235)
    border = np.concatenate([bg[0], bg[-1], bg[:, 0], bg[:, -1]])
    if border.mean() < 0.9:
        return False
    fg = ~bg
    ys, xs = np.nonzero(fg)
    if len(xs) < 100:
        return False
    fill = fg.sum() / max((np.ptp(xs) + 1) * (np.ptp(ys) + 1), 1)
    if fill < 0.25:  # a line drawing: mostly empty inside its box
        return False
    q = (rgb[fg] // 24).astype(np.int32)
    keys = q[:, 0] * 10000 + q[:, 1] * 100 + q[:, 2]
    _, counts = np.unique(keys, return_counts=True)
    top = np.sort(counts)[::-1][:12].sum() / counts.sum()
    gray = cv2.cvtColor(a[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    g = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    strong, mid = (g > 120).sum(), ((g > 25) & (g <= 120)).sum()
    return bool(top > 0.85 and strong >= 0.6 * mid)
