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


def recentre(rgba: np.ndarray, margin: float = 0.10) -> np.ndarray:
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
