"""Stage 1: turn any sketch input (Gradio sketchpad dict, upload, path, array) into
clean, normalised structural cues.

Outputs (all WORK_SIZE x WORK_SIZE):
  clean    - uint8 gray, black strokes on white
  scribble - uint8 gray, white strokes on black (ControlNet scribble convention)
  mask     - bool, filled silhouette of the outermost outline
  rgb      - uint8 RGB version of `clean` for providers that want colour input
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import WORK_SIZE


class PreprocessError(RuntimeError):
    pass


@dataclass
class Sketch:
    clean: np.ndarray
    scribble: np.ndarray
    mask: np.ndarray
    rgb: np.ndarray
    stroke_px: int


def _to_rgba(inp) -> np.ndarray:
    """Accept Gradio ImageEditor dicts, PIL, ndarray or a file path; return RGBA uint8."""
    if isinstance(inp, dict):  # gr.Sketchpad / gr.ImageEditor value
        img = inp.get("composite")
        if img is None:
            layers = [l for l in (inp.get("layers") or []) if l is not None]
            img = layers[-1] if layers else inp.get("background")
        inp = img
    if inp is None:
        raise PreprocessError("No sketch provided: draw on the canvas or upload an image.")
    if isinstance(inp, (str, Path)):
        inp = Image.open(inp)
    if isinstance(inp, Image.Image):
        inp = np.array(inp.convert("RGBA"))
    arr = np.asarray(inp)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, -1)
    if arr.shape[-1] == 3:
        arr = np.concatenate([arr, np.full(arr.shape[:2] + (1,), 255, arr.dtype)], -1)
    return arr.astype(np.uint8)


def composite_on_white(rgba: np.ndarray) -> np.ndarray:
    """Gradio's sketchpad returns transparent PNGs; naive RGB conversion turns the
    background black. Alpha-composite onto white first."""
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = rgba[..., :3].astype(np.float32) * a + 255.0 * (1 - a)
    return rgb.astype(np.uint8)


def _fill_outline(binary_strokes: np.ndarray) -> np.ndarray:
    """Filled silhouette from (possibly open) strokes: close small gaps, then fill
    external contours."""
    k = max(3, WORK_SIZE // 64) | 1
    closed = cv2.morphologyEx(binary_strokes, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(binary_strokes)
    side = binary_strokes.shape[0]
    boxes = [cv2.boundingRect(c) for c in contours]
    main = max((w * h for _, _, w, h in boxes), default=0)
    for c, (x, y, w, h) in zip(contours, boxes):
        if max(w, h) < 0.04 * side:  # specks
            continue
        part = np.zeros_like(binary_strokes)
        cv2.drawContours(part, [c], -1, 255, thickness=-1)
        stroke_area = cv2.countNonZero(cv2.bitwise_and(part, closed))
        # A small DETACHED open stroke (an arc cap drawn above a bottle) encloses almost
        # nothing: fill its convex hull instead of losing it. Never do this to the main
        # drawing: thick, detailed sketches (a trophy with handles) would become a blob.
        if w * h < 0.3 * main and cv2.countNonZero(part) < 2.5 * max(stroke_area, 1):
            cv2.drawContours(part, [cv2.convexHull(c)], -1, 255, thickness=-1)
        filled |= part
    if not filled.any() and contours:
        cv2.drawContours(filled, [cv2.convexHull(np.vstack(contours))], -1, 255, thickness=-1)
    return filled > 0


def fill_outline(gray_black_on_white: np.ndarray) -> np.ndarray:
    strokes = (gray_black_on_white < 128).astype(np.uint8) * 255
    return _fill_outline(strokes)


def preprocess(inp, denoise: bool = True) -> Sketch:
    rgb = composite_on_white(_to_rgba(inp))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Background handling: photos of paper are uneven -> divide by blurred background
    bg = cv2.medianBlur(gray, 31) if min(gray.shape) > 64 else gray
    norm = cv2.divide(gray, np.maximum(bg, 1), scale=255)
    # Invert dark-background sketches (chalkboard / white-on-black)
    if np.median(gray) < 110:
        norm = 255 - gray
    norm = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(norm)

    _, strokes = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if denoise:
        strokes = cv2.morphologyEx(strokes, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(strokes, 8)
        min_area = max(8, int(strokes.size * 2e-5))
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                strokes[lab == i] = 0
    ys, xs = np.nonzero(strokes)
    if len(xs) < 20:
        raise PreprocessError("Sketch looks empty: draw a few clear strokes and try again.")

    # Crop to content with 12% margin, pad to square, resize
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    side = int(max(x1 - x0, y1 - y0) * 1.24) + 4
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    canvas = np.zeros((side, side), np.uint8)
    sx0, sy0 = cx - side // 2, cy - side // 2
    src = strokes[max(sy0, 0):sy0 + side, max(sx0, 0):sx0 + side]
    canvas[max(-sy0, 0):max(-sy0, 0) + src.shape[0], max(-sx0, 0):max(-sx0, 0) + src.shape[1]] = src
    strokes = cv2.resize(canvas, (WORK_SIZE, WORK_SIZE), interpolation=cv2.INTER_AREA)
    strokes = (strokes > 60).astype(np.uint8) * 255

    # Normalise stroke width so thin pen lines and fat brush lines look alike downstream
    dist = cv2.distanceTransform(strokes, cv2.DIST_L2, 3)
    stroke_px = max(1, int(round(2 * np.median(dist[strokes > 0]))))
    if stroke_px < 3:
        strokes = cv2.dilate(strokes, np.ones((2, 2), np.uint8))

    clean = 255 - strokes
    mask = _fill_outline(strokes)
    return Sketch(clean=clean, scribble=strokes, mask=mask,
                  rgb=np.stack([clean] * 3, -1), stroke_px=stroke_px)
