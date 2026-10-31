"""Prompt-only creation: draw a clean template sketch for the object named in the prompt,
then run the normal pipeline on it, so colours, per-part materials and shape words apply.

"a brown bottle with a green holder" -> bottle template + handle -> revolve -> brown body,
green handle. Offline and instant. Lathe objects are half-profiles (y up from the base,
radius), both as fractions of the drawing height; flat objects are outlines.
"""
from __future__ import annotations

import re

import cv2
import numpy as np

SIZE = 512
INK = (25, 25, 25)

# name -> (profile [(y, r)], height_px, default_handle, spout)
LATHE = {
    "bottle": ([(0, .17), (.02, .19), (.55, .19), (.68, .12), (.74, .065), (.86, .065), (.86, .08), (1, .08)], 400, False, False),
    "mug": ([(0, .27), (1, .30)], 300, True, False),
    "cup": ([(0, .24), (1, .34)], 250, True, False),
    "tumbler": ([(0, .17), (1, .21)], 330, False, False),
    "vase": ([(0, .14), (.08, .2), (.35, .26), (.6, .18), (.8, .09), (.92, .1), (1, .14)], 400, False, False),
    "glass": ([(0, .18), (.02, .18), (.04, .03), (.45, .03), (.5, .12), (.7, .2), (1, .18)], 400, False, False),
    "goblet": ([(0, .2), (.03, .2), (.06, .04), (.4, .04), (.45, .14), (.65, .24), (1, .26)], 380, False, False),
    "bowl": ([(0, .38), (.15, .6), (.6, .88), (1, .98)], 170, False, False),
    "jar": ([(0, .3), (.75, .32), (.82, .27), (.86, .29), (1, .29)], 330, False, False),
    "pot": ([(0, .24), (.88, .33), (.88, .37), (1, .37)], 300, False, False),
    "lamp": ([(0, .28), (.04, .28), (.05, .04), (.55, .04), (.56, .32), (1, .16)], 400, False, False),
    "candle": ([(0, .12), (.93, .12), (.96, .03), (1, .015)], 380, False, False),
    "teapot": ([(0, .3), (.4, .48), (.75, .38), (.8, .2), (.88, .2), (.95, .06), (1, .06)], 280, True, True),
    "kettle": ([(0, .38), (.6, .38), (.8, .3), (.85, .12), (.95, .12), (1, .03)], 300, True, True),
    "jug": ([(0, .25), (.5, .3), (.85, .22), (1, .26)], 340, True, False),
    "pitcher": ([(0, .25), (.5, .3), (.85, .22), (1, .26)], 340, True, False),
    "flask": ([(0, .3), (.45, .3), (.7, .08), (.95, .08), (1, .09)], 380, False, False),
    "bucket": ([(0, .3), (1, .4)], 300, False, False),
    "barrel": ([(0, .3), (.5, .37), (1, .3)], 340, False, False),
    "trophy": ([(0, .3), (.1, .3), (.12, .12), (.2, .05), (.45, .05), (.5, .12), (.65, .28), (1, .33)], 400, True, False),
    "pawn": ([(0, .28), (.08, .28), (.12, .2), (.18, .14), (.55, .08), (.6, .15), (.64, .08), (.72, .12), (.86, .14), (.95, .08), (1, .0)], 400, False, False),
    "column": ([(0, .2), (.06, .2), (.07, .15), (.93, .15), (.94, .2), (1, .2)], 420, False, False),
    "can": ([(0, .2), (.02, .22), (.98, .22), (1, .2)], 330, False, False),
    "spinning top": ([(0, .0), (.35, .3), (.5, .34), (.6, .3), (.65, .06), (1, .05)], 320, False, False),
}
ALIASES = {"bottles": "bottle", "mugs": "mug", "cups": "cup", "teacup": "cup", "vases": "vase",
           "wineglass": "glass", "chalice": "goblet", "jars": "jar", "flowerpot": "pot", "planter": "pot",
           "lantern": "lamp", "pillar": "column", "tower": "column", "chess": "pawn", "beaker": "tumbler",
           "container": "jar", "urn": "vase", "carafe": "jug", "canister": "can", "top": None}
FLAT = ("star", "heart", "gear", "coin", "medal", "plate", "shield", "badge", "token")


# organic shapes get Chaikin-rounded profiles; the rest keep crisp steps (caps, rims, plinths)
ROUNDED = {"vase", "barrel", "jug", "pitcher", "teapot", "kettle", "bowl", "flask", "spinning top",
           "pawn", "trophy", "goblet", "glass"}


def _chaikin(profile, iters=2):
    p = np.array(profile, float)
    for _ in range(iters):
        q = [p[0]]
        for a, b in zip(p[:-1], p[1:]):
            q += [0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b]
        q.append(p[-1])
        p = np.array(q)
    return [tuple(x) for x in p]


def _lathe_points(profile, h_px):
    cx, yb = SIZE // 2, SIZE // 2 + h_px // 2
    right = [(cx + r * h_px, yb - y * h_px) for y, r in profile]
    left = [(cx - r * h_px, yb - y * h_px) for y, r in reversed(profile)]
    return np.int32(right + left), cx, yb


def _draw_lathe(name, words: set[str], text: str):
    profile, h_px, handle, spout = LATHE[name]
    if name in ROUNDED:
        profile = _chaikin(profile)
    img = np.full((SIZE, SIZE, 3), 255, np.uint8)
    pts, cx, yb = _lathe_points(profile, h_px)
    cv2.polylines(img, [pts], True, INK, 4, cv2.LINE_AA)
    body = np.zeros((SIZE, SIZE), np.uint8)
    cv2.fillPoly(body, [pts], 1)
    parts = []  # exact attachment masks, same pixel frame as `body`
    if re.search(r"\b(no|without)\s+(a\s+)?(handle|holder)", text):
        handle = False
    elif words & {"handle", "holder", "grip"}:
        handle = True
    ys = [y for y, _ in profile]
    rr = lambda y: float(np.interp(y, ys, [r for _, r in profile]))
    if handle:  # C-shaped curve on the right: leaves and rejoins the body wall at its edge
        y0, y1 = {"trophy": (.55, .95), "teapot": (.3, .72), "kettle": (.45, .85)}.get(name, (.25, .8))
        bulge = max(rr(y) for y in np.linspace(y0, y1, 20)) + 0.45  # apex ~0.22 h past the widest wall
        t = np.linspace(0, 1, 40)[:, None]
        p0 = np.array([cx + rr(y0) * h_px - 3, yb - y0 * h_px])
        p2 = np.array([cx + rr(y1) * h_px - 3, yb - y1 * h_px])
        p1 = np.array([cx + bulge * h_px, yb - (y0 + y1) / 2 * h_px])
        curve = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
        cv2.polylines(img, [np.int32(curve)], False, INK, 6, cv2.LINE_AA)
        m = np.zeros((SIZE, SIZE), np.uint8)
        cv2.polylines(m, [np.int32(curve)], False, 1, max(8, int(0.045 * h_px)), cv2.LINE_AA)
        parts.append(("handle", m > 0))
    if spout:  # spout on the left, starting inside the body wall and rising outwards
        y0 = .35
        p0 = (int(cx - (rr(y0) - .04) * h_px), int(yb - y0 * h_px))
        p1 = (int(cx - (rr(y0) + .2) * h_px), int(yb - .75 * h_px))
        cv2.line(img, p0, p1, INK, 12, cv2.LINE_AA)
        m = np.zeros((SIZE, SIZE), np.uint8)
        cv2.line(m, p0, p1, 1, max(12, int(0.06 * h_px)), cv2.LINE_AA)
        parts.append(("spout", m > 0))
    return img, body > 0, parts


def _draw_flat(name):
    img = np.full((SIZE, SIZE, 3), 255, np.uint8)
    c = SIZE // 2
    if name == "star":
        pts = [(c + (190 if i % 2 == 0 else 78) * np.cos(np.pi / 2 + i * np.pi / 5),
                c + 10 - (190 if i % 2 == 0 else 78) * np.sin(np.pi / 2 + i * np.pi / 5)) for i in range(10)]
        cv2.polylines(img, [np.int32(pts)], True, INK, 4, cv2.LINE_AA)
    elif name == "heart":
        t = np.linspace(0, 2 * np.pi, 200)
        x, y = 16 * np.sin(t) ** 3, 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        cv2.polylines(img, [np.int32(np.c_[c + x * 11, c - 20 - y * 11])], True, INK, 4, cv2.LINE_AA)
    elif name == "gear":
        a = np.linspace(0, 2 * np.pi, 361)
        r = 165 + 30 * (np.sin(a * 12) > 0)
        cv2.polylines(img, [np.int32(np.c_[c + r * np.cos(a), c + r * np.sin(a)])], True, INK, 4, cv2.LINE_AA)
        cv2.circle(img, (c, c), 55, INK, 4, cv2.LINE_AA)
    elif name == "shield":
        pts = [(c - 150, c - 170), (c + 150, c - 170), (c + 150, c), (c, c + 190), (c - 150, c)]
        cv2.polylines(img, [np.int32(pts)], True, INK, 4, cv2.LINE_AA)
    else:  # coin, medal, plate, badge, token: a disc
        cv2.circle(img, (c, c), 180, INK, 4, cv2.LINE_AA)
    from .preprocess import fill_outline
    body = fill_outline(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    if name == "gear":  # keep the axle hole
        hole = np.zeros_like(body, np.uint8)
        cv2.circle(hole, (c, c), 53, 1, -1)
        body &= hole == 0
    return img, body, []


def find_template(text: str) -> str | None:
    t = (text or "").lower()
    if "spinning top" in t:
        return "spinning top"
    if "wine glass" in t:
        return "glass"
    hits = [ALIASES.get(w, w) for w in re.findall(r"[a-z]+", t)]
    hits = [w for w in hits if w in LATHE or w in FLAT]
    if len(hits) > 1 and "glass" in hits:  # "a glass vase": glass is the material
        hits = [w for w in hits if w != "glass"]
    return hits[0] if hits else None  # first object word in the prompt wins


def available_names() -> list[str]:
    return sorted(set(LATHE) | set(FLAT))


def make_template(text: str) -> dict | None:
    """-> {name, img (RGB sketch), body (bool mask), parts [(name, bool mask)]} or None.
    Body and parts are exact geometry in one pixel frame, so the 3D build needs no detection."""
    name = find_template(text)
    if name is None:
        return None
    words = set(re.findall(r"[a-z]+", text.lower()))
    img, body, parts = _draw_lathe(name, words, text.lower()) if name in LATHE else _draw_flat(name)
    return {"name": name, "img": img, "body": body, "parts": parts}


def make_sketch(text: str) -> tuple[np.ndarray, str] | None:
    """-> (RGB sketch image, template name), or None if no template matches the prompt."""
    t = make_template(text)
    return (t["img"], t["name"]) if t else None
