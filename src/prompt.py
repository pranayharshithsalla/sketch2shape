"""Prompt understanding for the Sketch-CAD path: colours (any of the 148 CSS colour names,
multi-word names, light/dark/pale/deep variants, #hex), materials, and per-part colours
("a red mug with a black handle and gold rim", "cap in silver").

Everything here is deterministic and offline; the geometry still comes from the sketch.
"""
from __future__ import annotations

import re

from PIL import ImageColor

# Hand-tuned versions of common colours (CSS "green" is very dark, "red" very saturated)
_BASE = {"red": (200, 45, 40), "blue": (45, 95, 205), "green": (55, 150, 70), "yellow": (235, 200, 50),
         "orange": (235, 125, 35), "purple": (130, 65, 175), "pink": (235, 130, 175), "black": (35, 35, 40),
         "white": (240, 240, 236), "grey": (140, 140, 145), "gray": (140, 140, 145), "brown": (125, 80, 45),
         "violet": (140, 80, 210), "lavender": (180, 160, 225), "cream": (245, 235, 205), "mint": (160, 225, 190),
         "peach": (250, 190, 150), "rose": (230, 120, 150), "amber": (240, 170, 30), "sky": (120, 185, 235),
         "charcoal": (55, 58, 62), "scarlet": (220, 40, 30), "emerald": (40, 160, 100), "ruby": (170, 20, 60),
         "sapphire": (30, 70, 170), "jade": (60, 160, 120), "mustard": (215, 170, 40), "burgundy": (120, 20, 45),
         "rust": (170, 80, 40), "sand": (220, 195, 150), "copper": (190, 110, 70), "bronze": (176, 120, 60),
         "gold": (222, 180, 70), "golden": (222, 180, 70), "silver": (205, 207, 212)}
_CSS = {k: ImageColor.getrgb(v)[:3] for k, v in ImageColor.colormap.items()}
_SHADES = {"dark": 0.6, "deep": 0.7, "light": 1.35, "pale": 1.5, "bright": 1.15, "pastel": 1.45, "neon": 1.2}

# name -> (rgb, metallic, roughness)
MATERIALS = {
    "wood": ((150, 100, 60), 0.0, 0.7), "metal": ((185, 188, 195), 0.95, 0.3),
    "steel": ((185, 188, 195), 0.95, 0.3), "gold": ((222, 180, 70), 1.0, 0.25),
    "silver": ((205, 207, 212), 1.0, 0.2), "copper": ((190, 110, 70), 1.0, 0.3),
    "bronze": ((176, 120, 60), 1.0, 0.35), "brass": ((200, 165, 80), 1.0, 0.3),
    "chrome": ((220, 222, 228), 1.0, 0.08), "aluminium": ((200, 203, 208), 0.9, 0.35),
    "iron": ((90, 92, 96), 0.9, 0.55), "ceramic": ((236, 232, 222), 0.0, 0.15),
    "porcelain": ((245, 243, 238), 0.0, 0.1), "plastic": ((90, 140, 210), 0.0, 0.45),
    "rubber": ((40, 40, 42), 0.0, 0.95), "glass": ((200, 225, 235), 0.0, 0.05),
    "crystal": ((225, 238, 245), 0.0, 0.02), "fabric": ((170, 120, 120), 0.0, 0.95),
    "leather": ((110, 65, 40), 0.0, 0.6), "stone": ((140, 138, 132), 0.0, 0.9),
    "marble": ((230, 228, 222), 0.0, 0.25), "clay": ((185, 110, 75), 0.0, 0.85),
    "terracotta": ((195, 100, 65), 0.0, 0.85), "concrete": ((160, 160, 155), 0.0, 0.95),
    "jade": ((60, 160, 120), 0.0, 0.2), "ivory": ((245, 240, 225), 0.0, 0.3),
}
ALIASES = {"wooden": "wood", "golden": "gold", "metallic": "metal", "aluminum": "aluminium",
           "stainless": "steel", "clay": "clay", "pottery": "clay", "stoneware": "ceramic"}
# Specific materials win over generic ones: "gold metal" -> gold
_GENERIC = {"metal"}

PARTS = {"handle": "handle", "holder": "handle", "grip": "handle", "cap": "cap", "lid": "cap",
         "top": "cap", "cork": "cap", "stopper": "cap", "neck": "cap", "body": "body", "base": "base",
         "bottom": "base", "foot": "base", "stand": "base", "rim": "rim", "spout": "spout",
         "nozzle": "spout"}

DEFAULT_RGB = (95, 140, 200)


def _tokens(text: str) -> list[str]:
    t = (text or "").lower().replace("-", " ")
    t = re.sub(r"\bcolou?red\b|\bcolou?r\b|\bshade of\b|\btinted\b", " ", t)
    return re.findall(r"#[0-9a-f]{6}|#[0-9a-f]{3}|[a-z]+", t)


def _shade(rgb, f):
    if f < 1:
        return tuple(int(c * f) for c in rgb)
    return tuple(int(c + (255 - c) * (f - 1) / f) for c in rgb)


def find_colours(tokens: list[str]) -> list[tuple[int, int, tuple]]:
    """[(start_idx, end_idx, rgb)] for colour mentions, longest match first."""
    out, i = [], 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("#"):
            out.append((i, i + 1, ImageColor.getrgb(tok)[:3]))
            i += 1
            continue
        hit = None
        for n in (3, 2, 1):  # "light sea green", "sky blue", "red"
            if i + n <= len(tokens):
                name = "".join(tokens[i:i + n])
                if n == 1 and name in _BASE:
                    hit = (i, i + 1, _BASE[name])
                elif name in _CSS and name not in ("tan", "linen", "snow", "oldlace") or (
                        n == 1 and name in ("tan",) and i + 1 < len(tokens)):
                    hit = (i, i + n, _CSS[name])
                if hit:
                    break
        if hit and i > 0 and tokens[i - 1] in _SHADES and not (hit[1] - hit[0] > 1):
            hit = (i - 1, hit[1], _shade(hit[2], _SHADES[tokens[i - 1]]))
        if hit:
            out.append(hit)
            i = hit[1]
        else:
            i += 1
    return out


def _material_at(tokens, j):
    tok = ALIASES.get(tokens[j], tokens[j])
    return tok if tok in MATERIALS else None


def parse(text: str, preset: str = "auto") -> dict:
    """-> {"body": (rgb, metal, rough), "parts": {part: (rgb, metal, rough)}, "summary": str}"""
    tokens = _tokens(text)
    colours = find_colours(tokens)
    mats = [(j, _material_at(tokens, j)) for j in range(len(tokens)) if _material_at(tokens, j)]
    colour_idx = {k for s, e, _ in colours for k in range(s, e)}

    def part_after(end, start):
        """Part noun governed by a colour/material word: 'black handle', 'handle in black'."""
        k = end
        while k < len(tokens) and (_material_at(tokens, k) or tokens[k] in _SHADES or k in colour_idx
                                   or tokens[k] in ("coloured", "colored")):
            k += 1  # skip stacked adjectives: "black rubber handle", "navy blue"
        if k < len(tokens) and tokens[k] in PARTS:
            return PARTS[tokens[k]]
        for k in range(start - 1, max(start - 4, -1), -1):  # "handle in black", "cap of gold"
            if tokens[k] in ("in", "of", "is", "made"):
                continue
            if tokens[k] in PARTS:
                return PARTS[tokens[k]]
            break
        return None

    body_rgb, body_mat = None, None
    parts: dict[str, dict] = {}
    for s, e, rgb in colours:
        p = part_after(e, s)
        if p and p != "body":
            parts.setdefault(p, {}).setdefault("rgb", rgb)  # first word wins: "navy blue handle"
        elif body_rgb is None:
            body_rgb = rgb
    for j, m in sorted(mats, key=lambda x: x[1] in _GENERIC):
        p = part_after(j + 1, j)
        if p and p != "body":
            parts.setdefault(p, {}).setdefault("mat", m)
        elif body_mat is None:
            body_mat = m
    if preset in MATERIALS:
        body_mat = preset

    def resolve(rgb, mat):
        base_rgb, metal, rough = MATERIALS.get(mat, (DEFAULT_RGB, 0.0, 0.5))
        return (rgb or base_rgb), metal, rough

    body = resolve(body_rgb, body_mat)
    res_parts = {p: resolve(d.get("rgb") or (None if d.get("mat") else body[0]), d.get("mat") or body_mat)
                 for p, d in parts.items()}
    bits = [f"body {_name(body)}"] + [f"{p} {_name(v)}" for p, v in res_parts.items()]
    return {"body": body, "parts": res_parts, "summary": "; ".join(bits)}


def _name(v) -> str:
    (r, g, b), metal, rough = v
    finish = "metallic" if metal > 0.5 else ("glossy" if rough < 0.2 else "matte" if rough > 0.8 else "satin")
    return f"#{r:02x}{g:02x}{b:02x} {finish}"
