"""Draws the hard-case sketch set (holes, loops, thin structures) into samples/hard_cases/.
Each case lists the prompt and what the analysis must find (through-holes, string grid).
Replace any PNG with a real drawing/photo of the same object (keep the file name).

    python scripts/make_hard_cases.py
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "samples" / "hard_cases"
K, T = (25, 25, 25), 4


def canvas():
    return np.full((600, 600, 3), 255, np.uint8)


def genie_lamp():  # oil lamp: squat body, long spout, looped handle (double outline), lid knob
    c = canvas()
    body = np.array([[170, 330], [205, 285], [300, 268], [390, 285], [425, 330], [395, 372],
                     [300, 388], [205, 372]], np.int32)
    cv2.polylines(c, [body], True, K, T)
    cv2.polylines(c, [np.array([[400, 300], [470, 270], [540, 212], [552, 222], [480, 292], [418, 345]],
                               np.int32)], False, K, T)                     # spout
    cv2.ellipse(c, (300, 252), (40, 16), 0, 0, 360, K, T)                   # lid
    cv2.circle(c, (300, 226), 12, K, T)                                     # knob
    cv2.ellipse(c, (300, 410), (55, 14), 0, 0, 360, K, T)                   # foot
    cv2.line(c, (270, 388), (262, 400), K, T)
    cv2.line(c, (330, 388), (338, 400), K, T)
    cv2.ellipse(c, (190, 328), (72, 48), 0, 60, 300, K, T)                  # handle loop outer
    cv2.ellipse(c, (192, 328), (44, 26), 0, 50, 310, K, T)                  # handle loop inner
    return c


def racket():  # badminton racket: oval double-line frame, string grid, shaft, grip
    c = canvas()
    cv2.ellipse(c, (300, 190), (120, 160), 0, 0, 360, K, T)
    cv2.ellipse(c, (300, 190), (104, 144), 0, 0, 360, K, T)
    for x in range(220, 390, 22):                                          # vertical strings
        dy = int(144 * np.sqrt(max(0, 1 - ((x - 300) / 104) ** 2)))
        cv2.line(c, (x, 190 - dy + 2), (x, 190 + dy - 2), K, 2)
    for y in range(70, 320, 24):                                           # horizontal strings
        dx = int(104 * np.sqrt(max(0, 1 - ((y - 190) / 144) ** 2)))
        cv2.line(c, (300 - dx + 2, y), (300 + dx - 2, y), K, 2)
    cv2.line(c, (293, 350), (293, 470), K, T)                               # shaft
    cv2.line(c, (307, 350), (307, 470), K, T)
    cv2.rectangle(c, (286, 470), (314, 575), K, T)                          # grip
    return c


def mug():
    c = canvas()
    cv2.rectangle(c, (170, 160), (380, 460), K, T)
    cv2.ellipse(c, (360, 310), (95, 95), 0, -85, 85, K, T)
    cv2.ellipse(c, (362, 310), (58, 60), 0, -85, 85, K, T)
    return c


def teapot():
    c = canvas()
    cv2.ellipse(c, (300, 340), (140, 110), 0, 0, 360, K, T)                 # body
    cv2.ellipse(c, (300, 232), (60, 16), 0, 0, 360, K, T)                   # lid
    cv2.circle(c, (300, 205), 13, K, T)                                     # knob
    cv2.polylines(c, [np.array([[425, 330], [500, 280], [545, 215], [560, 225], [520, 300], [432, 375]],
                               np.int32)], False, K, T)                     # spout
    cv2.ellipse(c, (175, 335), (95, 75), 0, 75, 285, K, T)                  # handle outer
    cv2.ellipse(c, (178, 335), (62, 48), 0, 80, 280, K, T)                  # handle inner
    return c


def chair():  # open back: top rail + vertical slats (outlined) + seat + legs
    c = canvas()
    cv2.rectangle(c, (190, 70), (410, 110), K, T)                           # top rail
    for x in (190, 255, 320, 390):                                          # slats
        cv2.rectangle(c, (x, 110), (x + 20, 300), K, T)
    cv2.rectangle(c, (180, 300), (420, 335), K, T)                          # seat
    for x in (190, 395):
        cv2.rectangle(c, (x, 335), (x + 15, 540), K, T)                     # legs
    return c


def bicycle_wheel():
    c = canvas()
    cv2.circle(c, (300, 300), 230, K, T)
    cv2.circle(c, (300, 300), 212, K, T)
    cv2.circle(c, (300, 300), 22, K, T)
    for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        cv2.line(c, (int(300 + 22 * np.cos(a)), int(300 + 22 * np.sin(a))),
                 (int(300 + 212 * np.cos(a)), int(300 + 212 * np.sin(a))), K, 2)
    return c


def wire_basket():
    c = canvas()
    top, bot = (150, 450, 170), (190, 410, 470)
    cv2.line(c, (top[0], top[2]), (top[1], top[2]), K, T)
    cv2.line(c, (bot[0], bot[2]), (bot[1], bot[2]), K, T)
    cv2.line(c, (top[0], top[2]), (bot[0], bot[2]), K, T)
    cv2.line(c, (top[1], top[2]), (bot[1], bot[2]), K, T)
    for f in np.linspace(0.15, 0.85, 6):                                   # vertical wires
        cv2.line(c, (int(top[0] + f * (top[1] - top[0])), top[2]),
                 (int(bot[0] + f * (bot[1] - bot[0])), bot[2]), K, 2)
    for f in np.linspace(0.2, 0.8, 4):                                     # horizontal wires
        y = int(top[2] + f * (bot[2] - top[2]))
        x0 = int(top[0] + f * (bot[0] - top[0]))
        cv2.line(c, (x0, y), (600 - x0, y), K, 2)
    cv2.ellipse(c, (300, 170), (120, 110), 0, 180, 360, K, T)              # handle arc
    return c


def scissors():  # two blades + two finger loops (ring outlines)
    c = canvas()
    cv2.polylines(c, [np.array([[300, 300], [285, 60], [305, 60], [318, 300]], np.int32)], True, K, T)
    cv2.polylines(c, [np.array([[300, 300], [330, 70], [348, 78], [318, 300]], np.int32)], True, K, T)
    cv2.circle(c, (309, 300), 9, K, T)
    for cx in (240, 380):
        cv2.ellipse(c, (cx, 420), (62, 75), 0, 0, 360, K, T)
        cv2.ellipse(c, (cx, 420), (40, 52), 0, 0, 360, K, T)
    cv2.line(c, (300, 310), (262, 348), K, T)
    cv2.line(c, (318, 310), (358, 348), K, T)
    return c


CASES = {  # name: (drawing, prompt, expected through-holes, has string grid)
    "genie_lamp": (genie_lamp, "a gold genie oil lamp with a looped handle", 1, False),
    "racket": (racket, "a badminton racket with strings", 1, True),
    "mug": (mug, "a white mug with a handle", 1, False),
    "teapot": (teapot, "a teapot with a handle and spout", 1, False),
    "chair": (chair, "a wooden chair with an open slatted back", 3, False),
    "bicycle_wheel": (bicycle_wheel, "a bicycle wheel with spokes", 1, True),
    "wire_basket": (wire_basket, "a wire basket with a handle", 2, True),
    "scissors": (scissors, "a pair of scissors with finger loops", 2, False),
}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {}
    for name, (fn, prompt, holes, grid) in CASES.items():
        cv2.imwrite(str(OUT / f"{name}.png"), fn())
        meta[name] = {"prompt": prompt, "expected_holes": holes, "string_grid": grid}
    (OUT / "cases.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote", len(CASES), "hard cases to", OUT)
