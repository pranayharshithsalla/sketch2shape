"""Draws simple synthetic line sketches into examples/ so the pipeline can be tested
before hand-drawn sketches exist. Replace/extend with real hand-drawn scans."""
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "examples"
OUT.mkdir(exist_ok=True)
K, T = (30, 30, 30), 4  # ink colour, stroke thickness


def canvas():
    return np.full((600, 600, 3), 255, np.uint8)


def mug():
    c = canvas()
    cv2.ellipse(c, (270, 170), (110, 30), 0, 0, 360, K, T)
    cv2.line(c, (160, 170), (175, 450), K, T)
    cv2.line(c, (380, 170), (365, 450), K, T)
    cv2.ellipse(c, (270, 450), (95, 25), 0, 0, 180, K, T)
    cv2.ellipse(c, (385, 300), (70, 85), 0, -80, 80, K, T)
    cv2.ellipse(c, (378, 300), (40, 50), 0, -80, 80, K, T)
    return c


def chair():
    c = canvas()
    cv2.rectangle(c, (180, 90), (420, 300), K, T)          # backrest
    cv2.line(c, (180, 300), (150, 360), K, T)
    cv2.line(c, (420, 300), (450, 360), K, T)
    cv2.line(c, (150, 360), (450, 360), K, T)              # seat front edge
    for x0, x1 in [(160, 160), (440, 440), (200, 200), (400, 400)]:
        cv2.line(c, (x0, 360 if x0 in (160, 440) else 300), (x1, 520 if x0 in (160, 440) else 470), K, T)
    return c


def car():
    c = canvas()
    pts = np.array([[60, 380], [70, 320], [160, 300], [220, 230], [380, 230], [450, 300],
                    [540, 320], [545, 380]], np.int32)
    cv2.polylines(c, [pts], False, K, T)
    cv2.line(c, (60, 380), (130, 380), K, T)
    cv2.line(c, (210, 380), (390, 380), K, T)
    cv2.line(c, (470, 380), (545, 380), K, T)
    for x in (170, 430):
        cv2.circle(c, (x, 385), 42, K, T)
        cv2.circle(c, (x, 385), 15, K, T)
    cv2.line(c, (300, 235), (300, 300), K, T)
    cv2.polylines(c, [np.array([[235, 295], [250, 245], [290, 245], [290, 295]])], True, K, 3)
    return c


def toy():  # teddy bear
    c = canvas()
    cv2.circle(c, (300, 190), 85, K, T)
    cv2.circle(c, (230, 115), 32, K, T)
    cv2.circle(c, (370, 115), 32, K, T)
    cv2.ellipse(c, (300, 390), (110, 130), 0, 0, 360, K, T)
    cv2.circle(c, (270, 175), 8, K, -1)
    cv2.circle(c, (330, 175), 8, K, -1)
    cv2.ellipse(c, (300, 220), (25, 15), 0, 0, 360, K, T)
    cv2.ellipse(c, (185, 330), (35, 60), 30, 0, 360, K, T)
    cv2.ellipse(c, (415, 330), (35, 60), -30, 0, 360, K, T)
    cv2.ellipse(c, (240, 505), (45, 35), 0, 0, 360, K, T)
    cv2.ellipse(c, (360, 505), (45, 35), 0, 0, 360, K, T)
    return c


if __name__ == "__main__":
    for name, fn in [("mug", mug), ("chair", chair), ("car", car), ("teddy", toy)]:
        cv2.imwrite(str(OUT / f"{name}.png"), fn())
        print("wrote", OUT / f"{name}.png")
