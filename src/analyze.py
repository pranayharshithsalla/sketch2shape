"""Input analysis: what must survive into 3D. Finds through-holes (handle loops, racket heads,
open frames), thin structures (wires, single-line handles) and line families (racket strings,
basket wires, wheel spokes) in a preprocessed sketch, and builds a hole-aware solid mask.

Region rule for line drawings (every enclosed area of a line sketch is white, so colour alone
can't tell a hole from a body):
  * the enclosed white regions are labelled; a region whose border mostly faces the outside
    background is part of the solid silhouette (the body, a handle band, a shaft);
  * a region bordered only by other parts of the object is a through-hole candidate (the gap
    inside a handle loop, the head of a racket);
  * two adjacent candidates can't both be holes (alternating slats/gaps): the thinner is solid;
  * a region that contains a family of string/spoke lines is a hole spanned by those lines.
Users can override any region with the click-to-toggle tool (see `toggle`).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import cv2
import numpy as np

log = logging.getLogger(__name__)

HANDLE_WORDS = {"handle", "handles", "loop", "looped", "hoop", "bail"}
WIRE_WORDS = {"wireframe", "cage", "hanger"}  # drawn entirely as wire: every enclosed area is open


@dataclass
class InputAnalysis:
    mask: np.ndarray                  # bool: solid (holes removed, strings/wires kept)
    outer: np.ndarray                 # bool: filled outer silhouette (holes filled)
    strokes: np.ndarray               # bool: drawn lines
    labels: np.ndarray                # int: enclosed white regions (0 = none / strokes / exterior)
    holes: list[dict] = field(default_factory=list)   # {id, area, bbox, centroid, kind}
    families: list[dict] = field(default_factory=list)  # string / wire / spoke line families
    skeleton: np.ndarray | None = None  # bool: medial axis of the solid mask
    thin: np.ndarray | None = None      # bool: skeleton pixels of thin parts
    stroke_px: float = 4.0
    size: float = 512.0                 # object size in px (max bbox side)

    @property
    def hole_count(self) -> int:
        return len(self.holes)

    @property
    def hole_mask(self) -> np.ndarray:
        m = np.zeros(self.labels.shape, bool)
        for h in self.holes:
            m |= self.labels == h["id"]
        return m

    @property
    def expected_genus(self) -> int:
        """Through-holes of the solid, ignoring the many small cells strings/spokes create."""
        return len(self.holes)

    def summary(self) -> str:
        kinds = ", ".join(f"{h['kind']}" for h in self.holes) or "none"
        fam = ", ".join(f"{f['kind']}×{len(f['lines'])}" for f in self.families) or "none"
        return (f"{self.hole_count} through-hole(s) [{kinds}] · line families: {fam} · "
                f"thin-structure skeleton {int(self.thin.sum()) if self.thin is not None else 0} px")


# ---------------------------------------------------------------- line families
def _segments(strokes: np.ndarray, size: float) -> np.ndarray:
    thin = cv2.ximgproc.thinning(strokes.astype(np.uint8) * 255) if hasattr(cv2, "ximgproc") else None
    src = thin if thin is not None else _skeleton(strokes).astype(np.uint8) * 255
    segs = cv2.HoughLinesP(src, 1, np.pi / 180, threshold=int(size * 0.08),
                           minLineLength=int(size * 0.16), maxLineGap=int(size * 0.02))
    return np.zeros((0, 4)) if segs is None else np.asarray(segs).reshape(-1, 4).astype(float)


def _merge_collinear(segs: np.ndarray, tol: float) -> list[np.ndarray]:
    """Hough returns near-duplicate segments for one drawn line: keep one per line."""
    out: list[np.ndarray] = []
    for s in sorted(segs, key=lambda s: -np.hypot(s[2] - s[0], s[3] - s[1])):
        d = np.array([s[2] - s[0], s[3] - s[1]])
        d /= np.linalg.norm(d) + 1e-9
        n = np.array([-d[1], d[0]])
        dup = False
        for o in out:
            od = np.array([o[2] - o[0], o[3] - o[1]])
            od /= np.linalg.norm(od) + 1e-9
            if abs(d @ od) > 0.985 and abs((np.array(o[:2]) - s[:2]) @ n) < tol:
                # same line only if they overlap along it (two spokes either side of a hub don't)
                a0, a1 = sorted((np.array(s[:2]) @ d, np.array(s[2:]) @ d))
                b0, b1 = sorted((np.array(o[:2]) @ d, np.array(o[2:]) @ d))
                if min(a1, b1) - max(a0, b0) > -tol:
                    dup = True
                    break
        if not dup:
            out.append(s)
    return out


def _inside_both_sides(seg, outer: np.ndarray, off: float) -> bool:
    """A string/spoke has the object on both sides; an outline edge has outside on one side."""
    x0, y0, x1, y1 = seg
    d = np.array([x1 - x0, y1 - y0])
    d /= np.linalg.norm(d) + 1e-9
    n = np.array([-d[1], d[0]])
    h, w = outer.shape
    for t in (0.3, 0.5, 0.7):
        p = np.array([x0, y0]) + t * np.array([x1 - x0, y1 - y0])
        for sgn in (1, -1):
            q = np.round(p + sgn * off * n).astype(int)
            if not (0 <= q[0] < w and 0 <= q[1] < h) or not outer[q[1], q[0]]:
                return False
    return True


def find_families(strokes: np.ndarray, outer: np.ndarray, stroke_px: float, size: float) -> list[dict]:
    """Parallel, evenly spaced lines (strings, wires, slats drawn as single lines) and radial
    lines meeting at one point (spokes), all lying inside the object."""
    segs = _merge_collinear(_segments(strokes, size), tol=max(3.0, stroke_px * 1.5))
    segs = [s for s in segs if _inside_both_sides(s, outer, max(4.0, stroke_px * 2.5))]
    if len(segs) < 3:
        return []
    ang = np.array([np.degrees(np.arctan2(s[3] - s[1], s[2] - s[0])) % 180 for s in segs])
    used, fams = set(), []
    for i in np.argsort(ang):
        if i in used:
            continue
        diff = np.minimum(np.abs(ang - ang[i]), 180 - np.abs(ang - ang[i]))
        grp = [j for j in np.nonzero(diff < 10)[0] if j not in used]
        if len(grp) < 3:
            continue
        th = np.radians(np.mean([ang[j] for j in grp]))
        n = np.array([-np.sin(th), np.cos(th)])
        offs = sorted(float(((np.array(segs[j][:2]) + np.array(segs[j][2:])) / 2) @ n) for j in grp)
        gaps = np.diff(offs)
        if len(gaps) and gaps.mean() > stroke_px * 2 and gaps.std() < 0.35 * gaps.mean():
            fams.append({"kind": "strings", "angle": float(np.degrees(th)), "spacing": float(gaps.mean()),
                         "lines": [list(map(float, segs[j])) for j in grp]})
            used.update(grp)
    rest = [j for j in range(len(segs)) if j not in used]
    if len(rest) >= 6:  # radial: least-squares point closest to all remaining lines
        A, b = [], []
        for j in rest:
            x0, y0, x1, y1 = segs[j]
            d = np.array([x1 - x0, y1 - y0])
            d /= np.linalg.norm(d) + 1e-9
            nn = np.array([-d[1], d[0]])
            A.append(nn)
            b.append(nn @ np.array([x0, y0]))
        A, b = np.array(A), np.array(b)
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
        dist = np.abs(A @ c - b)
        spokes = [rest[k] for k in np.nonzero(dist < max(6.0, 0.03 * size))[0]]
        if len(spokes) >= 6:
            fams.append({"kind": "spokes", "center": [float(c[0]), float(c[1])],
                         "lines": _polar_spokes(strokes, c, [segs[j] for j in spokes])})
    return fams


def _polar_spokes(strokes: np.ndarray, c, seed_lines) -> list[list[float]]:
    """Hough misses some spokes; with the hub centre known, scan every angle for a stroke
    running from the hub to the rim."""
    d = [np.hypot(x - c[0], y - c[1]) for ln in seed_lines for x, y in (ln[:2], ln[2:])]
    r0, r1 = min(d) + 2, max(d) - 2
    h, w = strokes.shape
    rs = np.linspace(r0, r1, 40)
    hits = []
    for a in np.radians(np.arange(0, 360, 1.0)):
        x = np.clip(np.round(c[0] + rs * np.cos(a)).astype(int), 0, w - 1)
        y = np.clip(np.round(c[1] + rs * np.sin(a)).astype(int), 0, h - 1)
        hits.append(strokes[y, x].mean())
    hits = np.array(hits)
    out, i = [], 0
    while i < 360:
        if hits[i] > 0.8:  # a run of high-coverage angles = one spoke; take its centre
            j = i
            while j + 1 < 360 and hits[j + 1] > 0.8:
                j += 1
            a = np.radians((i + j) / 2)
            out.append([float(c[0] + r0 * np.cos(a)), float(c[1] + r0 * np.sin(a)),
                        float(c[0] + r1 * np.cos(a)), float(c[1] + r1 * np.sin(a))])
            i = j + 1
        else:
            i += 1
    return out if len(out) >= len(seed_lines) else [list(map(float, ln)) for ln in seed_lines]


def _erase_lines(strokes: np.ndarray, families: list[dict], width: int) -> np.ndarray:
    out = strokes.astype(np.uint8).copy()
    for f in families:
        for x0, y0, x1, y1 in f["lines"]:
            # keep both ends: they sit on the frame / rim / rail the line is attached to
            p0, p1 = np.array([x0, y0]), np.array([x1, y1])
            a, b = p0 + 0.06 * (p1 - p0), p1 - 0.06 * (p1 - p0)
            cv2.line(out, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), 0, width)
    return out > 0


def _line_band(line, shape, width: int) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    x0, y0, x1, y1 = line
    cv2.line(m, (int(x0), int(y0)), (int(x1), int(y1)), 1, width + 2)
    return m > 0


def _components(walls: np.ndarray):
    """-> (exterior bool, labels, border label ids) of the white space between walls."""
    walls = cv2.dilate(walls.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0  # seal 1-px gaps
    _, lab = cv2.connectedComponents((~walls).astype(np.uint8), connectivity=4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    lab[walls] = 0
    return np.isin(lab, list(border)), lab, border


# ---------------------------------------------------------------- regions
def _skeleton(mask: np.ndarray) -> np.ndarray:
    from skimage.morphology import skeletonize
    return skeletonize(mask.astype(bool))


def analyze(strokes: np.ndarray, outer: np.ndarray, text: str = "",
            overrides: dict[int, bool] | None = None) -> InputAnalysis:
    """strokes: bool line map; outer: bool filled silhouette (preprocess._fill_outline).
    overrides: {region_id: is_hole} from the click-to-toggle tool."""
    strokes = strokes.astype(bool)
    outer = outer.astype(bool) | strokes
    ys, xs = np.nonzero(outer)
    size = float(max(np.ptp(xs), np.ptp(ys), 1)) if len(xs) else float(outer.shape[0])
    d = cv2.distanceTransform(strokes.astype(np.uint8), cv2.DIST_L2, 3)
    sk_lines = _skeleton(strokes)  # width measured on the centre line, not over all pixels
    stroke_px = float(max(2.0, 2 * np.median(d[sk_lines]))) if sk_lines.any() else 4.0
    wire = bool(set(re.findall(r"[a-z]+", (text or "").lower())) & WIRE_WORDS)

    families = find_families(strokes, outer, stroke_px, size)
    ew = int(round(stroke_px)) + 3
    base_ext, _, _ = _components(strokes)
    # a "string" whose removal opens the outline to the outside (e.g. a basket rim) is a real edge
    for f in families:
        f["lines"] = [ln for ln in f["lines"]
                      if not (_components(_erase_lines(strokes, [{"lines": [ln]}], ew))[0] & ~base_ext).any()]
    families = [f for f in families if len(f["lines"]) >= (6 if f["kind"] == "spokes" else 3)]
    walls = _erase_lines(strokes, families, ew)
    exterior, lab, border = _components(walls)
    if (exterior & ~base_ext).any():  # lines only leak together: don't trust them
        families = []
        walls = strokes
        exterior, lab, border = _components(walls)
    n = int(lab.max()) + 1
    min_area = max(30, int(1e-3 * outer.size))
    r = int(round(stroke_px)) + 5
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    regions = {}
    labels = np.zeros_like(lab)
    for i in range(1, n):
        if i in border:
            continue
        m = lab == i
        area = int(m.sum())
        if area < min_area:
            continue
        ring = (cv2.dilate(m.astype(np.uint8), ker) > 0) & ~m & (lab > 0)  # what lies across its walls
        ext = float((ring & exterior).sum() / max(ring.sum(), 1))
        thick = float(cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3).max())
        nb = set(np.unique(lab[ring])) - {0, i} - border
        regions[i] = {"mask": m, "area": area, "ext": ext, "thick": thick, "nb": nb}
        labels[m] = i

    # lines families mark the region they span as a (stringed) hole
    fam_regions = {}
    for f in families:
        for x0, y0, x1, y1 in f["lines"]:
            for t in (0.3, 0.7):  # not the midpoint: a spoke "diameter" is centred on the hub
                mx, my = int(round(x0 + t * (x1 - x0))), int(round(y0 + t * (y1 - y0)))
                rid = int(lab[my, mx]) if 0 <= my < lab.shape[0] and 0 <= mx < lab.shape[1] else 0
                if rid in regions:
                    fam_regions[rid] = f["kind"]

    min_thick = max(1.2 * stroke_px, 0.012 * size)  # a screw head or knob is not a hole
    is_hole = {}
    for rid in sorted(regions, key=lambda k: -regions[k]["thick"]):
        g = regions[rid]
        if rid in fam_regions or wire:
            is_hole[rid] = True
        elif (g["ext"] < 0.10 and g["thick"] >= min_thick
              and not any(is_hole.get(o) for o in g["nb"])):
            is_hole[rid] = True
        else:
            is_hole[rid] = False
    # A loop drawn with ONE line (basket handle, wire handle) encloses the same kind of region as
    # a solid dome drawn with one line; geometry can't tell them apart, the prompt can.
    words = set(re.findall(r"[a-z]+", (text or "").lower()))
    if words & HANDLE_WORDS and not any(v and rid not in fam_regions for rid, v in is_hole.items()):
        total = float(outer.sum())
        loops = [rid for rid, g in regions.items() if not is_hole[rid] and 0.15 < g["ext"] < 0.75
                 and g["area"] < 0.35 * total and g["thick"] >= min_thick and rid not in fam_regions]
        if loops:
            is_hole[min(loops, key=lambda k: regions[k]["ext"])] = True
    for rid, v in (overrides or {}).items():
        if rid in regions:
            is_hole[rid] = bool(v)

    holes, hole_mask = [], np.zeros(outer.shape, bool)
    for rid, h in is_hole.items():
        if not h:
            continue
        m = regions[rid]["mask"]
        yy, xx = np.nonzero(m)
        holes.append({"id": int(rid), "area": regions[rid]["area"],
                      "bbox": [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())],
                      "centroid": [float(xx.mean()), float(yy.mean())],
                      "kind": fam_regions.get(rid, "wire cell" if wire else "through-hole")})
        hole_mask |= m
    # strings/spokes stay solid inside their hole
    fam_px = np.zeros(outer.shape, np.uint8)
    for f in families:
        for x0, y0, x1, y1 in f["lines"]:
            cv2.line(fam_px, (int(x0), int(y0)), (int(x1), int(y1)), 1, max(2, int(round(stroke_px))))
    mask = (outer & ~hole_mask) | (fam_px > 0) | strokes

    skel = _skeleton(mask)
    width = 2 * cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    thin = skel & (width <= max(0.025 * size, 1.8 * stroke_px))
    a = InputAnalysis(mask=mask, outer=outer, strokes=strokes, labels=labels, holes=holes,
                      families=families, skeleton=skel, thin=thin, stroke_px=stroke_px, size=size)
    a.regions = {k: {kk: vv for kk, vv in v.items() if kk != "mask"} for k, v in regions.items()}
    log.info("analysis: %s", a.summary())
    return a


def toggle(a: InputAnalysis, x: int, y: int, overrides: dict[int, bool]) -> dict[int, bool]:
    """Click-to-mark/unmark: flip the region under (x, y) between hole and solid."""
    rid = int(a.labels[y, x]) if 0 <= y < a.labels.shape[0] and 0 <= x < a.labels.shape[1] else 0
    if rid:
        current = any(h["id"] == rid for h in a.holes)
        overrides = {**overrides, rid: not current}
    return overrides


def overlay(a: InputAnalysis, sketch_gray: np.ndarray | None = None) -> np.ndarray:
    """RGB debug view: solid (cyan tint), through-holes (amber), strings (magenta), thin
    skeleton (red), enclosed solid regions numbered for the toggle tool."""
    h, w = a.mask.shape
    base = np.full((h, w, 3), 255, np.uint8) if sketch_gray is None else np.dstack([sketch_gray] * 3).copy()
    out = base.astype(np.float32)
    out[a.mask] = out[a.mask] * 0.55 + np.array([0, 229, 255]) * 0.45
    hm = a.hole_mask
    out[hm] = out[hm] * 0.35 + np.array([245, 158, 11]) * 0.65
    out[a.strokes] = (20, 20, 30)
    if a.thin is not None:
        t = cv2.dilate(a.thin.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        out[t] = (230, 40, 60)
    for f in a.families:
        for x0, y0, x1, y1 in f["lines"]:
            cv2.line(out, (int(x0), int(y0)), (int(x1), int(y1)), (200, 40, 200), 2)
    img = out.clip(0, 255).astype(np.uint8)
    for hh in a.holes:
        cx, cy = map(int, hh["centroid"])
        cv2.putText(img, "hole", (cx - 18, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 60, 0), 1, cv2.LINE_AA)
    return img
