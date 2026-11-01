"""Sketch2Shape: Gradio UI. Draw, upload or just type -> textured, validated 3D asset."""
from __future__ import annotations

import argparse
import re
import threading
import traceback
from pathlib import Path

import gradio as gr
import numpy as np

from src import api, config, designer, image2mesh, sketch2image
from src.designer import Session
from src.ui_theme import (CSS, CYAN, LOGO_HTML, MODES, THEME, grid_canvas, metrics_html,
                          panel_title)
from src.decal import FONTS, SIDES
from src.pipeline import StageError, candidate_to_3d, make_candidates, run
from src.prompt import MATERIALS
from src.render import turntable_gif
from src.templates import available_names, make_sketch

# (sketch, prompt) pairs: show that the prompt refines the drawing
EXAMPLE_PROMPTS = {
    "mug": "a red ceramic mug with a black handle",
    "cup": "a sky blue cup with a gold rim and white handle",
    "star": "a gold star",
    "chair": "a wooden chair",
    "car": "a red car",
    "teddy": "a brown teddy bear",
}
EXAMPLES = [[str(p), EXAMPLE_PROMPTS.get(p.stem, "")] for p in sorted(config.EXAMPLES.glob("*.png"))]
SERVER = {"port": 7860, "ar": False}  # set at launch; AR links need the LAN-served FastAPI app
KEEP_FINISH = "keep current finish"
CONTROLS = [("Shape + style", "both"), ("Shape only", "shape"), ("Style & colour only", "style")]


def _understood(r, prompt: str) -> str:
    """'Prompt understood as' line: object, engine, shape modifiers, per-part colour swatches."""
    obj = prompt.strip() or "—"
    if r.caption:
        obj = f"{r.caption} *(auto-caption, offline CLIP)*" + (f" + “{prompt.strip()}”" if prompt.strip() else "")
    bits = [f"**Object:** {obj}", f"**Engine:** `{r.mesh_provider}`"]
    for line in r.logs:
        line = line.strip()
        if "template from the prompt" in line:
            name = re.search(r"'([^']+)'", line).group(1)
            bits[0] = f"**Object:** {prompt.strip()} *(no sketch: built from the '{name}' template)*"
        if line.startswith("prompt shape modifiers:"):
            bits.append("**Shape words:** " + line.split(":", 1)[1].strip())
        if line.startswith("materials:"):
            sw = re.sub(r"(\w+) (#[0-9a-f]{6}) (\w+)",
                        lambda m: f"{m.group(1)} <span style='display:inline-block;width:12px;height:12px;"
                                  f"border-radius:3px;background:{m.group(2)};border:1px solid #888'></span> "
                                  f"{m.group(3)}", line.split(":", 1)[1].strip())
            bits.append("**Materials:** " + sw)
    return "  \n".join(bits)


def _status_md() -> str:
    img = ", ".join(sketch2image.available())
    mesh = ", ".join(image2mesh.available())
    return (f"**Active providers**: image → `{img}` · 3D → `{mesh}`. "
            "Missing keys are skipped; the offline `local` path means a .glb is always produced.")


def generate(draw, upload, prompt, material, n_cands, img_prov, mesh_prov, skip_2d, seed,
             progress=gr.Progress()):
    src = upload if upload is not None else draw
    if upload is None and _unedited_template(draw, prompt):
        src = None  # build from the exact template geometry
    try:
        r = run(src, prompt or "", material, int(n_cands), img_prov, mesh_prov, skip_2d, int(seed),
                progress=progress)
    except StageError as e:
        raise gr.Error(f"Stage '{e.stage}' failed: {str(e)[:400]}")
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Unexpected error: {e}")

    best = r.scores[r.best]
    rep = r.report
    stats = (
        f"{_understood(r, prompt or '')}\n\n"
        f"### Fidelity **{best['score']:.2f}**  (silhouette IoU {best['iou']:.2f} · edge {best['edge']:.2f})\n"
        f"| | |\n|---|---|\n"
        f"| Image provider | `{r.image_provider}` |\n| 3D provider | `{r.mesh_provider}` |\n"
        f"| Parts / materials | {rep.get('parts', 1)} |\n"
        f"| Vertices / faces | {rep['vertices']:,} / {rep['faces']:,} |\n"
        f"| Extents (unit box) | {rep['extents']} |\n| Watertight | {'yes' if rep['watertight'] else 'no'} |\n"
        f"| Colour preserved | {rep['colour']} |\n| GLB re-opens | {'yes' if rep['glb_reopens'] else 'no'} "
        f"({rep['glb_size_kb']} KB) |\n"
        f"| **Total time** | **{r.timings['total']} s** |\n\n"
        + " · ".join(f"{k} {v}s" for k, v in r.timings.items() if k != "total")
        + ("\n\n" + accuracy_md(r.accuracy) if r.accuracy else "")
    )
    pre = [(r.images["sketch_clean"], "cleaned sketch"), (r.images["sketch_mask"], "hole-aware silhouette")]
    if r.images.get("analysis"):
        pre.append((r.images["analysis"], "detected holes (amber) · strings (magenta) · thin parts (red)"))
    handoff = {"plan": r.plan, "glb": r.glb, "files": [r.glb, r.obj_zip, r.stl], "report": r.report,
               "prompt": prompt or r.caption or "my sketch", "model": r.mesh_provider,
               "seconds": r.timings.get("total")}
    return (pre, r.candidates, r.glb, [r.glb, r.obj_zip, r.stl], stats,
            "\n".join(r.logs) + f"\n\nSaved to {r.run_dir}", r.glb, None, handoff,
            gr.Button(visible=bool(r.plan)))


def sketch_from_prompt(prompt):
    """'Draw it for me': put the prompt's template sketch on the canvas, where it can be edited."""
    hit = make_sketch(prompt or "")
    if hit is None:
        raise gr.Error("No template for that prompt yet. Name one of: " + ", ".join(available_names()))
    img, _ = hit
    rgba = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
    return {"background": rgba, "layers": [], "composite": rgba}, gr.Tabs(selected="draw")


def _unedited_template(draw, prompt) -> bool:
    """True when the canvas holds exactly the 'Draw it for me' template with no strokes added,
    so the exact template geometry can be used instead of re-detecting it from pixels."""
    if not isinstance(draw, dict) or not prompt:
        return False
    canvas = draw.get("composite") if draw.get("composite") is not None else draw.get("background")
    hit = make_sketch(prompt)
    if canvas is None or hit is None:
        return False
    layers = [np.asarray(l) for l in (draw.get("layers") or []) if l is not None]
    if any(l.ndim == 3 and l.shape[-1] == 4 and l[..., 3].max() > 0 for l in layers):
        return False  # the user drew on top of the template
    # the canvas may come back resized / re-encoded: compare loosely in grayscale
    import cv2
    c = np.asarray(canvas)
    c = cv2.cvtColor(c[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    t = cv2.cvtColor(hit[0], cv2.COLOR_RGB2GRAY)
    c = cv2.resize(c, t.shape[::-1], interpolation=cv2.INTER_AREA)
    return float(np.abs(c.astype(int) - t.astype(int)).mean()) < 6.0


def _canvas_reset():
    g = grid_canvas()
    return {"background": g, "layers": [], "composite": g}


def _ar(glb):
    """(QR image, markdown) for viewing `glb` in AR on a phone on the same network."""
    if not glb or not SERVER["ar"]:
        return None, ("AR needs the LAN server: run `python app.py` (not `--share`) and open the page "
                      "from this PC; the QR then points your phone at it.")
    url = f"http://{api.lan_ip()}:{SERVER['port']}/ar?m={api.rel(glb)}"
    return api.qr_png(url), (f"**Scan with your phone** (same Wi-Fi) → tap *View in your room*.  \n"
                             f"[{url}]({url})  \nAndroid: Scene Viewer · iPhone: Quick Look · desktop: 3D view.")


def accuracy_md(a: dict | None) -> str:
    """Accuracy report: IoU, thin-feature recall, expected vs found holes, refinement history."""
    if not a:
        return "*No sketch/photo in this design yet: structure is checked against an input drawing.*"
    b = a.get("before", {})
    lines = [f"### Accuracy **{a['accuracy']:.0%}**" + (f"  (raw model {b.get('accuracy', 0):.0%})" if a.get("refined") else ""),
             "| | raw | final |", "|---|---|---|",
             f"| Silhouette IoU | {b.get('iou', 0):.2f} | {a['iou']:.2f} |",
             f"| Thin-feature recall | {b.get('thin_recall', 0):.2f} | {a['thin_recall']:.2f} |",
             f"| Through-holes expected / open | {b.get('holes_expected', 0)} / {b.get('holes_open', 0)} | "
             f"{a['holes_expected']} / {a['holes_open']} |",
             f"| Extra holes | {b.get('holes_extra', 0)} | {a['holes_extra']} |",
             "", f"*Input: {a.get('summary', '')}*  ", f"*Accuracy = 0.4·IoU + 0.3·thin recall + 0.3·hole match; "
             f"target {a.get('target', 0.9):.0%}. Carving only opens holes visible from the sketch's side.*"]
    if a.get("warning"):
        lines.insert(1, f"> ⚠️ **{a['warning']}**")
    hist = a.get("history") or []
    if len(hist) > 1:
        lines += ["", "**Refinement history:** " + " → ".join(
            f"{h['step']} ({h['accuracy']:.0%})" if "accuracy" in h else f"{h['step']} (failed)" for h in hist)]
    return "\n".join(lines)


def _acc_images(cur) -> list:
    a = cur.get("accuracy_report") if cur else None
    if not a or not a.get("refined"):
        return []
    from pathlib import Path
    d = Path(a["raw_glb"]).parent / "refine" / "debug"
    if not d.exists():
        d = Path(cur["glb"]).parent / "refine" / "debug"
    out = []
    for f, cap in (("refine_before.png", "raw model vs input"), ("refine_after.png", "refined vs input")):
        if (d / f).exists():
            out.append((str(d / f), cap + " · green ok · red missing · blue extra"))
    return out


def _parts_view(cur):
    """Checkbox choices for the Colours panel: every part (with its colour), plus 'all petals'-style
    groups when several parts share a name."""
    from src.recolor import groups, list_parts
    try:
        parts = list_parts(cur.get("plain_glb") or cur["glb"])
    except Exception:
        parts = []
    choices = [(f"{p['label']}  ·  {p['colour']}", p["name"]) for p in parts]
    choices += [(f"▣ {g}", "group:" + "|".join(v)) for g, v in groups(parts).items()]
    return gr.CheckboxGroup(choices=choices, value=[]), _swatches(parts)


def _swatches(parts) -> str:
    if not parts:
        return '<div class="k" style="color:var(--text-2)">Build a model to list its parts.</div>'
    chips = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:6px;margin:3px 10px 3px 0;font-size:12px;'
        f'color:var(--text-2)"><span style="width:14px;height:14px;border-radius:4px;background:{p["colour"]};'
        f'border:1px solid var(--border);display:inline-block"></span>{p["label"]}</span>' for p in parts)
    return f'<div>{chips}</div>'


def _designer_view(session, history):
    """Outputs (in d_out order): clear inputs, chat, model, download, files, metrics, title,
    create-button label, session, version timeline, validation report, AR QR, AR text,
    accuracy report, before/after images, raw-model toggle."""
    cur = session.current
    if not cur:
        return ("", _canvas_reset(), None, history, None, gr.DownloadButton(value=None, interactive=False),
                None, metrics_html(None, None, None, None), panel_title("3D Preview", "nothing yet"),
                gr.Button(value="Create 3D Shape"), session, gr.Radio(choices=[], value=None), None,
                None, "", accuracy_md(None), [], gr.Checkbox(value=False, interactive=False),
                gr.CheckboxGroup(choices=[], value=[]), _swatches([]))
    rep = cur["report"]
    title = panel_title("3D Preview", f'<span class="badge">v{len(session.versions)} · {cur["model"]}</span>')
    tl = designer.timeline(session)
    report = {**rep, "texture_present": "texture" in str(rep.get("colour", "")),
              "decals": [d.get("text") or "logo" for d in cur.get("decals") or []],
              "sketch_fidelity": cur.get("accuracy"), "engine": cur["model"], "build_seconds": cur.get("seconds")}
    qr, ar_md = _ar(cur["glb"])
    return ("", _canvas_reset(), None, history, cur["glb"],
            gr.DownloadButton(value=cur["glb"], interactive=True), cur["files"],
            metrics_html(cur.get("accuracy"), bool(rep.get("watertight")), rep.get("parts"), cur.get("seconds")),
            title, gr.Button(value="Update 3D Shape"), session, gr.Radio(choices=tl, value=tl[-1]), report,
            qr, ar_md, accuracy_md(cur.get("accuracy_report")), _acc_images(cur),
            gr.Checkbox(value=False, interactive=bool(cur.get("raw_glb"))),
            *_parts_view(cur))


def _attachment(mode, canvas, upload):
    """The image sent with this message, chosen by the mode: canvas, upload, or none."""
    if mode == MODES[1]:
        return upload
    if mode == MODES[0]:
        return canvas if designer.image_png(canvas) is not None else None
    return None


def designer_turn(mode, message, canvas, upload, control, history, session):
    image = _attachment(mode, canvas, upload)
    history = list(history or [])
    if message and message.strip():
        history.append({"role": "user", "content": message})
    png = designer.image_png(image)
    if png is not None:  # show the sketch/photo in the conversation
        shot = config.OUTPUTS / "_chat" / f"{__import__('uuid').uuid4().hex[:10]}.png"
        shot.parent.mkdir(parents=True, exist_ok=True)
        shot.write_bytes(png)
        history.append({"role": "user", "content": {"path": str(shot)}})
    try:
        session, reply = designer.turn(session, message, image, control or "both", photo=mode == MODES[1])
    except Exception as e:
        traceback.print_exc()
        reply = f"⚠️ I couldn't apply that ({str(e)[:200]}). Try rephrasing, or undo."
    history.append({"role": "assistant", "content": reply})
    return _designer_view(session, history)


def designer_undo(history, session):
    session, reply = designer.undo(session)
    history = list(history or []) + [{"role": "assistant", "content": reply}]
    return _designer_view(session, history)


def designer_new():
    return _designer_view(designer.Session(), [])


def _say(history, text):
    return list(history or []) + [{"role": "assistant", "content": text}]


def _hex(color) -> str:
    """ColorPicker may return '#rrggbb' or 'rgba(r, g, b, a)'."""
    m = re.findall(r"[\d.]+", str(color or ""))
    if str(color).startswith("rgb") and len(m) >= 3:
        return "#" + "".join(f"{int(float(v)):02x}" for v in m[:3])
    return color or "#FFFFFF"


def designer_decal(text, logo, font, color, side, x, y, size, rot, move, history, session):
    try:
        session, reply = designer.add_decal(session, text, logo, _hex(color), font, x, y, size, rot, side,
                                            replace_last=bool(move))
    except Exception as e:
        traceback.print_exc()
        reply = f"⚠️ Couldn't bake that: {str(e)[:200]}"
    return _designer_view(session, _say(history, reply))


def decal_live(text, logo, font, color, side, x, y, size, rot, move, history, session):
    """Slider released with 'live move' on: re-place the last decal (like dragging it)."""
    cur = session.current if session else None
    if not move or not cur or cur.get("kind") != "decal":
        return [gr.skip()] * 20
    return designer_decal(text, logo, font, color, side, x, y, size, rot, True, history, session)


def versions_revert(choice, history, session):
    tl = designer.timeline(session)
    if choice not in tl:
        return _designer_view(session, _say(history, "Pick a version in the timeline first."))
    session, reply = designer.revert(session, tl.index(choice))
    return _designer_view(session, _say(history, reply))


def versions_compare(choice, session):
    tl = designer.timeline(session)
    if choice not in tl:
        return gr.Model3D(visible=False), gr.HTML(visible=False)
    v = session.versions[tl.index(choice)]
    cur = session.current
    note = (f'<div class="panel-title"><h3>Compare</h3><span>{choice}  ⟷  current '
            f'v{len(session.versions)}</span></div>')
    return gr.Model3D(value=v["glb"], visible=True), gr.HTML(note, visible=True) if cur else gr.HTML(visible=False)


def studio_candidates(mode, message, canvas, upload, auto, history, session, as_is=False,
                      progress=gr.Progress()):
    """Generate 4 candidates (one batch where the provider supports it), rank them, and either
    wait for the user's pick or (Auto) send the best one straight to 3D."""
    image = _attachment(mode, canvas, upload)
    if image is None and not (message or "").strip():
        raise gr.Error("Draw or upload something (or type a prompt) to generate candidates.")
    if session.run_dir is None:
        session.run_dir = config.OUTPUTS / __import__("time").strftime("design_%Y%m%d_%H%M%S")
        session.run_dir.mkdir(parents=True, exist_ok=True)
    progress(0.1, desc="Generating + ranking 4 candidates")
    try:
        png0 = designer.image_png(image)
        if png0 is not None:  # the candidates' sketch is the shape target for accuracy/refinement
            session.ref_png, session.ref_text = png0, (message or "").strip()
        r = make_candidates(image, message or "", 4, as_is=bool(as_is),
                            out_dir=session.run_dir / f"cands_{len(session.versions) + 1}",
                            hole_overrides=session.hole_overrides)
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Candidate generation failed: {str(e)[:300]}")
    png = designer.image_png(image)
    state = {"run": {k: v for k, v in r.items() if k != "sketch"}, "sel": r["best"], "ref": png,
             "prompt": (message or "").strip() or r["caption"] or "candidate"}
    t = r["timings"]
    note = ("**Used as-is** (clean illustration: no diffusion redraw). " if r.get("bypass") else "") + (
            f"Generated {len(r['cands'])} candidates with `{r['provider']}` in {t['total']} s "
            f"(generate {t['generate']} s · background {t['remove_bg']} s · rank {t['rank']} s). "
            f"Best match: #{r['best'] + 1}.")
    gallery = [(c["path"], c["label"]) for c in r["cands"]]
    if auto:
        view = studio_build(state, _say(history, note + " Auto: building it in 3D."), session, progress)
        return (gr.Gallery(value=gallery, visible=True, selected_index=r["best"]), gr.Button(visible=True),
                state, *view)
    return (gr.Gallery(value=gallery, visible=True, selected_index=r["best"]), gr.Button(visible=True), state,
            *_designer_view(session, _say(history, note + " Click a card, then *Build 3D*.")))


def cand_select(state, evt: gr.SelectData):
    if state:
        state["sel"] = evt.index
    return state


def studio_build(state, history, session, progress=gr.Progress()):
    """Only the chosen candidate goes through the expensive image-to-3D step."""
    if not state:
        raise gr.Error("Generate candidates first.")
    i = state["sel"] if state["sel"] is not None else state["run"]["best"]
    c = state["run"]["cands"][i]
    progress(0.3, desc=f"Reconstructing candidate #{i + 1} in 3D")
    n = len(session.versions) + 1
    try:
        sk = designer.analysis_of(session) if state.get("ref") is not None else None
        out = candidate_to_3d(c["rgba_path"], session.run_dir / f"v{n}", name=f"model_v{n}",
                              analysis=sk.analysis if sk is not None else None, procedural=session.procedural)
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"3D reconstruction failed: {str(e)[:300]}")
    acc = (out.get("accuracy") or {}).get("accuracy") or designer.sketch_accuracy(out["glb"], state.get("ref"))
    designer.add_version(session, out["glb"], [out["glb"], out["obj_zip"], out["stl"]], out["report"],
                         f"{state['prompt']} (candidate #{i + 1})", out["provider"], "candidate",
                         out["timings"]["total"], accuracy=acc)
    session.current["accuracy_report"], session.current["raw_glb"] = out.get("accuracy"), out.get("raw_glb")
    session.history.append((f"(build candidate #{i + 1})", f"Built with {out['provider']}."))
    return _designer_view(session, _say(history, f"Built candidate #{i + 1} with `{out['provider']}` "
                                                 f"in {out['timings']['total']} s."))


def _analysis_png(mode, canvas, upload):
    image = _attachment(mode, canvas, upload)
    return designer.image_png(image)


def studio_analyse(mode, canvas, upload, message, session):
    """Detect holes / thin parts / strings in the current sketch and show them for correction."""
    png = _analysis_png(mode, canvas, upload)
    if png is None:
        raise gr.Error("Draw or upload a sketch first.")
    import hashlib
    h = hashlib.md5(png).hexdigest()
    if getattr(session, "hole_src", None) != h:  # corrections belong to one drawing
        session.hole_overrides, session.hole_src = {}, h
    session.ref_png, session.ref_text = png, (message or "").strip()
    sk = designer.analysis_of(session)
    from src.analyze import overlay
    a = sk.analysis
    md = (f"**{a.hole_count} through-hole(s)** · {len(a.families)} string/wire famil"
          f"{'y' if len(a.families) == 1 else 'ies'} · thin parts {'yes' if a.thin.sum() > 10 else 'no'}  \n"
          "Amber = hole, magenta = strings (rebuilt as tubes), red = thin parts. "
          "**Click any enclosed area to mark/unmark it as a hole.**")
    return overlay(a, sk.clean), md, session


def studio_toggle_hole(session, evt: gr.SelectData):
    if session is None or session.ref_png is None:
        raise gr.Error("Press *Analyse sketch* first.")
    from src.analyze import overlay, toggle
    sk = designer.analysis_of(session)
    x, y = evt.index[0], evt.index[1]
    session.hole_overrides = toggle(sk.analysis, int(x), int(y), session.hole_overrides)
    sk = designer.analysis_of(session)
    return overlay(sk.analysis, sk.clean), session


def studio_procedural(flag, session):
    session.procedural = bool(flag)
    return session


def studio_show_raw(show, session):
    cur = session.current if session else None
    if not cur:
        return gr.skip()
    return cur["raw_glb"] if show and cur.get("raw_glb") else cur["glb"]


def studio_recolor(picked, colour, material, history, session):
    names = []
    for v in picked or []:
        names += v[len("group:"):].split("|") if v.startswith("group:") else [v]
    names = list(dict.fromkeys(names))
    try:
        session, reply = designer.recolor_parts(session, names, _hex(colour),
                                                None if material in (None, "", KEEP_FINISH) else material)
    except Exception as e:
        traceback.print_exc()
        reply = f"⚠️ Couldn't recolour: {str(e)[:200]}"
    return _designer_view(session, _say(history, reply))


def studio_select_all(session):
    cur = session.current if session else None
    if not cur:
        return gr.skip()
    from src.recolor import list_parts
    return gr.CheckboxGroup(value=[p["name"] for p in list_parts(cur.get("plain_glb") or cur["glb"])])


def studio_spin(session):
    cur = session.current if session else None
    if not cur:
        raise gr.Error("Create a model first.")
    return make_gif(cur["glb"])


def studio_print_preflight(session):
    cur = session.current if session else None
    if not cur or not cur.get("glb") or not Path(cur["glb"]).exists():
        raise gr.Error("Build a 3D model first before running print pre-flight.")
    from src.export import finalize
    glb_p = Path(cur["glb"])
    fin = finalize(str(glb_p), glb_p.parent, name=glb_p.stem, run_printability=True)
    p_info = fin["report"].get("printability", {})
    cur["report"]["printability"] = p_info

    ov = p_info.get("overhang", {})
    st = p_info.get("stability", {})
    est = p_info.get("estimate", {})

    status_badge = "✅ STABLE" if st.get("is_stable") else f"⚠️ {st.get('rating', 'UNSTABLE')}"
    support_note = f"⚠️ Requires supports (~{ov.get('support_required_area_pct', 0)}% of surface)" if ov.get("needs_supports") else "✅ Minimal/no supports required"
    wt_str = "✅ Yes" if p_info.get("watertight") else "❌ No"

    md = (
        f"### 🖨️ 3D Print Pre-Flight Report\n"
        f"- **Stability:** **{status_badge}** (tipping angle: **{st.get('tipping_angle_deg', 0)}°**, margin: {st.get('stability_margin', 0)})\n"
        f"- **Overhangs:** **{ov.get('support_required_area_pct', 0)}%** ({support_note})\n"
        f"- **Estimated Weight (PLA):** **{est.get('filament_pla_grams', 0)} g** (at ~10cm scaled height, 20% infill)\n"
        f"- **Estimated Print Time:** ~**{est.get('estimated_print_time_hours', 0)} hrs**\n"
        f"- **Watertight:** {wt_str}\n"
    )

    report_view = {
        **cur["report"],
        "texture_present": "texture" in str(cur["report"].get("colour", "")),
        "decals": [d.get("text") or "logo" for d in cur.get("decals") or []],
        "sketch_fidelity": cur.get("accuracy"),
        "engine": cur["model"],
        "build_seconds": cur.get("seconds"),
    }
    return gr.Markdown(value=md, visible=True), report_view


def studio_export_extra(session):
    cur = session.current if session else None
    if not cur or not cur.get("glb") or not Path(cur["glb"]).exists():
        raise gr.Error("Build a 3D model first before exporting extra formats.")
    from src.export import finalize
    glb_p = Path(cur["glb"])
    fin = finalize(str(glb_p), glb_p.parent, name=glb_p.stem, export_scad=True, generate_lods_flag=True)

    new_files = list(cur.get("files") or [])
    if fin.get("scad") and Path(fin["scad"]).exists() and fin["scad"] not in new_files:
        new_files.append(fin["scad"])
    for lod_name, lod_dict in fin.get("lods", {}).items():
        lod_path = lod_dict.get("path")
        if lod_path and Path(lod_path).exists() and lod_path not in new_files:
            new_files.append(lod_path)

    cur["files"] = new_files
    gr.Info(f"Exported OpenSCAD (.scad) and {len(fin.get('lods', {}))} LOD models to 'Other formats' below!")
    return new_files


def switch_mode(mode, session):
    """Top-bar mode selector: show the matching input and adapt the prompt hint."""
    editing = session is not None and session.current is not None
    hints = {MODES[0]: "Optional: describe it (e.g. a gold trophy with two handles)",
             MODES[1]: "Optional: what is it, and how should it look?",
             MODES[2]: "Describe what you need, e.g. a wooden chair with a red leather seat"}
    hint = "Describe a change, e.g. make the handles thinner" if editing else hints[mode]
    return (gr.Sketchpad(visible=mode == MODES[0]), gr.Image(visible=mode == MODES[1]),
            gr.HTML(visible=mode == MODES[2]), gr.Textbox(placeholder=hint))


def rotate_view(action, cam, session):
    """Rotation controls: orbit / tilt / zoom / reset the 3D preview camera."""
    alpha, beta, radius = cam or (45.0, 75.0, None)
    step = {"left": (-45, 0, 1), "right": (45, 0, 1), "up": (0, -15, 1), "down": (0, 15, 1),
            "in": (0, 0, 0.8), "out": (0, 0, 1.25)}
    if action == "reset":
        alpha, beta, radius = 45.0, 75.0, None
    else:
        da, db, dr = step[action]
        alpha, beta = (alpha + da) % 360, min(max(beta + db, 5), 175)
        radius = (radius or 3.0) * dr if dr != 1 else radius
    glb = session.current["glb"] if session is not None and session.current else None
    return gr.Model3D(value=glb, camera_position=(alpha, beta, radius)), [alpha, beta, radius]


def refine_in_chat(handoff):
    """Hand a Sketch → 3D result to the Studio so it can be refined in conversation."""
    if not handoff or not handoff.get("plan"):
        raise gr.Error("Only AI-CAD results can be refined in the Studio (needs GEMINI_API_KEY).")
    s = designer.Session()
    s.run_dir = config.OUTPUTS / ("design_from_sketch_" + __import__("time").strftime("%Y%m%d_%H%M%S"))
    s.run_dir.mkdir(parents=True, exist_ok=True)
    s.versions.append(dict(handoff))
    msg = f"Built from your sketch: {handoff['plan'].get('object', 'object')}. Tell me what to change."
    s.history.append((handoff["prompt"], msg))
    history = [{"role": "user", "content": f"(sketch) {handoff['prompt']}"},
               {"role": "assistant", "content": msg}]
    view = _designer_view(s, history)
    return (*view, gr.Tabs(selected="designer"))


def _already_running(port: int) -> bool:
    """True if a Sketch2Shape server already answers on this port."""
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1.5) as r:
            return b"Sketch2Shape" in r.read(4096) or r.status == 200
    except Exception:
        return False


def _free_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        return 99.0  # unknown: try the preload anyway


def make_gif(glb):
    """Runs after the model is shown, so the spin preview never delays the main result."""
    if not glb:
        return None
    try:
        from pathlib import Path
        return turntable_gif(glb, Path(glb).with_name(Path(glb).stem + "_spin.gif"))
    except Exception:
        traceback.print_exc()
        return None


def build() -> gr.Blocks:
    with gr.Blocks(title="Sketch2Shape") as demo:
        with gr.Row(elem_id="topbar"):
            with gr.Column(scale=2, min_width=280):
                gr.HTML(LOGO_HTML)
            with gr.Column(scale=3, min_width=520):
                mode = gr.Radio(MODES, value=MODES[0], show_label=False, container=False, elem_id="modes")
        with gr.Tabs() as main_tabs:
            with gr.Tab("Studio", id="designer"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, elem_classes="panel"):
                        gr.HTML(panel_title("Sketch", "draw · upload · or just explain"))
                        d_canvas = gr.Sketchpad(value=_canvas_reset(), type="numpy", canvas_size=(512, 512),
                                                height=440, show_label=False, elem_id="sketchpad",
                                                brush=gr.Brush(colors=[CYAN, "#FFFFFF", "#F59E0B"],
                                                               default_size=4, color_mode="defaults"),
                                                eraser=gr.Eraser(default_size=12))
                        d_upload = gr.Image(type="numpy", image_mode="RGBA", height=440, show_label=False,
                                            visible=False)
                        d_explain = gr.HTML('<div class="metric" style="padding:22px"><div class="v ok">💬</div>'
                                            '<div class="k" style="text-transform:none;font-size:13px">'
                                            'Describe the object below: shape, parts, colours, materials. '
                                            'You can switch to a sketch or photo at any time.</div></div>',
                                            visible=False)
                        chat_box = gr.Textbox(show_label=False, lines=3, max_lines=6,
                                              placeholder="Optional: describe it (e.g. a gold trophy with two handles)")
                        control = gr.Radio(CONTROLS, value="both", label="This input controls", elem_id="control")
                        with gr.Accordion("🕳️ Holes & thin parts", open=False):
                            with gr.Row():
                                analyse_btn = gr.Button("🔍 Analyse sketch", size="sm")
                                proc_chk = gr.Checkbox(True, label="Rebuild thin parts procedurally")
                                asis_chk = gr.Checkbox(False, label="Use image as-is (skip generation)")
                            hole_img = gr.Image(show_label=False, interactive=False, height=300, type="numpy")
                            hole_md = gr.Markdown("Detects handle loops, open frames, strings and wires so "
                                                  "they survive into 3D. Click a region to correct it.")
                        create_btn = gr.Button("Create 3D Shape", variant="primary", elem_id="create-btn")
                        with gr.Row(equal_height=True):
                            cand_btn = gr.Button("🎲 Generate 4 candidates", size="sm", scale=2)
                            auto_pick = gr.Checkbox(False, label="Auto: best one → 3D", scale=1)
                        cand_gallery = gr.Gallery(visible=False, columns=2, height="auto", allow_preview=False,
                                                  object_fit="contain",
                                                  label="Candidates · sketch 50% · 3D-ready 30% · prompt 20%")
                        build_cand = gr.Button("Build 3D from the selected candidate", visible=False, size="sm")
                        cand_state = gr.State()
                        with gr.Row():
                            undo_btn = gr.Button("↩ Undo", size="sm")
                            new_btn = gr.Button("＋ New design", size="sm")
                        with gr.Accordion("Conversation & versions", open=False):
                            chat = gr.Chatbot(show_label=False, height=320, elem_id="chat")
                        gr.Examples(["a gold trophy on a black marble base", "a wooden chair with a red leather seat",
                                     "make the handles thinner", "paint it navy blue", "make it twice as tall"],
                                    inputs=chat_box, label="Try")
                    with gr.Column(scale=6, elem_classes="panel"):
                        d_title = gr.HTML(panel_title("3D Preview", "nothing yet"))
                        d_model = gr.Model3D(show_label=False, height=470, elem_id="viewport",
                                             clear_color=(0.075, 0.114, 0.212, 1.0))
                        with gr.Row(elem_classes="ctl"):
                            rot = {k: gr.Button(lbl, size="sm", elem_classes="ctl") for k, lbl in
                                   [("left", "⟲ Left"), ("right", "Right ⟳"), ("up", "▲ Tilt"),
                                    ("down", "▼ Tilt"), ("in", "＋ Zoom"), ("out", "－ Zoom"),
                                    ("reset", "⤾ Reset")]}
                        d_metrics = gr.HTML(metrics_html(None, None, None, None))
                        d_dl = gr.DownloadButton("Download 3D Model (.GLB)", variant="primary",
                                                 elem_id="dl-btn", interactive=False)
                        with gr.Accordion("🕘 Version timeline · revert · compare", open=True):
                            versions = gr.Radio([], show_label=False, elem_id="timeline")
                            with gr.Row():
                                revert_btn = gr.Button("↩ Revert to selected", size="sm")
                                compare_btn = gr.Button("⟷ Compare with current", size="sm")
                            compare_note = gr.HTML(visible=False)
                            compare_model = gr.Model3D(visible=False, height=300, show_label=False,
                                                       clear_color=(0.075, 0.114, 0.212, 1.0))
                        with gr.Accordion("🎨 Colours & materials · pick one or more parts", open=False):
                            parts_html = gr.HTML(_swatches([]))
                            parts_cb = gr.CheckboxGroup(choices=[], label="Parts to recolour", elem_id="parts")
                            with gr.Row():
                                part_colour = gr.ColorPicker("#E0115F", label="New colour", scale=1)
                                part_mat = gr.Dropdown([KEEP_FINISH] + sorted(__import__("src.textures", fromlist=["SPECS"]).SPECS),
                                                       value=KEEP_FINISH, label="Material", scale=2)
                            with gr.Row():
                                select_all = gr.Button("Select all parts", size="sm")
                                recolor_btn = gr.Button("Apply colour", variant="primary", size="sm")
                        with gr.Accordion("🔤 Text & logo · baked into the texture", open=False):
                            with gr.Row():
                                dec_text = gr.Textbox(label="Text", placeholder="e.g. BRAND NAME", scale=3)
                                dec_color = gr.ColorPicker("#111111", label="Colour", scale=1)
                            with gr.Row():
                                dec_font = gr.Dropdown(list(FONTS), value="Bold sans", label="Font")
                                dec_side = gr.Radio(list(SIDES), value="front", label="Side")
                            dec_logo = gr.Image(label="…or a logo (PNG, transparent background)", type="numpy",
                                                image_mode="RGBA", height=120)
                            with gr.Row():
                                dec_x = gr.Slider(0, 1, 0.5, step=0.01, label="← position →")
                                dec_y = gr.Slider(0, 1, 0.5, step=0.01, label="↓ position ↑")
                            with gr.Row():
                                dec_size = gr.Slider(0.05, 1.2, 0.4, step=0.01, label="Size")
                                dec_rot = gr.Slider(-180, 180, 0, step=1, label="Rotation °")
                            with gr.Row():
                                dec_btn = gr.Button("Bake onto model", variant="primary", size="sm")
                                dec_move = gr.Checkbox(False, label="Live move: sliders re-place the last decal")
                        with gr.Accordion("📱 View in AR · 360° turntable", open=False):
                            with gr.Row():
                                ar_qr = gr.Image(show_label=False, height=180, width=180, interactive=False,
                                                 type="pil")
                                ar_md = gr.Markdown()
                            spin_btn = gr.Button("Render 360° turntable (GIF)", size="sm")
                            spin_img = gr.Image(show_label=False, type="filepath", height=240, interactive=False)
                        with gr.Accordion("📐 Accuracy report · holes · thin parts", open=False):
                            acc_md = gr.Markdown(accuracy_md(None))
                            raw_chk = gr.Checkbox(False, label="Show raw model (before refinement)", interactive=False)
                            acc_imgs = gr.Gallery(show_label=False, columns=2, height=260, allow_preview=True)
                        with gr.Accordion("✅ Validation report", open=False):
                            with gr.Row():
                                print_btn = gr.Button("🔍 Run print pre-flight", size="sm")
                                export_extra_btn = gr.Button("📦 Export OpenSCAD / Mobile LOD", size="sm")
                            print_preflight_md = gr.Markdown(visible=False)
                            d_report = gr.JSON(show_label=False)
                        with gr.Accordion("Other formats (.obj + textures, .stl, .scad, LODs)", open=False):
                            d_files = gr.File(show_label=False, file_count="multiple")
                session = gr.State(Session())
                cam = gr.State([45.0, 75.0, None])
                d_out = [chat_box, d_canvas, d_upload, chat, d_model, d_dl, d_files, d_metrics, d_title,
                         create_btn, session, versions, d_report, ar_qr, ar_md, acc_md, acc_imgs, raw_chk,
                         parts_cb, parts_html]
                turn_in = [mode, chat_box, d_canvas, d_upload, control, chat, session]
                create_btn.click(designer_turn, turn_in, d_out)
                chat_box.submit(designer_turn, turn_in, d_out)
                cand_btn.click(studio_candidates, [mode, chat_box, d_canvas, d_upload, auto_pick, chat, session,
                                                   asis_chk],
                               [cand_gallery, build_cand, cand_state] + d_out)
                cand_gallery.select(cand_select, cand_state, cand_state)
                build_cand.click(studio_build, [cand_state, chat, session], d_out)
                dec_in = [dec_text, dec_logo, dec_font, dec_color, dec_side, dec_x, dec_y, dec_size, dec_rot,
                          dec_move, chat, session]
                dec_btn.click(designer_decal, dec_in, d_out)
                for sl in (dec_x, dec_y, dec_size, dec_rot):
                    sl.release(decal_live, dec_in, d_out)
                revert_btn.click(versions_revert, [versions, chat, session], d_out)
                compare_btn.click(versions_compare, [versions, session], [compare_model, compare_note])
                spin_btn.click(studio_spin, session, spin_img)
                print_btn.click(studio_print_preflight, session, [print_preflight_md, d_report])
                export_extra_btn.click(studio_export_extra, session, d_files)
                analyse_btn.click(studio_analyse, [mode, d_canvas, d_upload, chat_box, session],
                                  [hole_img, hole_md, session])
                hole_img.select(studio_toggle_hole, session, [hole_img, session])
                proc_chk.change(studio_procedural, [proc_chk, session], session)
                raw_chk.change(studio_show_raw, [raw_chk, session], d_model)
                recolor_btn.click(studio_recolor, [parts_cb, part_colour, part_mat, chat, session], d_out)
                select_all.click(studio_select_all, session, parts_cb)
                undo_btn.click(designer_undo, [chat, session], d_out)
                new_btn.click(designer_new, None, d_out)
                mode.change(switch_mode, [mode, session], [d_canvas, d_upload, d_explain, chat_box])
                for k, btn in rot.items():
                    btn.click(rotate_view, [gr.State(k), cam, session], [d_model, cam])

            with gr.Tab("Advanced: engines & pipeline", id="sketch"):
                gr.Markdown(_status_md())
                with gr.Row():
                    with gr.Column(scale=5):
                        with gr.Tabs() as tabs:
                            with gr.Tab("Draw", id="draw"):
                                draw = gr.Sketchpad(
                                    label="Canvas (brush, eraser, clear via the toolbar)", type="numpy",
                                    canvas_size=(512, 512), height=520,
                                    brush=gr.Brush(colors=["#111111"], default_size=4, color_mode="fixed"))
                            with gr.Tab("Upload", id="upload"):
                                upload = gr.Image(label="Sketch image (PNG/JPG, photo of paper is fine)",
                                                  type="numpy", image_mode="RGBA", height=520)
                        prompt = gr.Textbox(
                            label="Prompt (optional with a sketch; enough on its own for common objects)",
                            placeholder="e.g. a brown bottle with a green holder  ·  a white teapot with a gold lid",
                            info="Shape words refine your drawing: cylindrical · hexagonal / square / octagonal · "
                                 "domed / pointed top · ribbed · tall / short · wide / slim · thick / thin / no handle "
                                 "· solid / hollow · glossy / matte. Materials: gold, silver, copper, wood, ceramic, "
                                 "glass, plastic… Colours: any colour name or #hex. Per part: '... with a black handle / "
                                 "gold cap / silver rim / wooden lid'.")
                        with gr.Row():
                            draw_for_me = gr.Button("✏️ Draw it for me (from the prompt)", variant="secondary")
                        gr.Markdown("<small>**No sketch? Just type it.** Leave the canvas empty and the prompt builds the "
                                    "object. Works for: " + ", ".join(available_names()) + ".</small>")
                        if EXAMPLES:
                            gr.Examples(EXAMPLES, inputs=[upload, prompt],
                                        label="Examples: click one, then Generate (opens in the Upload tab)")
                        material = gr.Dropdown(["auto"] + sorted(MATERIALS), value="auto", label="Material preset")
                        mesh_prov = gr.Dropdown(
                            [("Auto (AI-CAD if a Gemini key is set, else CAD / local AI)", "auto"),
                             ("AI-CAD: AI reads your sketch and rebuilds it as clean CAD (Gemini key)", "ai_cad"),
                             ("Revolve: lathe objects (mug, vase, bottle, lamp)", "revolve"),
                             ("Extrude: flat objects (sign, logo, gear, star)", "extrude"),
                             ("Inflate: organic shapes (toys, animals)", "inflate"),
                             ("TripoSR, local AI (unlimited, no internet)", "triposr_local"),
                             ("TRELLIS, cloud AI (best quality, daily quota)", "trellis"),
                             ("Offline fallback: inflate with image colours", "local")],
                            value="auto", label="Geometry engine")
                        with gr.Accordion("Advanced", open=False):
                            n_cands = gr.Slider(1, 4, value=config.NUM_CANDIDATES, step=1,
                                                label="Best-of-N candidates (scored against your sketch)")
                            img_prov = gr.Dropdown(["auto"] + list(sketch2image.PROVIDERS), value="auto",
                                                   label="Sketch-to-image provider")
                            skip_2d = gr.Checkbox(False, label="Skip generative 2D step (use offline shaded sketch)")
                            seed = gr.Number(42, precision=0, label="Seed")
                        btn = gr.Button("Generate 3D", variant="primary", size="lg")
                    with gr.Column(scale=6):
                        model = gr.Model3D(label="3D preview (drag to rotate, scroll to zoom)", height=480,
                                           clear_color=(0.95, 0.95, 0.97, 1.0))
                        stats = gr.Markdown()
                        files = gr.File(label="Download (.glb, .obj+mtl zip, .stl)", file_count="multiple")
                        to_chat = gr.Button("🎨 Refine this model in the Studio", visible=False)
                with gr.Row():
                    pre = gr.Gallery(label="1 · Preprocessing", columns=2, height=220)
                    cands = gr.Gallery(label="2 · Candidates ranked (sketch 50% · 3D-ready 30% · prompt 20%)", columns=4, height=220)
                    spin = gr.Image(label="3 · 360° turntable (rendered after the model)", height=220, type="filepath")
                log = gr.Textbox(label="Pipeline log", lines=10, max_lines=30)
                glb_state = gr.State()
                handoff = gr.State()

                btn.click(generate, [draw, upload, prompt, material, n_cands, img_prov, mesh_prov, skip_2d, seed],
                          [pre, cands, model, files, stats, log, glb_state, spin, handoff, to_chat]
                          ).then(make_gif, glb_state, spin)
                draw_for_me.click(sketch_from_prompt, prompt, [draw, tabs])

        to_chat.click(refine_in_chat, handoff, d_out + [main_tabs])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link")
    ap.add_argument("--port", type=int, default=None,
                    help="default: first free port from 7860 upwards")
    ap.add_argument("--offline", action="store_true", help="no network: Sketch-CAD + local TripoSR only")
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 lets a phone on the same Wi-Fi open AR links")
    a = ap.parse_args()
    if a.port is None and _already_running(7860):
        print("Sketch2Shape is already running: open http://127.0.0.1:7860\n"
              "(Each extra copy loads its own AI models and can run the PC out of memory. "
              "Use --port 7861 if you really want a second one.)")
        raise SystemExit(0)
    if a.offline:
        config.set_offline()
    if config.use_gpu():  # CUDA: SD1.5 + ControlNet (batched candidates) and TripoSR on GPU go first
        from src import gpu_local
        gpu_local.register()
    from src import caption_local, triposr_local

    def _warm():  # load local models in the background so the first click is fast
        try:
            caption_local._load()
            if triposr_local.available() and _free_ram_gb() >= 3.0:
                triposr_local.get_model()
        except Exception as e:  # e.g. low RAM: models load on first use instead
            print(f"Background model preload skipped ({type(e).__name__}); models will load on first use.")
    threading.Thread(target=_warm, daemon=True).start()
    demo = build().queue(default_concurrency_limit=2)
    if a.share:  # public gradio.live link (the LAN AR viewer + REST API need the default mode)
        demo.launch(server_port=a.port, share=True, theme=THEME, css=CSS, allowed_paths=[str(config.OUTPUTS)])
    else:
        import uvicorn
        from fastapi import FastAPI
        SERVER.update(port=a.port or 7860, ar=True)
        app = FastAPI(title="Sketch2Shape API", description=api.__doc__)
        api.register(app)
        app = gr.mount_gradio_app(app, demo, path="", theme=THEME, css=CSS, allowed_paths=[str(config.OUTPUTS)])
        print(f"Sketch2Shape: http://127.0.0.1:{SERVER['port']}  ·  phones on this Wi-Fi: "
              f"http://{api.lan_ip()}:{SERVER['port']}  ·  API docs: /docs")
        uvicorn.run(app, host=a.host, port=SERVER["port"], log_level="warning")
