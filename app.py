"""Sketch2Shape: Gradio UI. Draw, upload or just type -> textured, validated 3D asset."""
from __future__ import annotations

import argparse
import re
import threading
import traceback

import gradio as gr
import numpy as np

from src import config, image2mesh, sketch2image
from src.pipeline import StageError, run
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
    )
    pre = [(r.images["sketch_clean"], "cleaned sketch"), (r.images["sketch_mask"], "filled silhouette")]
    return (pre, r.candidates, r.glb, [r.glb, r.obj_zip, r.stl], stats,
            "\n".join(r.logs) + f"\n\nSaved to {r.run_dir}", r.glb, None)


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


def make_gif(glb):
    """Runs after the model is shown, so the spin preview never delays the main result."""
    if not glb:
        return None
    try:
        return turntable_gif(glb, str(glb).replace("model.glb", "spin.gif"))
    except Exception:
        traceback.print_exc()
        return None


def build() -> gr.Blocks:
    with gr.Blocks(title="Sketch2Shape") as demo:
        gr.Markdown("# ✏️ → 🧊 Sketch2Shape\nDraw or upload a sketch, **or just type what you want**, "
                    "and get a textured, validated 3D asset (.glb / .obj / .stl).")
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
                    [("Auto (CAD for lathe/flat objects, else best available AI)", "auto"),
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
        with gr.Row():
            pre = gr.Gallery(label="1 · Preprocessing", columns=2, height=220)
            cands = gr.Gallery(label="2 · Candidates ranked by Sketch Fidelity Score", columns=4, height=220)
            spin = gr.Image(label="3 · 360° turntable (rendered after the model)", height=220, type="filepath")
        log = gr.Textbox(label="Pipeline log", lines=10, max_lines=30)
        glb_state = gr.State()

        btn.click(generate, [draw, upload, prompt, material, n_cands, img_prov, mesh_prov, skip_2d, seed],
                  [pre, cands, model, files, stats, log, glb_state, spin]
                  ).then(make_gif, glb_state, spin)
        draw_for_me.click(sketch_from_prompt, prompt, [draw, tabs])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link")
    ap.add_argument("--port", type=int, default=None,
                    help="default: first free port from 7860 upwards")
    ap.add_argument("--offline", action="store_true", help="no network: Sketch-CAD + local TripoSR only")
    a = ap.parse_args()
    if a.offline:
        config.set_offline()
    from src import caption_local, triposr_local

    def _warm():  # load local models in the background so the first click is fast
        caption_local._load()
        if triposr_local.available():
            triposr_local.get_model()
    threading.Thread(target=_warm, daemon=True).start()
    build().queue(default_concurrency_limit=2).launch(
        server_port=a.port, share=a.share, theme=gr.themes.Soft(), allowed_paths=[str(config.OUTPUTS)])
