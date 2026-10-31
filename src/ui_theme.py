"""Sketch2Shape visual identity: deep cosmic navy surfaces, electric-cyan accents and glow,
a dark grid sketchpad. Theme + CSS + small HTML helpers for the Gradio UI."""
from __future__ import annotations

import gradio as gr
import numpy as np

BG_MAIN = "#070B14"
BG_SURFACE = "#0E1626"
BG_ELEVATED = "#131D36"
BORDER = "#1C2B4E"
CYAN = "#00E5FF"
CYAN_HOVER = "#33EBFF"
CYAN_DARK = "#00B4D8"
ON_CYAN = "#05101E"
TEXT = "#F0F6FC"
TEXT_2 = "#94A3B8"
TEXT_MUTED = "#475569"
GRID_BG = "#0B1120"

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.cyan, neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill=BG_MAIN, body_background_fill_dark=BG_MAIN,
    body_text_color=TEXT, body_text_color_dark=TEXT,
    body_text_color_subdued=TEXT_2, body_text_color_subdued_dark=TEXT_2,
    background_fill_primary=BG_SURFACE, background_fill_primary_dark=BG_SURFACE,
    background_fill_secondary=BG_ELEVATED, background_fill_secondary_dark=BG_ELEVATED,
    block_background_fill=BG_SURFACE, block_background_fill_dark=BG_SURFACE,
    block_border_color=BORDER, block_border_color_dark=BORDER, block_border_width="1px",
    block_label_background_fill=BG_ELEVATED, block_label_background_fill_dark=BG_ELEVATED,
    block_label_text_color=TEXT_2, block_label_text_color_dark=TEXT_2,
    block_title_text_color=TEXT, block_title_text_color_dark=TEXT,
    border_color_primary=BORDER, border_color_primary_dark=BORDER,
    input_background_fill=BG_ELEVATED, input_background_fill_dark=BG_ELEVATED,
    input_border_color=BORDER, input_border_color_dark=BORDER,
    input_placeholder_color=TEXT_MUTED, input_placeholder_color_dark=TEXT_MUTED,
    button_primary_background_fill=CYAN, button_primary_background_fill_dark=CYAN,
    button_primary_background_fill_hover=CYAN_HOVER, button_primary_background_fill_hover_dark=CYAN_HOVER,
    button_primary_text_color=ON_CYAN, button_primary_text_color_dark=ON_CYAN,
    button_secondary_background_fill=BG_ELEVATED, button_secondary_background_fill_dark=BG_ELEVATED,
    button_secondary_background_fill_hover="#1A2747", button_secondary_background_fill_hover_dark="#1A2747",
    button_secondary_text_color=TEXT, button_secondary_text_color_dark=TEXT,
    button_secondary_border_color=BORDER, button_secondary_border_color_dark=BORDER,
    color_accent=CYAN, color_accent_soft="rgba(0,229,255,0.12)",
    color_accent_soft_dark="rgba(0,229,255,0.12)",
    link_text_color=CYAN, link_text_color_dark=CYAN,
    shadow_drop="none", shadow_drop_lg="none",
)

CSS = f"""
:root, .dark {{ --bg-main:{BG_MAIN}; --bg-surface:{BG_SURFACE}; --bg-elev:{BG_ELEVATED};
  --border:{BORDER}; --cyan:{CYAN}; --cyan-dark:{CYAN_DARK}; --text:{TEXT}; --text-2:{TEXT_2};
  --muted:{TEXT_MUTED}; --glow:0 0 20px rgba(0,229,255,0.25); }}
body, gradio-app, .gradio-container {{ background:var(--bg-main) !important; color:var(--text); }}
.gradio-container {{ max-width:1440px !important; margin:auto; }}
footer {{ display:none !important; }}

/* top bar */
#topbar {{ align-items:center; padding:14px 20px; margin-bottom:14px; border:1px solid var(--border);
  border-radius:16px; background:linear-gradient(180deg,#0E1626 0%,#0A1120 100%); }}
#logo {{ display:flex; align-items:center; gap:12px; }}
#logo .mark {{ width:38px; height:38px; border-radius:11px; display:grid; place-items:center;
  background:rgba(0,229,255,0.12); border:1px solid rgba(0,229,255,0.35); box-shadow:var(--glow);
  font-size:20px; }}
#logo .name {{ font-size:22px; font-weight:800; letter-spacing:-0.02em; color:var(--text); }}
#logo .name b {{ color:var(--cyan); }}
#logo .tag {{ font-size:12px; color:var(--text-2); margin-top:2px; }}
#modes {{ background:transparent !important; border:none !important; }}
#modes .wrap {{ gap:8px; justify-content:flex-end; flex-wrap:nowrap !important; }}
#modes label {{ white-space:nowrap; }}
#modes label {{ border:1px solid var(--border) !important; background:var(--bg-elev) !important;
  border-radius:999px !important; padding:9px 16px !important; color:var(--text-2) !important;
  transition:all .15s ease; cursor:pointer; }}
#modes label:hover {{ border-color:var(--cyan-dark) !important; color:var(--text) !important; }}
#modes label.selected, #modes label:has(input:checked) {{ background:rgba(0,229,255,0.12) !important;
  border-color:var(--cyan) !important; color:var(--cyan) !important; box-shadow:var(--glow); }}
#modes input[type=radio] {{ display:none; }}

/* panels */
.panel {{ background:var(--bg-surface) !important; border:1px solid var(--border) !important;
  border-radius:18px !important; padding:16px !important; }}
.panel-title {{ display:flex; justify-content:space-between; align-items:center; margin:0 0 10px 2px; }}
.panel-title h3 {{ margin:0; font-size:15px; font-weight:700; letter-spacing:.02em; color:var(--text); }}
.panel-title span {{ font-size:12px; color:var(--text-2); }}
.badge {{ display:inline-block; padding:3px 10px; border-radius:999px; font-size:11px; font-weight:600;
  color:var(--cyan); background:rgba(0,229,255,0.10); border:1px solid rgba(0,229,255,0.35); }}

/* sketchpad: dark grid */
#sketchpad, #sketchpad .image-container, #sketchpad .container {{ background-color:{GRID_BG} !important;
  background-image:linear-gradient(rgba(0,229,255,0.05) 1px,transparent 1px),
                   linear-gradient(90deg,rgba(0,229,255,0.05) 1px,transparent 1px) !important;
  background-size:24px 24px !important; border-radius:14px !important; }}
#sketchpad {{ border:1px solid var(--border) !important; }}

/* buttons */
#create-btn {{ font-size:17px !important; font-weight:800 !important; padding:14px !important;
  border-radius:14px !important; box-shadow:0 0 24px rgba(0,229,255,0.35) !important;
  letter-spacing:.01em; }}
#create-btn:hover {{ box-shadow:0 0 34px rgba(0,229,255,0.55) !important; }}
#dl-btn {{ font-weight:700 !important; border-radius:12px !important; box-shadow:var(--glow) !important; }}
.ctl button, button.ctl {{ min-width:0 !important; border-radius:10px !important; }}

/* 3D viewport */
#viewport {{ background:var(--bg-elev) !important; border-radius:14px !important; }}

/* metrics */
.metrics {{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:6px 0 4px; }}
.metric {{ background:var(--bg-elev); border:1px solid var(--border); border-radius:14px; padding:12px 14px; }}
.metric .v {{ font-size:24px; font-weight:800; color:var(--text); line-height:1.1; }}
.metric .v.ok {{ color:var(--cyan); text-shadow:0 0 12px rgba(0,229,255,0.45); }}
.metric .v.no {{ color:#F59E0B; }}
.metric .k {{ font-size:11px; color:var(--text-2); margin-top:4px; text-transform:uppercase; letter-spacing:.06em; }}

/* chat */
#chat .message {{ border-radius:12px !important; }}
textarea, input {{ color:var(--text) !important; }}
.tab-nav button.selected, button[role=tab][aria-selected=true] {{ color:var(--cyan) !important;
  border-color:var(--cyan) !important; }}
"""

LOGO_HTML = """<div id="logo"><div class="mark">✏️</div><div><div class="name">Sketch<b>2</b>Shape</div>
<div class="tag">Sketch, photo or words → refined, printable 3D</div></div></div>"""

MODES = ["✏️ Draw rough sketch", "📷 Upload an image", "💬 Explain what you need"]


def grid_canvas(size: int = 512) -> np.ndarray:
    """Dark grid background for the sketchpad (RGBA). Strokes live on their own layer, so the
    grid never ends up in the sketch that is read."""
    bg = np.empty((size, size, 4), np.uint8)
    bg[...] = (11, 17, 32, 255)
    line = np.array([11, 17, 32]) * 0.95 + np.array([0, 229, 255]) * 0.05
    bg[::24, :, :3] = line
    bg[:, ::24, :3] = line
    return bg


def panel_title(title: str, note: str = "") -> str:
    return f'<div class="panel-title"><h3>{title}</h3><span>{note}</span></div>'


def metrics_html(accuracy: float | None, print_ready: bool | None, parts: int | None,
                 seconds: float | None) -> str:
    acc = "—" if accuracy is None else f"{round(accuracy * 100)}%"
    pr = "—" if print_ready is None else ("Yes" if print_ready else "No")
    prc = "" if print_ready is None else ("ok" if print_ready else "no")
    accc = "ok" if (accuracy or 0) >= 0.75 else ""
    return (f'<div class="metrics">'
            f'<div class="metric"><div class="v {accc}">{acc}</div><div class="k">Accuracy</div></div>'
            f'<div class="metric"><div class="v {prc}">{pr}</div><div class="k">Print ready</div></div>'
            f'<div class="metric"><div class="v">{parts if parts is not None else "—"}</div><div class="k">Parts</div></div>'
            f'<div class="metric"><div class="v">{f"{seconds:.1f}s" if seconds else "—"}</div><div class="k">Build time</div></div>'
            f'</div>')
