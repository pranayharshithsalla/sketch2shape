"""Stages 2-3: auto-caption + sketch-guided image generation behind a provider router.

Providers (tried in config.IMAGE_PROVIDERS order, skipped when their key is missing):
  gemini    - Gemini image model (image+text -> image). Strong structure preservation.
  replicate - Stable Diffusion 1.5 + ControlNet scribble (jagilley/controlnet-scribble).
  hf        - FLUX.1 Kontext [dev] via Hugging Face Inference Providers (image-to-image).
  local     - offline "shaded silhouette" render from the sketch itself. Never fails.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests
from PIL import Image

from . import config
from .preprocess import Sketch
from .router import ProviderError, with_hf_tokens, with_retries

log = logging.getLogger(__name__)

GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-flash-latest")
HF_IMAGE_MODEL = os.getenv("HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-Kontext-dev")
REPLICATE_CN_MODEL = os.getenv("REPLICATE_CN_MODEL", "jagilley/controlnet-scribble")

MATERIALS = {
    "auto": "",
    "wood": "made of polished natural wood with visible grain",
    "metal": "made of brushed metal",
    "ceramic": "made of glossy glazed ceramic",
    "plastic": "made of smooth matte plastic",
    "glass": "made of frosted glass",
    "fabric": "upholstered in soft fabric",
}
NEGATIVE = ("multiple objects, text, watermark, cropped, cut off, busy background, shadow on floor, "
            "lowres, blurry, deformed, extra parts, solid filled blob, closed frame, missing handle, "
            "missing strings")


def _png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _download(url: str) -> np.ndarray:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return np.array(Image.open(io.BytesIO(r.content)).convert("RGB"))


# ---------------------------------------------------------------- captioning
def auto_caption(sketch: Sketch) -> str:
    """Name the sketched object so a blank prompt still gives ControlNet a subject."""
    # 1) local CLIP: offline, ~0.1 s, no quota
    from .caption_local import caption as clip_caption
    hit = clip_caption(sketch.clean)
    if hit:
        log.info("local CLIP caption: %s (p=%.2f)", *hit)
        return hit[0]
    # 2) cloud fallbacks: short timeout, no retries, failures go into provider cooldown
    if config.GEMINI_API_KEY:
        try:
            txt = with_retries(_gemini_call, GEMINI_TEXT_MODEL, [
                {"text": "This is a rough hand-drawn sketch of a single object. Reply with ONLY a short "
                         "noun phrase naming the object (e.g. 'a coffee mug', 'a wooden chair')."},
                {"inline_data": {"mime_type": "image/png",
                                 "data": base64.b64encode(_png_bytes(sketch.clean)).decode()}}],
                image_out=False, timeout=15, name="caption:gemini", retries=0)
            txt = re.sub(r"[^a-zA-Z0-9 ,'-]", "", txt).strip()[:60]
            if txt:
                return txt
        except Exception as e:
            log.warning("Gemini caption failed: %s", e)
    if config.HF_TOKEN:
        try:
            from huggingface_hub import InferenceClient
            from .labels import LABELS
            res = with_retries(
                InferenceClient(token=config.HF_TOKEN, timeout=15).zero_shot_image_classification,
                _png_bytes(sketch.rgb), candidate_labels=[f"a sketch of {l}" for l in LABELS[:60]],
                model="openai/clip-vit-large-patch14", name="caption:hf", retries=0)
            return res[0].label.replace("a sketch of ", "")
        except Exception as e:
            log.warning("HF zero-shot caption failed: %s", e)
    return "an object"


def names_object(prompt: str) -> bool:
    """Does the prompt name the object ('a red mug') or only describe it ('make it gold')?"""
    from .labels import LABELS
    from .sketch3d import FLAT_WORDS, LATHE_WORDS
    nouns = {l.split()[-1] for l in LABELS} | LATHE_WORDS | FLAT_WORDS
    words = set(re.findall(r"[a-z]+", prompt.lower()))
    return bool(words & (nouns | {w + "s" for w in nouns}))


def build_prompt(user_prompt: str, caption: str, material: str = "auto") -> str:
    user_prompt = user_prompt.strip()
    subject = user_prompt if names_object(user_prompt) else ", ".join(x for x in (caption, user_prompt) if x)
    mat = MATERIALS.get(material, "" if material in ("", "auto") else f"made of {material}")
    return (f"{subject}{', ' + mat if mat else ''}. Single object, product render, full object in frame, "
            f"centered, same viewpoint, pose and proportions as the sketch, soft studio lighting, "
            f"plain pure white background, high detail, realistic materials and colours")


# ---------------------------------------------------------------- providers
def _gemini_call(model: str, parts: list, image_out: bool = True, temperature: float = 0.6,
                 timeout: float | None = None):
    body = {"contents": [{"parts": parts}], "generationConfig": {"temperature": temperature}}
    if image_out:
        body["generationConfig"]["responseModalities"] = ["IMAGE", "TEXT"]
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": config.GEMINI_API_KEY, "Content-Type": "application/json"},
        json=body, timeout=timeout or config.PROVIDER_TIMEOUT_S)
    if r.status_code != 200:
        raise ProviderError(f"gemini HTTP {r.status_code}: {r.text[:200]}")
    for cand in r.json().get("candidates", []):
        for p in cand.get("content", {}).get("parts", []):
            if image_out and "inlineData" in p:
                return np.array(Image.open(io.BytesIO(base64.b64decode(p["inlineData"]["data"]))).convert("RGB"))
            if not image_out and "text" in p:
                return p["text"]
    raise ProviderError("gemini returned no " + ("image" if image_out else "text"))


def gen_gemini(sketch: Sketch, prompt: str, seed: int) -> np.ndarray:
    instr = ("Turn this rough sketch into a photorealistic 3D-rendered product image. Keep EXACTLY the "
             "same outline, parts, proportions and viewpoint as the sketch; do not add or remove parts. "
             f"Description: {prompt}")
    return _gemini_call(GEMINI_IMAGE_MODEL, [
        {"text": instr},
        {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(_png_bytes(sketch.clean)).decode()}}],
        temperature=0.4 + 0.15 * (seed % 4))


def gen_replicate(sketch: Sketch, prompt: str, seed: int) -> np.ndarray:
    import replicate
    client = replicate.Client(api_token=config.REPLICATE_API_TOKEN)
    version = client.models.get(REPLICATE_CN_MODEL).latest_version.id
    out = client.run(f"{REPLICATE_CN_MODEL}:{version}", input={
        "image": io.BytesIO(_png_bytes(sketch.clean)), "prompt": prompt, "num_samples": "1",
        "image_resolution": "512", "ddim_steps": 25, "scale": 9, "seed": seed,
        "a_prompt": "best quality, extremely detailed, product photo", "n_prompt": NEGATIVE})
    urls = [str(getattr(o, "url", o)) for o in (out if isinstance(out, list) else [out])]
    return _download(urls[-1])  # first item is the detected scribble map


def gen_hf(sketch: Sketch, prompt: str, seed: int) -> np.ndarray:
    from huggingface_hub import InferenceClient
    client = InferenceClient(provider="auto", token=config.HF_TOKEN)
    img = client.image_to_image(
        _png_bytes(sketch.clean),
        prompt=("Convert this sketch into a realistic rendered object, keep the exact shape and "
                f"viewpoint. {prompt}"),
        model=HF_IMAGE_MODEL, seed=seed)
    return np.array(img.convert("RGB"))


KONTEXT_SPACE = os.getenv("KONTEXT_SPACE", "black-forest-labs/FLUX.1-Kontext-Dev")


def gen_kontext_space(sketch: Sketch, prompt: str, seed: int) -> np.ndarray:
    """FLUX.1 Kontext [dev] on the official HF ZeroGPU Space: free with an HF_TOKEN
    (uses the same daily GPU quota as the TRELLIS Space)."""
    import tempfile
    from gradio_client import Client, handle_file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(_png_bytes(sketch.clean))
    instr = ("Turn this rough line sketch into a realistic rendered product photo. Keep exactly the "
             f"same outline, parts, proportions and viewpoint. {prompt}")
    res = with_hf_tokens(lambda tok: Client(KONTEXT_SPACE, token=tok, verbose=False).predict(
        handle_file(f.name), instr, seed, False, 2.5, 20, api_name="/infer"))
    path = res[0] if isinstance(res, (list, tuple)) else res
    path = path.get("path") or path.get("value") if isinstance(path, dict) else path
    return np.array(Image.open(path).convert("RGB"))


_COLOURS = {"red": (200, 50, 45), "blue": (50, 90, 200), "green": (60, 150, 70), "yellow": (230, 200, 60),
            "orange": (235, 130, 40), "purple": (130, 70, 170), "pink": (230, 130, 170), "black": (40, 40, 45),
            "white": (235, 235, 230), "grey": (140, 140, 145), "gray": (140, 140, 145), "brown": (130, 85, 50),
            "wood": (160, 110, 65), "wooden": (160, 110, 65), "metal": (170, 175, 185), "gold": (212, 175, 55),
            "silver": (190, 192, 198), "ceramic": (225, 220, 210), "glass": (190, 215, 225)}


def gen_local(sketch: Sketch, prompt: str, seed: int) -> np.ndarray:
    """Offline fallback: shade the filled silhouette like a lit, rounded solid and keep
    the drawn interior lines as surface detail. Gives image-to-3D a solid object
    instead of thin strokes."""
    subject = prompt.split(". Single object")[0]  # ignore the boilerplate ("white background")
    words = re.findall(r"[a-z]+", subject.lower())
    base = next((np.array(_COLOURS[w], np.float32) for w in words if w in _COLOURS),
                np.array((95, 140, 200), np.float32))
    rng = np.random.default_rng(seed)  # each seed: a different light, depth profile and tint
    if seed >= 1000:
        base = np.clip(base * rng.uniform(0.88, 1.12, 3), 0, 255)
    mask = sketch.mask.astype(np.uint8)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    h = np.clip(dist / max(dist.max(), 1), 0, 1) ** [0.5, 0.35, 0.7, 0.5][(seed // 1000) % 4]
    gy, gx = np.gradient(cv2.GaussianBlur(h, (0, 0), 3))
    n = np.dstack([-gx * 40, -gy * 40, np.ones_like(h)])
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    light = np.array([[-0.4, -0.5, 0.77], [0.4, -0.5, 0.77], [0.0, -0.7, 0.7], [-0.6, -0.2, 0.77]][(seed // 1000) % 4])
    light = light / np.linalg.norm(light)
    shade = np.clip(n @ light, 0, 1) * 0.65 + 0.35
    img = np.full(mask.shape + (3,), 255, np.float32)
    img[mask > 0] = (base[None, None] * shade[..., None])[mask > 0]
    lines = (sketch.scribble > 0) & (mask > 0)
    img[lines] *= 0.55
    return np.clip(img, 0, 255).astype(np.uint8)


PROVIDERS = {
    "gemini": (gen_gemini, lambda: bool(config.GEMINI_API_KEY)),
    "replicate": (gen_replicate, lambda: bool(config.REPLICATE_API_TOKEN)),
    "hf": (gen_hf, lambda: bool(config.HF_TOKEN)),
    "kontext_space": (gen_kontext_space, lambda: bool(config.HF_TOKEN)),
    "local": (gen_local, lambda: True),
}


BATCH: dict = {}  # provider -> fn(sketch, prompt, seeds) generating every candidate in one call


def available() -> list[str]:
    return [p for p in config.IMAGE_PROVIDERS if p in PROVIDERS and PROVIDERS[p][1]()]


def generate_candidates(sketch: Sketch, prompt: str, n: int, provider: str = "auto",
                        base_seed: int = 42) -> tuple[list[np.ndarray], str, list[str]]:
    """Generate up to n candidates in parallel with the first provider that works.
    Returns (images, provider_used, log_lines)."""
    order = available() if provider == "auto" else [provider]
    logs = []
    for name in order:
        fn = PROVIDERS[name][0]
        k = max(n, 4) if name in ("local", "sd_controlnet") else n  # local/GPU candidates are cheap
        seeds = [base_seed + 1000 * i for i in range(k)]
        if name in BATCH:
            try:
                imgs = BATCH[name](sketch, prompt, seeds)
                logs.append(f"image provider '{name}' produced {len(imgs)} candidates in one batch")
                return imgs, name, logs
            except Exception as e:
                logs.append(f"image provider '{name}' batch failed: {str(e)[:160]}")
                continue
        with ThreadPoolExecutor(max_workers=k) as ex:
            futs = [ex.submit(with_retries, fn, sketch, prompt, s, name=f"image:{name}") for s in seeds]
            imgs, errs = [], []
            for f in futs:
                try:
                    imgs.append(f.result())
                except Exception as e:
                    errs.append(str(e))
        if imgs:
            logs.append(f"image provider '{name}' produced {len(imgs)}/{k} candidates")
            return imgs, name, logs
        logs.append(f"image provider '{name}' failed: {errs[0][:160] if errs else 'unknown'}")
    raise ProviderError("All image providers failed:\n" + "\n".join(logs))
