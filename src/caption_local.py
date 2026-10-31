"""Offline auto-caption: CLIP ViT-B/32 zero-shot over an object vocabulary, on CPU.

Lets a blank prompt still choose the right engine ("a mug" -> revolve) with no network
calls or quota. Weights (~600 MB, safetensors) download once from the Hugging Face Hub.
"""
from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from .labels import LABELS

log = logging.getLogger(__name__)
MODEL_ID = "openai/clip-vit-base-patch32"
_model = _proc = _text = None


def _load():
    global _model, _proc, _text
    if _model is None:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        _model = CLIPModel.from_pretrained(MODEL_ID, use_safetensors=True).eval()
        _proc = CLIPProcessor.from_pretrained(MODEL_ID)
        prompts = [f"a simple black line drawing of {l}" for l in LABELS]
        tok = _proc.tokenizer(prompts, return_tensors="pt", padding=True)
        with torch.no_grad():  # towers + projections directly: stable across transformers versions
            t = _model.text_projection(_model.text_model(**tok).pooler_output)
        _text = t / t.norm(dim=-1, keepdim=True)
    return _model, _proc, _text


def _image_embedding(model, proc, img):
    px = proc.image_processor(images=img, return_tensors="pt")["pixel_values"]
    f = model.visual_projection(model.vision_model(pixel_values=px).pooler_output)
    return f / f.norm(dim=-1, keepdim=True)


def caption(gray_sketch: np.ndarray, min_conf: float = 0.12) -> tuple[str, float] | None:
    """-> (label, probability) or None if CLIP is unavailable or unsure."""
    try:
        import torch
        model, proc, text = _load()
        img = Image.fromarray(gray_sketch).convert("RGB")
        with torch.no_grad():
            probs = (100.0 * _image_embedding(model, proc, img) @ text.T).softmax(dim=-1)[0].numpy()
        i = int(probs.argmax())
        return (LABELS[i], float(probs[i])) if probs[i] >= min_conf else None
    except Exception as e:  # missing torch/transformers or no cached weights offline
        log.warning("local CLIP caption unavailable: %s", e)
        return None
