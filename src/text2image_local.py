"""Local text-to-image for prompt-only creation of organic objects (animals, characters):
SD-Turbo (1-step distilled Stable Diffusion) on CPU via diffusers, ~5-15 s at 512 px.
The image then goes through background removal and TripoSR like any other candidate.
Weights (~2.5 GB, fp16) download once from the Hugging Face Hub; no API, no quota.
"""
from __future__ import annotations

import logging
import os

import numpy as np

log = logging.getLogger(__name__)
MODEL_ID = os.getenv("T2I_MODEL", "stabilityai/sd-turbo")
_pipe = None


def available() -> bool:
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _get():
    global _pipe
    if _pipe is None:
        import torch
        from diffusers import AutoPipelineForText2Image
        # bfloat16 halves RAM (~2.6 GB) and runs natively on modern CPUs; fp32 needs ~5 GB
        dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16
        _pipe = AutoPipelineForText2Image.from_pretrained(MODEL_ID, variant="fp16", torch_dtype=dtype,
                                                          use_safetensors=True, low_cpu_mem_usage=True)
        _pipe.set_progress_bar_config(disable=True)
        _pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return _pipe


def generate(prompt: str, seed: int = 42, steps: int = 2) -> np.ndarray:
    """Product-style render of a single object on white, ready for rembg + image-to-3D."""
    import torch
    pipe = _get()
    full = (f"{prompt}, a single object, full body in frame, centered, 3D render, studio lighting, "
            f"plain white background, high detail")
    g = torch.Generator("cpu").manual_seed(seed)
    img = pipe(full, num_inference_steps=steps, guidance_scale=0.0, height=512, width=512,
               generator=g).images[0]
    return np.array(img.convert("RGB"))
