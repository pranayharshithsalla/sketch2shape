"""Key-free GPU providers for Colab (T4) or any CUDA machine:
  sd_controlnet - Stable Diffusion 1.5 + ControlNet v1.1 scribble (diffusers)
  triposr       - TripoSR image-to-3D (vertex colours)

Call register() once; it adds both providers to the routers and puts them first.
Heavy imports (torch, diffusers, tsr) only happen here, so the CPU app never needs them.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from . import config, image2mesh, sketch2image

_pipe = None
_tsr = None


def _get_pipe():
    global _pipe
    if _pipe is None:
        import torch
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
        cn = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_scribble", torch_dtype=torch.float16)
        _pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=cn, torch_dtype=torch.float16,
            safety_checker=None).to("cuda")
        _pipe.scheduler = UniPCMultistepScheduler.from_config(_pipe.scheduler.config)
    return _pipe


def gen_sd_controlnet(sketch, prompt: str, seed: int) -> np.ndarray:
    import torch
    pipe = _get_pipe()
    cond = Image.fromarray(sketch.scribble).convert("RGB")  # white strokes on black
    out = pipe(prompt, image=cond, num_inference_steps=25, guidance_scale=7.5,
               controlnet_conditioning_scale=0.9 + 0.1 * ((seed // 1000) % 3),
               negative_prompt=sketch2image.NEGATIVE,
               generator=torch.Generator("cuda").manual_seed(seed)).images[0]
    return np.array(out.convert("RGB"))


def _get_tsr():
    global _tsr
    if _tsr is None:
        from tsr.system import TSR
        _tsr = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
        _tsr.renderer.set_chunk_size(8192)
        _tsr.to("cuda")
    return _tsr


def mesh_triposr(rgba: np.ndarray, out_dir: Path, seed: int = 0) -> str:
    import torch
    import trimesh
    model = _get_tsr()
    a = rgba[..., 3:4].astype(np.float32) / 255
    rgb = rgba[..., :3].astype(np.float32) / 255 * a + 0.5 * (1 - a)  # TripoSR expects grey bg
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    with torch.no_grad():
        codes = model([img], device="cuda")
    mesh = model.extract_mesh(codes, True, resolution=256)[0]
    # TripoSR is z-up; rotate to the glTF y-up convention
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    p = out_dir / f"triposr_{uuid.uuid4().hex[:6]}.glb"
    mesh.export(p)
    return str(p)


def register():
    sketch2image.PROVIDERS["sd_controlnet"] = (gen_sd_controlnet, lambda: True)
    image2mesh.PROVIDERS["triposr"] = (mesh_triposr, lambda: True)
    for lst, name in [(config.IMAGE_PROVIDERS, "sd_controlnet"), (config.MESH_PROVIDERS, "triposr")]:
        if name not in lst:
            lst.insert(0, name)
