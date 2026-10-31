"""TripoSR (MIT, Stability AI + Tripo) running locally: CPU or CUDA, no API, no quota.

Weights (~1.7 GB) download once from the Hugging Face Hub (model downloads are not
quota-limited) into the HF cache. The TripoSR source lives in third_party/TripoSR
(fetched by scripts/setup_local_models.py). torchmcubes (needs a C++/CUDA build) is
replaced by a scikit-image shim so it installs with plain pip on Windows.
"""
from __future__ import annotations

import logging
import sys
import types
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from .config import ROOT

log = logging.getLogger(__name__)
TSR_DIR = ROOT / "third_party" / "TripoSR"
_model = None


def _install_mcubes_shim():
    if "torchmcubes" in sys.modules:
        return
    import torch
    from skimage.measure import marching_cubes as sk_mc

    def marching_cubes(vol, thresh):
        v, f, _, _ = sk_mc(vol.cpu().numpy(), level=thresh)
        # torchmcubes returns (x, y, z) = reversed array axes; TripoSR flips back with [2,1,0]
        return (torch.from_numpy(np.ascontiguousarray(v[:, ::-1])).float(),
                torch.from_numpy(f[:, ::-1].copy().astype(np.int64)))

    sys.modules["torchmcubes"] = types.SimpleNamespace(marching_cubes=marching_cubes)


def available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return (TSR_DIR / "tsr" / "system.py").exists()


def device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


_VIT_RENAMES = [  # transformers 4.x ViT names (checkpoint) -> transformers 5.x names
    (".encoder.layer.", ".layers."), (".attention.attention.query.", ".attention.q_proj."),
    (".attention.attention.key.", ".attention.k_proj."), (".attention.attention.value.", ".attention.v_proj."),
    (".attention.output.dense.", ".attention.o_proj."), (".intermediate.dense.", ".mlp.fc1."),
    (".output.dense.", ".mlp.fc2."),
]


def _load_tsr():
    """TSR.from_pretrained, but remapping ViT keys when the installed transformers
    uses the newer module names (same weights, renamed modules)."""
    import torch
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf
    from tsr.system import TSR
    cfg = OmegaConf.load(hf_hub_download("stabilityai/TripoSR", "config.yaml"))
    OmegaConf.resolve(cfg)
    model = TSR(cfg)
    ckpt = torch.load(hf_hub_download("stabilityai/TripoSR", "model.ckpt"), map_location="cpu",
                      weights_only=True)
    expected = set(model.state_dict())
    if not set(ckpt) <= expected:
        remapped = {}
        for k, v in ckpt.items():
            nk = k
            if k.startswith("image_tokenizer.model.encoder.layer."):
                for a, b in _VIT_RENAMES:
                    nk = nk.replace(a, b)
            remapped[nk] = v
        ckpt = remapped
    model.load_state_dict(ckpt)
    return model


def get_model():
    global _model
    if _model is None:
        import torch
        _install_mcubes_shim()
        if str(TSR_DIR) not in sys.path:
            sys.path.insert(0, str(TSR_DIR))
        torch.set_num_threads(max(1, torch.get_num_threads()))
        _model = _load_tsr()
        _model.renderer.set_chunk_size(8192 if device() == "cuda" else 32768)
        _model.to(device())
    return _model


def reconstruct(rgba: np.ndarray, out_dir: Path, resolution: int = 192) -> str:
    """rgba: background-removed object image. Returns path of a vertex-coloured GLB (y-up)."""
    import torch
    import trimesh
    model = get_model()
    # TripoSR expects the object on 50% grey, filling ~85% of the frame
    a = rgba[..., 3:4].astype(np.float32) / 255
    rgb = rgba[..., :3].astype(np.float32) / 255 * a + 0.5 * (1 - a)
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    with torch.no_grad():
        codes = model([img], device=device())
        mesh = model.extract_mesh(codes, True, resolution=resolution)[0]
    trimesh.smoothing.filter_taubin(mesh, iterations=6)  # remove marching-cubes grid noise
    # TripoSR outputs z-up, object facing +x; rotate to glTF y-up, facing the viewer (+z)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    p = out_dir / f"triposr_{uuid.uuid4().hex[:6]}.glb"
    mesh.export(p)
    return str(p)
