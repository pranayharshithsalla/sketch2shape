"""Central configuration: paths, API keys and tunables (all overridable via .env)."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Set OUTPUT_DIR outside OneDrive/Dropbox if a sync client locks freshly written files
OUTPUTS = Path(os.getenv("OUTPUT_DIR") or ROOT / "outputs")
EXAMPLES = ROOT / "examples"
OUTPUTS.mkdir(parents=True, exist_ok=True)

# HF_TOKEN may hold several comma-separated tokens (e.g. one per teammate); ZeroGPU
# calls rotate to the next token when one runs out of daily quota.
HF_TOKENS = [t.strip() for t in (os.getenv("HF_TOKEN") or "").split(",") if t.strip()]
HF_TOKEN = HF_TOKENS[0] if HF_TOKENS else None
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN") or None
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or None

# Provider order; the router skips providers whose keys are missing.
IMAGE_PROVIDERS = [p.strip() for p in os.getenv(
    "IMAGE_PROVIDERS", "kontext_space,gemini,replicate,hf,local").split(",") if p.strip()]
# Cloud TRELLIS first when quota allows (best quality); local TripoSR is the unlimited
# workhorse; "local" inflation is the last resort that never fails.
MESH_PROVIDERS = [p.strip() for p in os.getenv(
    "MESH_PROVIDERS", "trellis,hunyuan3d,triposr_local,replicate_trellis,local").split(",") if p.strip()]

OFFLINE = {"on": False}


def set_offline():
    """No network: Sketch-CAD + local TripoSR + offline inflation only (no AI planner)."""
    OFFLINE["on"] = True
    IMAGE_PROVIDERS[:] = ["local"]
    MESH_PROVIDERS[:] = ["triposr_local", "local"]


# DEVICE: auto | cuda | cpu. With a CUDA GPU, SD1.5+ControlNet and TripoSR-on-GPU are
# registered first (see gpu_local.py); on CPU the app uses the hosted/CPU fallbacks.
DEVICE = os.getenv("DEVICE", "auto").lower()


def use_gpu() -> bool:
    if DEVICE == "cpu":
        return False
    from .gpu_local import cuda_ok
    return cuda_ok()


# Each hosted candidate costs ~25 s of HF ZeroGPU quota; 2 balances fidelity vs quota
NUM_CANDIDATES = int(os.getenv("NUM_CANDIDATES", "2"))
WORK_SIZE = 512                 # side of the square working canvas
PROVIDER_TIMEOUT_S = int(os.getenv("PROVIDER_TIMEOUT_S", "240"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))

# Structural accuracy (0.4 IoU + 0.3 thin-feature recall + 0.3 hole match): below the target the
# model is carved/extended to the sketch, then rebuilt with another engine (best result kept)
ACCURACY_TARGET = float(os.getenv("ACCURACY_TARGET", "0.9"))
ACCURACY_RETRY = os.getenv("ACCURACY_RETRY", "0") != "0"  # backend switch costs a 2nd full build (60-100 s on CPU): opt-in

# Fidelity score weights (tuned on examples/, see REPORT.md)
FIDELITY_W_SIL = 0.6
FIDELITY_W_EDGE = 0.4
FIDELITY_CHAMFER_DIV = 20.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
