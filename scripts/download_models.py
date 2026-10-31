"""Pre-fetch every model weight the app can use, so the first run (and demos) need no downloads.

    python scripts/download_models.py            # CPU set: TripoSR, CLIP, rembg  (~2.5 GB)
    python scripts/download_models.py --gpu      # + SD 1.5 + ControlNet scribble (~6 GB more)

Weights go to the standard Hugging Face cache (HF_HOME) and ~/.u2net for rembg.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="also fetch Stable Diffusion 1.5 + ControlNet")
    a = ap.parse_args()
    from huggingface_hub import snapshot_download

    print("[1/4] TripoSR source + weights (MIT)")
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "setup_local_models.py")])

    print("[2/4] CLIP ViT-B/32 for auto-caption + prompt-match ranking (MIT)")
    snapshot_download("openai/clip-vit-base-patch32",
                      allow_patterns=["*.json", "*.txt", "model.safetensors"])

    print("[3/4] rembg u2net background removal (Apache-2.0)")
    from rembg import new_session
    new_session("u2net")

    if a.gpu:
        print("[4/4] Stable Diffusion 1.5 (CreativeML OpenRAIL-M) + ControlNet v1.1 scribble (OpenRAIL)")
        snapshot_download("stable-diffusion-v1-5/stable-diffusion-v1-5",
                          allow_patterns=["*.json", "*.txt", "unet/*.safetensors", "vae/*.safetensors",
                                          "text_encoder/*.safetensors", "tokenizer/*"],
                          ignore_patterns=["*ema*", "*.bin", "*non_ema*"])
        snapshot_download("lllyasviel/control_v11p_sd15_scribble",
                          allow_patterns=["*.json", "*.safetensors"])
    else:
        print("[4/4] skipped SD 1.5 + ControlNet (GPU only; pass --gpu)")
    print("All weights cached.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
