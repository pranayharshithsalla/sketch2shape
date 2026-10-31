"""One-time setup for the local, quota-free AI path.

1. Fetches the TripoSR source (MIT, github.com/VAST-AI-Research/TripoSR) into third_party/,
   pinned to a fixed commit.
2. Downloads the TripoSR weights (~1.7 GB) into the Hugging Face cache. Model downloads
   are not subject to ZeroGPU quotas and need no token.

    python scripts/setup_local_models.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "third_party" / "TripoSR"
REPO = "https://github.com/VAST-AI-Research/TripoSR.git"
COMMIT = "107cefdc244c39106fa830359024f6a2f1c78871"


def main():
    if not (DEST / "tsr" / "system.py").exists():
        DEST.parent.mkdir(exist_ok=True)
        subprocess.check_call(["git", "clone", "--depth", "1", REPO, str(DEST)])
        subprocess.check_call(["git", "-C", str(DEST), "fetch", "--depth", "1", "origin", COMMIT])
        subprocess.check_call(["git", "-C", str(DEST), "checkout", "-q", COMMIT])
    commit = subprocess.check_output(["git", "-C", str(DEST), "rev-parse", "HEAD"], text=True).strip()
    print("TripoSR source:", DEST, "@", commit[:10])

    from huggingface_hub import hf_hub_download
    for f in ("config.yaml", "model.ckpt"):
        print("weights:", hf_hub_download("stabilityai/TripoSR", f))
    print("Done. Choose 'TripoSR, local AI' in the app, or leave the engine on Auto.")


if __name__ == "__main__":
    sys.exit(main())
