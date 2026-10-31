#!/usr/bin/env bash
# One command: set up on first run, then start Sketch2Shape.  Usage: ./run.sh [app.py args]
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  make setup PY="${PYTHON:-python3}"
fi
if [ -x .venv/bin/python ]; then exec .venv/bin/python app.py "$@"; else exec .venv/Scripts/python.exe app.py "$@"; fi
