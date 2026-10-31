# One-command setup / run (Linux, macOS, WSL, Git Bash). Windows PowerShell: .\run.ps1
PY ?= python3
VENV = .venv
BIN = $(VENV)/bin
ifeq ($(OS),Windows_NT)
  BIN = $(VENV)/Scripts
endif

.PHONY: setup run test smoke models docker docker-gpu clean
setup:            ## create venv, install deps, fetch model weights
	$(PY) -m venv $(VENV)
	$(BIN)/python -m pip install -U pip
	$(BIN)/python -m pip install -r requirements.txt
	$(BIN)/python scripts/download_models.py
	@test -f .env || cp .env.example .env
run:              ## start the app on http://127.0.0.1:7860
	$(BIN)/python app.py
test:             ## unit tests
	$(BIN)/python -m pytest -q tests
smoke:            ## offline end-to-end: sketch -> candidates -> glb -> decal -> reload
	$(BIN)/python scripts/smoke_test.py
models:
	$(BIN)/python scripts/download_models.py
docker:
	docker compose up --build
docker-gpu:
	docker compose --profile gpu up --build app-gpu
clean:
	rm -rf outputs/run_* outputs/design_* outputs/cands_* outputs/smoke outputs/_dtest
