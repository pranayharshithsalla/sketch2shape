# One command on Windows PowerShell: set up on first run, then start Sketch2Shape.
#   .\run.ps1            (add app.py arguments after it, e.g. .\run.ps1 --offline)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path .venv)) {
    py -3 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -U pip
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe scripts\download_models.py
    if (-not (Test-Path .env)) { Copy-Item .env.example .env }
}
.\.venv\Scripts\python.exe app.py @args
