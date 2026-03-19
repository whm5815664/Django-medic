$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Test-Path ".\\.venv\\Scripts\\python.exe")) {
  python -m venv .venv
}

& .\\.venv\\Scripts\\python -m pip install --upgrade pip
& .\\.venv\\Scripts\\python -m pip install -r .\\requirements.txt

if (-not (Test-Path ".\\.env")) {
  Copy-Item .\\.env.example .\\.env
}

Write-Host "OK: venv ready. Next: scripts\\migrate.ps1 then scripts\\run.ps1"

