$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

& .\\.venv\\Scripts\\python .\\manage.py runserver

