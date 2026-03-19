$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

& .\\.venv\\Scripts\\python .\\manage.py makemigrations
& .\\.venv\\Scripts\\python .\\manage.py migrate

