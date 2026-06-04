@echo off
cd /d "%~dp0"
start "" /B powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -STA -WindowStyle Hidden -File "%~dp0start_webmedic.ps1" -Hidden
exit /b 0
