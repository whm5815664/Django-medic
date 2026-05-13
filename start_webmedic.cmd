@echo off
setlocal
cd /d "%~dp0"
REM 双击启动：conda 环境 webmedic + Django runserver + main\Agent 下 opencode serve
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -STA -File "%~dp0start_webmedic.ps1"
endlocal
