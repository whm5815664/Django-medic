#Requires -Version 5.1
# WebMedic 一键启动：Django（与 .vscode/launch.json 中 Django 调试配置一致）+ main\Agent 下 OpenCode serve
# 关闭本窗口将尝试结束已启动的相关进程树。

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

if ($PSScriptRoot) {
    $ScriptRoot = $PSScriptRoot
} else {
    $ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
}
$ProjectRoot = $ScriptRoot
$AgentDir = Join-Path $ProjectRoot "main\Agent"
$MutexName = "Global\WebMedicDjangoLauncherMutex2026"
$mutex = $null
$mutexOwned = $false
$djangoCmdPid = 0
$opencodeCmdPid = 0

function Get-CondaBatPath {
    $condaBase = $null
    $condaExe = Get-Command conda.exe -ErrorAction SilentlyContinue
    if ($condaExe) {
        $out = & conda.exe info --base 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) {
            $condaBase = ($out | Select-Object -Last 1).ToString().Trim()
        }
    }
    if (-not $condaBase -and $env:CONDA_EXE -and (Test-Path $env:CONDA_EXE)) {
        $condaBase = Split-Path (Split-Path $env:CONDA_EXE -Parent) -Parent
    }
    if (-not $condaBase) { return $null }
    $candidates = @(
        (Join-Path $condaBase "condabin\conda.bat"),
        (Join-Path $condaBase "Scripts\conda.bat")
    )
    foreach ($p in $candidates) {
        if (Test-Path $p) { return $p }
    }
    return $null
}

function Stop-ProcessTreeBestEffort {
    param([int]$ProcessId)
    if ($ProcessId -le 0) { return }
    try {
        & taskkill.exe /PID $ProcessId /T /F 2>$null | Out-Null
    } catch {
        try { Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue } catch { }
    }
}

try {
    $mutex = New-Object System.Threading.Mutex($false, $MutexName)
    $owned = $false
    try {
        $owned = $mutex.WaitOne(0, $false)
    } catch {
        $owned = $false
    }
    if (-not $owned) {
        try { $mutex.Dispose() } catch { }
        $mutex = $null
        [System.Windows.Forms.MessageBox]::Show(
            "WebMedic 启动器已在运行（或上次未正常释放锁）。`n`n若已无服务在跑，可稍等再试，或结束残留的 PowerShell 窗口后重试。",
            "已在运行",
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Information
        ) | Out-Null
        exit 0
    }
    $mutexOwned = $true

    $condaBat = Get-CondaBatPath
    if (-not $condaBat) {
        [System.Windows.Forms.MessageBox]::Show(
            "未找到 conda（conda.exe / conda.bat）。请先安装 Anaconda/Miniconda 并确保 conda 在 PATH 中。",
            "启动失败",
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Error
        ) | Out-Null
        exit 1
    }

    if (-not (Test-Path (Join-Path $ProjectRoot "manage.py"))) {
        [System.Windows.Forms.MessageBox]::Show(
            "未在项目根目录找到 manage.py。",
            "启动失败",
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Error
        ) | Out-Null
        exit 1
    }
    if (-not (Test-Path $AgentDir)) {
        [System.Windows.Forms.MessageBox]::Show(
            "未找到目录：main\Agent",
            "启动失败",
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Error
        ) | Out-Null
        exit 1
    }

    # 与 .vscode/launch.json「Python Debugger: Django」一致：runserver 127.0.0.1:8000，PYTHONUNBUFFERED=1
    $djangoLine = "CALL `"$condaBat`" activate webmedic && SET PYTHONUNBUFFERED=1 && CD /D `"$ProjectRoot`" && python manage.py runserver 127.0.0.1:8000"
    $p1 = Start-Process -FilePath "cmd.exe" -ArgumentList @("/c", $djangoLine) -WorkingDirectory $ProjectRoot `
        -WindowStyle Minimized -PassThru
    $djangoCmdPid = $p1.Id

    # 用户步骤：在 main\Agent 下执行 opencode serve（原文 opencde 为笔误）
    $opencodeLine = "CALL `"$condaBat`" activate webmedic && CD /D `"$AgentDir`" && opencode serve --port 4096"
    $p2 = Start-Process -FilePath "cmd.exe" -ArgumentList @("/c", $opencodeLine) -WorkingDirectory $AgentDir `
        -WindowStyle Minimized -PassThru
    $opencodeCmdPid = $p2.Id

    [System.Windows.Forms.Application]::EnableVisualStyles()

    $form = New-Object System.Windows.Forms.Form
    $form.Text = "WebMedic 启动器"
    $form.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterScreen
    $form.FormBorderStyle = [System.Windows.Forms.FormBorderStyle]::FixedDialog
    $form.MaximizeBox = $false
    $form.MinimizeBox = $true
    $form.ClientSize = New-Object System.Drawing.Size(460, 200)

    $lbl = New-Object System.Windows.Forms.Label
    $lbl.AutoSize = $false
    $lbl.Dock = [System.Windows.Forms.DockStyle]::Top
    $lbl.Height = 120
    $lbl.Padding = New-Object System.Windows.Forms.Padding(12, 12, 12, 0)
    $lbl.Text = "服务已启动：`r`n· Django 开发服：http://127.0.0.1:8000`r`n· OpenCode：http://127.0.0.1:4096`r`n`r`n关闭本窗口将结束上述两个命令行进程及其子进程。"
    $form.Controls.Add($lbl)

    $btn = New-Object System.Windows.Forms.Button
    $btn.Text = "关闭全部并退出"
    $btn.Width = 160
    $btn.Height = 32
    $btn.Anchor = [System.Windows.Forms.AnchorStyles]::Bottom -bor [System.Windows.Forms.AnchorStyles]::Right
    $btn.Location = New-Object System.Drawing.Point(260, 140)
    $form.Controls.Add($btn)

    $form.Add_FormClosed({
            Stop-ProcessTreeBestEffort -ProcessId $djangoCmdPid
            Stop-ProcessTreeBestEffort -ProcessId $opencodeCmdPid
        })
    $btn.Add_Click({ $form.Close() })

    [void]$form.ShowDialog()
} finally {
    if ($mutex -and $mutexOwned) {
        try { $mutex.ReleaseMutex() } catch { }
    }
    if ($mutex) {
        try { $mutex.Dispose() } catch { }
    }
}
