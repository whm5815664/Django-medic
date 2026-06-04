#Requires -Version 5.1
# WebMedic 启动器：runsslserver + main\Agent opencode serve

param([switch]$Hidden)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not $Hidden) {
    $ps1 = if ($PSCommandPath) { $PSCommandPath } else { $MyInvocation.MyCommand.Path }
    Start-Process powershell.exe -ArgumentList @(
        "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-STA",
        "-WindowStyle", "Hidden", "-File", $ps1, "-Hidden"
    ) -WindowStyle Hidden | Out-Null
    exit 0
}

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$ProjectRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$AgentDir = Join-Path $ProjectRoot "main\Agent"
$MutexName = "Global\WebMedicDjangoLauncherMutex2026"

$script:djangoPid = 0
$script:opencodePid = 0
$script:djangoPort = 0
$script:openCodePort = 0
$script:shuttingDown = $false
$script:servicesRunning = $false
$script:homeUrl = ""
$script:mutex = $null
$script:mutexOwned = $false

function Get-CondaBat {
    $base = $null
    if (Get-Command conda.exe -ErrorAction SilentlyContinue) {
        $out = & conda.exe info --base 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) { $base = ($out | Select-Object -Last 1).ToString().Trim() }
    }
    if (-not $base -and $env:CONDA_EXE -and (Test-Path $env:CONDA_EXE)) {
        $base = Split-Path (Split-Path $env:CONDA_EXE -Parent) -Parent
    }
    if (-not $base) { return $null }
    foreach ($p in @("condabin\conda.bat", "Scripts\conda.bat")) {
        $full = Join-Path $base $p
        if (Test-Path $full) { return $full }
    }
    return $null
}

function Test-PortNumber {
    param([string]$Text, [int]$Value)
    if ($Text -notmatch '^\d+$') { return $false }
    return ($Value -ge 1 -and $Value -le 65535)
}

function Get-UserPortSettings {
    param(
        [System.Windows.Forms.TextBox]$DjangoBox,
        [System.Windows.Forms.TextBox]$OpenCodeBox
    )
    $djangoPort = 0
    $openCodePort = 0
    if (-not [int]::TryParse($DjangoBox.Text.Trim(), [ref]$djangoPort) -or -not (Test-PortNumber $DjangoBox.Text.Trim() $djangoPort)) {
        throw "Django 端口无效，请输入 1–65535 的整数。"
    }
    if (-not [int]::TryParse($OpenCodeBox.Text.Trim(), [ref]$openCodePort) -or -not (Test-PortNumber $OpenCodeBox.Text.Trim() $openCodePort)) {
        throw "OpenCode 端口无效，请输入 1–65535 的整数。"
    }
    if ($djangoPort -eq $openCodePort) {
        throw "Django 与 OpenCode 端口不能相同。"
    }
    return @{ DjangoPort = $djangoPort; OpenCodePort = $openCodePort }
}

function Get-LanIPv4Addresses {
    $addrs = @()
    try {
        $addrs = @(Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
            Where-Object { $_.IPAddress -ne "127.0.0.1" -and $_.PrefixOrigin -ne "WellKnown" } |
            Select-Object -ExpandProperty IPAddress -Unique)
    } catch { }
    if ($addrs.Length -eq 0) {
        try {
            $addrs = @([System.Net.Dns]::GetHostAddresses([System.Net.Dns]::GetHostName()) |
                Where-Object { $_.AddressFamily -eq "InterNetwork" -and $_.IPAddressToString -ne "127.0.0.1" } |
                ForEach-Object { $_.IPAddressToString } |
                Select-Object -Unique)
        } catch { }
    }
    return @($addrs)
}

function Get-LanAccessLines {
    param([int]$Port)
    $ips = @(Get-LanIPv4Addresses)
    if ($ips.Length -eq 0) {
        return @("局域网：未检测到 IPv4（可用 ipconfig 查看）")
    }
    return @($ips | ForEach-Object { "局域网：https://${_}:$Port/" })
}

function Get-PortStatusText {
    param([int]$Port, [string]$Label)
    $procIds = @(Get-PidsOnPort $Port)
    if ($procIds.Count -eq 0) {
        return "$Label 端口 $Port：可用"
    }
    $details = @()
    foreach ($procId in $procIds) {
        if ($procId -le 0) { continue }
        $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
        if ($proc) {
            $details += "PID $procId ($($proc.ProcessName))"
        } else {
            $details += "PID $procId"
        }
    }
    if ($details.Count -eq 0) {
        return "$Label 端口 $Port：可用"
    }
    return "$Label 端口 $Port：已占用（$($details -join '，')）"
}

function Start-ServiceCmd {
    param([string]$Title, [string]$Line)
    $proc = Start-Process cmd.exe -ArgumentList @("/k", "title $Title && $Line") -PassThru
    return $proc.Id
}

function Stop-ServiceTree {
    param([int]$ProcessId)
    if ($ProcessId -le 0) { return }
    & taskkill.exe /PID $ProcessId /T /F 2>$null | Out-Null
}

function Get-PidsOnPort {
    param([int]$Port)
    $found = New-Object System.Collections.Generic.List[int]
    try {
        $conns = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
        foreach ($conn in $conns) {
            $procId = [int]$conn.OwningProcess
            if ($procId -gt 0) { $found.Add($procId) }
        }
    } catch { }
    if ($found.Count -eq 0) {
        netstat -ano | Select-String ":$Port\s" | ForEach-Object {
            if ($_ -match "LISTENING\s+(\d+)\s*$") {
                $procId = [int]$Matches[1]
                if ($procId -gt 0) { $found.Add($procId) }
            }
        }
    }
    return @($found | Select-Object -Unique)
}

function Test-PortInUse {
    param([int]$Port)
    return (@(Get-PidsOnPort $Port)).Length -gt 0
}

function Stop-ProcessesOnPort {
    param([int]$Port)
    foreach ($procId in @(Get-PidsOnPort $Port)) {
        Stop-ServiceTree $procId
    }
}

function Invoke-ReleasePorts {
    param(
        [int]$DjangoPort,
        [int]$OpenCodePort
    )
    $results = @()
    foreach ($item in @(
            @{ Label = "Django"; Port = $DjangoPort }
            @{ Label = "OpenCode"; Port = $OpenCodePort }
        )) {
        $procIds = @(Get-PidsOnPort $item.Port)
        if ($procIds.Length -eq 0) {
            $results += "$($item.Label) 端口 $($item.Port)：未被占用"
            continue
        }
        Stop-ProcessesOnPort $item.Port
        $results += "$($item.Label) 端口 $($item.Port)：已强制结束（PID: $($procIds -join '，')）"
    }
    if ($script:servicesRunning) {
        $script:djangoPid = 0
        $script:opencodePid = 0
        $script:servicesRunning = $false
        $script:homeUrl = ""
        $script:djangoPort = 0
        $script:openCodePort = 0
        $btnOpen.Enabled = $false
        $btnStop.Enabled = $false
        $btnStart.Enabled = $true
        $txtDjangoPort.Enabled = $true
        $txtOpenCodePort.Enabled = $true
        $lblInfo.Text = ""
    }
    return $results
}

function Invoke-StartupRollback {
    Stop-ServiceTree $script:djangoPid
    Stop-ServiceTree $script:opencodePid
    if ($script:djangoPort -gt 0) { Stop-ProcessesOnPort $script:djangoPort }
    if ($script:openCodePort -gt 0) { Stop-ProcessesOnPort $script:openCodePort }
    $script:djangoPid = 0
    $script:opencodePid = 0
    $script:djangoPort = 0
    $script:openCodePort = 0
    $script:servicesRunning = $false
    $script:homeUrl = ""
}

function Reset-StartupUi {
    $btnStart.Enabled = $true
    $btnCheckPort.Enabled = $true
    $btnReleasePort.Enabled = $true
    $txtDjangoPort.Enabled = $true
    $txtOpenCodePort.Enabled = $true
    $btnOpen.Enabled = $false
    $btnStop.Enabled = $false
    $lblInfo.Text = ""
    $lblStatus.Text = "启动失败，请修改后重试"
    $progress.Value = 0
}

function Show-ErrorBox {
    param([string]$Message)
    [System.Windows.Forms.MessageBox]::Show($Message, "WebMedic 启动器", "OK", "Error") | Out-Null
}

function Update-Progress {
    param(
        [System.Windows.Forms.ProgressBar]$Bar,
        [System.Windows.Forms.Label]$Label,
        [int]$Value,
        [string]$Text
    )
    $Bar.Value = [Math]::Min($Value, $Bar.Maximum)
    $Label.Text = $Text
    [System.Windows.Forms.Application]::DoEvents()
}

$stopAll = {
    $script:shuttingDown = $true
    Stop-ServiceTree $script:djangoPid
    Stop-ServiceTree $script:opencodePid
    if ($script:djangoPort -gt 0) { Stop-ProcessesOnPort $script:djangoPort }
    if ($script:openCodePort -gt 0) { Stop-ProcessesOnPort $script:openCodePort }
    $script:djangoPid = 0
    $script:opencodePid = 0
    $script:servicesRunning = $false
}

[System.Windows.Forms.Application]::EnableVisualStyles()

$form = New-Object System.Windows.Forms.Form
$form.Text = "WebMedic 启动器"
$form.StartPosition = "CenterScreen"
$form.FormBorderStyle = "FixedDialog"
$form.MaximizeBox = $false
$form.ClientSize = New-Object System.Drawing.Size(480, 320)

$lblDjango = New-Object System.Windows.Forms.Label
$lblDjango.Text = "Django 端口："
$lblDjango.Location = New-Object System.Drawing.Point(16, 20)
$lblDjango.Size = New-Object System.Drawing.Size(100, 24)
$form.Controls.Add($lblDjango)

$txtDjangoPort = New-Object System.Windows.Forms.TextBox
$txtDjangoPort.Text = "8000"
$txtDjangoPort.Location = New-Object System.Drawing.Point(120, 18)
$txtDjangoPort.Size = New-Object System.Drawing.Size(100, 24)
$form.Controls.Add($txtDjangoPort)

$lblOpenCode = New-Object System.Windows.Forms.Label
$lblOpenCode.Text = "OpenCode 端口："
$lblOpenCode.Location = New-Object System.Drawing.Point(240, 20)
$lblOpenCode.Size = New-Object System.Drawing.Size(110, 24)
$form.Controls.Add($lblOpenCode)

$txtOpenCodePort = New-Object System.Windows.Forms.TextBox
$txtOpenCodePort.Text = "4096"
$txtOpenCodePort.Location = New-Object System.Drawing.Point(356, 18)
$txtOpenCodePort.Size = New-Object System.Drawing.Size(100, 24)
$form.Controls.Add($txtOpenCodePort)

$btnStart = New-Object System.Windows.Forms.Button
$btnStart.Text = "启动服务"
$btnStart.Location = New-Object System.Drawing.Point(16, 54)
$btnStart.Size = New-Object System.Drawing.Size(100, 32)
$form.Controls.Add($btnStart)

$btnCheckPort = New-Object System.Windows.Forms.Button
$btnCheckPort.Text = "检查端口"
$btnCheckPort.Location = New-Object System.Drawing.Point(128, 54)
$btnCheckPort.Size = New-Object System.Drawing.Size(100, 32)
$form.Controls.Add($btnCheckPort)

$btnReleasePort = New-Object System.Windows.Forms.Button
$btnReleasePort.Text = "解除占用"
$btnReleasePort.Location = New-Object System.Drawing.Point(240, 54)
$btnReleasePort.Size = New-Object System.Drawing.Size(100, 32)
$form.Controls.Add($btnReleasePort)

$lblStatus = New-Object System.Windows.Forms.Label
$lblStatus.Location = New-Object System.Drawing.Point(16, 96)
$lblStatus.Size = New-Object System.Drawing.Size(448, 24)
$lblStatus.Text = "请设置端口后点击「启动服务」"
$form.Controls.Add($lblStatus)

$progress = New-Object System.Windows.Forms.ProgressBar
$progress.Location = New-Object System.Drawing.Point(16, 124)
$progress.Size = New-Object System.Drawing.Size(448, 22)
$progress.Minimum = 0
$progress.Maximum = 100
$form.Controls.Add($progress)

$lblInfo = New-Object System.Windows.Forms.Label
$lblInfo.Location = New-Object System.Drawing.Point(16, 154)
$lblInfo.Size = New-Object System.Drawing.Size(448, 108)
$lblInfo.Text = ""
$form.Controls.Add($lblInfo)

$btnOpen = New-Object System.Windows.Forms.Button
$btnOpen.Text = "打开页面"
$btnOpen.Location = New-Object System.Drawing.Point(16, 276)
$btnOpen.Size = New-Object System.Drawing.Size(120, 32)
$btnOpen.Enabled = $false
$form.Controls.Add($btnOpen)

$btnStop = New-Object System.Windows.Forms.Button
$btnStop.Text = "关闭服务"
$btnStop.Location = New-Object System.Drawing.Point(344, 276)
$btnStop.Size = New-Object System.Drawing.Size(120, 32)
$btnStop.Enabled = $false
$form.Controls.Add($btnStop)

$btnCheckPort.Add_Click({
    try {
        $ports = Get-UserPortSettings $txtDjangoPort $txtOpenCodePort
        $lines = @(
            (Get-PortStatusText $ports.DjangoPort "Django")
            (Get-PortStatusText $ports.OpenCodePort "OpenCode")
        )
        $inUse = (Test-PortInUse $ports.DjangoPort) -or (Test-PortInUse $ports.OpenCodePort)
        $icon = if ($inUse) { "Warning" } else { "Information" }
        $title = if ($inUse) { "端口检查：存在占用" } else { "端口检查：可用" }
        [System.Windows.Forms.MessageBox]::Show(
            ($lines -join "`r`n"),
            $title,
            "OK",
            $icon
        ) | Out-Null
        $lblStatus.Text = if ($inUse) { "端口检查完成：存在占用" } else { "端口检查完成：均可使用" }
    } catch {
        Show-ErrorBox $_.Exception.Message
    }
})

$btnReleasePort.Add_Click({
    try {
        $ports = Get-UserPortSettings $txtDjangoPort $txtOpenCodePort
        $djangoInUse = Test-PortInUse $ports.DjangoPort
        $openInUse = Test-PortInUse $ports.OpenCodePort
        if (-not $djangoInUse -and -not $openInUse) {
            [System.Windows.Forms.MessageBox]::Show(
                "Django 端口 $($ports.DjangoPort) 与 OpenCode 端口 $($ports.OpenCodePort) 均未被占用。",
                "解除占用",
                "OK",
                "Information"
            ) | Out-Null
            $lblStatus.Text = "端口均未被占用"
            return
        }
        $confirm = [System.Windows.Forms.MessageBox]::Show(
            @(
                "将强制结束占用下列端口的进程："
                "· Django：$($ports.DjangoPort)$(if ($djangoInUse) { '（占用中）' } else { '（未占用）' })"
                "· OpenCode：$($ports.OpenCodePort)$(if ($openInUse) { '（占用中）' } else { '（未占用）' })"
                ""
                "是否继续？"
            ) -join "`r`n",
            "解除占用",
            "YesNo",
            "Warning"
        )
        if ($confirm -ne "Yes") { return }

        $results = Invoke-ReleasePorts $ports.DjangoPort $ports.OpenCodePort
        [System.Windows.Forms.MessageBox]::Show(
            ($results -join "`r`n"),
            "解除占用完成",
            "OK",
            "Information"
        ) | Out-Null
        $lblStatus.Text = "已尝试解除端口占用"
    } catch {
        Show-ErrorBox $_.Exception.Message
    }
})

$btnStart.Add_Click({
    if ($script:servicesRunning) { return }

    try {
        $ports = Get-UserPortSettings $txtDjangoPort $txtOpenCodePort
        $djangoPort = $ports.DjangoPort
        $openCodePort = $ports.OpenCodePort
    } catch {
        Show-ErrorBox $_.Exception.Message
        return
    }

    $btnStart.Enabled = $false
    $btnCheckPort.Enabled = $false
    $btnReleasePort.Enabled = $false
    $txtDjangoPort.Enabled = $false
    $txtOpenCodePort.Enabled = $false
    $progress.Value = 0

    $script:djangoPort = $djangoPort
    $script:openCodePort = $openCodePort

    try {
        Update-Progress $progress $lblStatus 5 "检查是否已在运行..."
        if (-not $script:mutexOwned) {
            $script:mutex = New-Object System.Threading.Mutex($false, $MutexName)
            if (-not $script:mutex.WaitOne(0, $false)) {
                throw "启动器已在运行，请勿重复打开。"
            }
            $script:mutexOwned = $true
        }

        Update-Progress $progress $lblStatus 10 "检查端口占用..."
        if (Test-PortInUse $djangoPort) { throw "Django 端口 $djangoPort 已被占用。" }
        if (Test-PortInUse $openCodePort) { throw "OpenCode 端口 $openCodePort 已被占用。" }

        Update-Progress $progress $lblStatus 15 "检查 conda 环境..."
        $condaBat = Get-CondaBat
        if (-not $condaBat) { throw "未找到 conda，请确认已安装并加入 PATH。" }
        if (-not (Test-Path (Join-Path $ProjectRoot "manage.py"))) { throw "未找到 manage.py。" }
        if (-not (Test-Path $AgentDir)) { throw "未找到 main\Agent 目录。" }

        $prefix = "chcp 65001 >nul && CALL `"$condaBat`" activate webmedic && SET PYTHONUNBUFFERED=1 && SET PYTHONIOENCODING=utf-8"

        Update-Progress $progress $lblStatus 40 "正在启动 Django (HTTPS)..."
        $djangoCmd = "$prefix && CD /D `"$ProjectRoot`" && python manage.py runsslserver 0.0.0.0:$djangoPort"
        $script:djangoPid = Start-ServiceCmd "WebMedic-Django" $djangoCmd

        Update-Progress $progress $lblStatus 70 "正在启动 OpenCode..."
        $openCmd = "$prefix && CD /D `"$AgentDir`" && opencode serve --port $openCodePort"
        $script:opencodePid = Start-ServiceCmd "WebMedic-OpenCode" $openCmd

        $script:homeUrl = "https://127.0.0.1:$djangoPort/"
        $script:servicesRunning = $true

        Update-Progress $progress $lblStatus 100 "启动完成"
        $lanLines = Get-LanAccessLines $djangoPort
        $lblInfo.Text = @(
            "服务已就绪（HTTPS）"
            "本机：$($script:homeUrl)"
        ) + $lanLines + @(
            "OpenCode：http://127.0.0.1:$openCodePort"
            "cmd 窗口已打开；点「关闭服务」将强制结束占用端口的进程。"
        ) -join "`r`n"
        $btnOpen.Enabled = $true
        $btnStop.Enabled = $true
        $btnCheckPort.Enabled = $true
        $btnReleasePort.Enabled = $true
    } catch {
        Show-ErrorBox "启动失败，已停止并清理：`r`n$($_.Exception.Message)"
        Invoke-StartupRollback
        Reset-StartupUi
        if ($_.Exception.Message -like "*已在运行*") {
            $form.Close()
        }
    }
})

$form.Add_FormClosing({
    param($sender, $e)
    if ($script:shuttingDown -or -not $script:servicesRunning) { return }
    $ans = [System.Windows.Forms.MessageBox]::Show(
        "关闭窗口将停止所有服务，是否继续？",
        "关闭服务",
        "YesNo",
        "Question"
    )
    if ($ans -ne "Yes") { $e.Cancel = $true; return }
    & $stopAll
})

$btnOpen.Add_Click({ Start-Process $script:homeUrl })
$btnStop.Add_Click({ & $stopAll; $form.Close() })

try {
    [void]$form.ShowDialog()
} finally {
    if ($script:mutex -and $script:mutexOwned) {
        try { $script:mutex.ReleaseMutex() } catch { }
    }
    if ($script:mutex) {
        try { $script:mutex.Dispose() } catch { }
    }
}
