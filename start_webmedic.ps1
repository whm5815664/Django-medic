#Requires -Version 5.1
# WebMedic 启动器：数据库检查 + Django / OpenCode 独立启停

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
$ConfigPath = Join-Path $ProjectRoot "launcher_config.json"
$MutexName = "Global\WebMedicDjangoLauncherMutex2026"

$script:djangoPid = 0
$script:opencodePid = 0
$script:djangoPort = 0
$script:openCodePort = 0
$script:djangoRunning = $false
$script:opencodeRunning = $false
$script:shuttingDown = $false
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

function Get-SinglePort {
    param(
        [System.Windows.Forms.TextBox]$PortBox,
        [string]$Label
    )
    $port = 0
    if (-not [int]::TryParse($PortBox.Text.Trim(), [ref]$port) -or -not (Test-PortNumber $PortBox.Text.Trim() $port)) {
        throw "$Label 端口无效，请输入 1–65535 的整数。"
    }
    return $port
}

function Get-CondaEnvName {
    param([System.Windows.Forms.TextBox]$EnvBox)
    $name = $EnvBox.Text.Trim()
    if (-not $name) { throw "Anaconda 环境包名不能为空。" }
    return $name
}

function Get-DatabaseSettings {
    param(
        [System.Windows.Forms.TextBox]$HostBox,
        [System.Windows.Forms.TextBox]$PortBox,
        [System.Windows.Forms.TextBox]$NameBox,
        [System.Windows.Forms.TextBox]$PasswordBox
    )
    $dbPort = 0
    $hostText = $HostBox.Text.Trim()
    if (-not $hostText) { throw "数据库 IP 不能为空。" }
    if (-not [int]::TryParse($PortBox.Text.Trim(), [ref]$dbPort) -or -not (Test-PortNumber $PortBox.Text.Trim() $dbPort)) {
        throw "数据库端口无效，请输入 1–65535 的整数。"
    }
    $dbName = $NameBox.Text.Trim()
    if (-not $dbName) { throw "数据库名不能为空。" }
    return @{
        Host = $hostText
        Port = $dbPort
        Name = $dbName
        Password = $PasswordBox.Text.Trim()
    }
}

function Escape-CmdEnvValue {
    param([string]$Value)
    return $Value.Replace('"', '""')
}

function Get-DatabaseEnvPrefix {
    param([hashtable]$Db)
    $pairs = [ordered]@{
        DB_HOST = ([string]$Db.Host).Trim()
        DB_PORT = ([string]$Db.Port).Trim()
        DB_NAME = ([string]$Db.Name).Trim()
        DB_USER = "root"
        DB_PASSWORD = ([string]$Db.Password).Trim()
    }
    return ($pairs.GetEnumerator() | ForEach-Object {
        $val = Escape-CmdEnvValue $_.Value
        "SET `"$($_.Key)=$val`""
    }) -join " && "
}

function Write-DatabaseDotEnv {
    param(
        [hashtable]$Db,
        [string]$Root
    )
    $envPath = Join-Path $Root ".env"
    $entries = [ordered]@{
        DB_HOST = ([string]$Db.Host).Trim()
        DB_PORT = ([string]$Db.Port).Trim()
        DB_NAME = ([string]$Db.Name).Trim()
        DB_USER = "root"
        DB_PASSWORD = ([string]$Db.Password).Trim()
    }
    $existing = [ordered]@{}
    if (Test-Path $envPath) {
        Get-Content $envPath -Encoding UTF8 | ForEach-Object {
            $line = $_.TrimEnd()
            if ($line -match '^\s*([^#=]+?)=(.*)$') {
                $existing[$Matches[1].Trim()] = $Matches[2].Trim()
            }
        }
    }
    foreach ($key in $entries.Keys) {
        $existing[$key] = $entries[$key]
    }
    $content = ($existing.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join "`n"
    [System.IO.File]::WriteAllText($envPath, "$content`n", [System.Text.UTF8Encoding]::new($false))
}

function Save-LauncherConfig {
    param([hashtable]$Config)
    $json = $Config | ConvertTo-Json -Compress
    [System.IO.File]::WriteAllText($ConfigPath, $json, [System.Text.UTF8Encoding]::new($false))
}

function Load-LauncherConfig {
    if (-not (Test-Path $ConfigPath)) { return $null }
    try {
        return (Get-Content $ConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json)
    } catch {
        return $null
    }
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

function Get-CondaCmdPrefix {
    param([string]$CondaEnv)
    $condaBat = Get-CondaBat
    if (-not $condaBat) { throw "未找到 conda，请确认已安装并加入 PATH。" }
    return "chcp 65001 >nul && CALL `"$condaBat`" activate $CondaEnv && SET PYTHONUNBUFFERED=1 && SET PYTHONIOENCODING=utf-8"
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

function Test-AnyServiceRunning {
    return ($script:djangoRunning -or $script:opencodeRunning)
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

function Update-BottomButtons {
    $btnOpen.Enabled = $script:djangoRunning
    $btnStop.Enabled = (Test-AnyServiceRunning)
}

function Ensure-LauncherMutex {
    if ($script:mutexOwned) { return }
    $script:mutex = New-Object System.Threading.Mutex($false, $MutexName)
    if (-not $script:mutex.WaitOne(0, $false)) {
        throw "启动器已在运行，请勿重复打开。"
    }
    $script:mutexOwned = $true
}

function Stop-DjangoService {
    Stop-ServiceTree $script:djangoPid
    if ($script:djangoPort -gt 0) { Stop-ProcessesOnPort $script:djangoPort }
    $script:djangoPid = 0
    $script:djangoPort = 0
    $script:djangoRunning = $false
    $script:homeUrl = ""
}

function Stop-OpenCodeService {
    Stop-ServiceTree $script:opencodePid
    if ($script:openCodePort -gt 0) { Stop-ProcessesOnPort $script:openCodePort }
    $script:opencodePid = 0
    $script:openCodePort = 0
    $script:opencodeRunning = $false
}

function Reset-DjangoUi {
    $btnStartDjango.Enabled = $true
    $btnCheckDjangoPort.Enabled = $true
    $btnReleaseDjangoPort.Enabled = $true
    $txtDjangoPort.Enabled = $true
    $lblDjangoStatus.Text = "网页服务未启动"
    $progressDjango.Value = 0
}

function Reset-OpenCodeUi {
    $btnStartOpenCode.Enabled = $true
    $btnCheckOpenCodePort.Enabled = $true
    $btnReleaseOpenCodePort.Enabled = $true
    $txtOpenCodePort.Enabled = $true
    $lblOpenCodeStatus.Text = "智能体服务未启动"
    $progressOpenCode.Value = 0
}

function Invoke-ReleaseSinglePort {
    param(
        [int]$Port,
        [string]$Label,
        [bool]$ServiceRunning,
        [scriptblock]$StopOwnedService
    )
    if ($ServiceRunning) {
        & $StopOwnedService
        return "$Label 端口 $Port：已停止服务"
    }
    $procIds = @(Get-PidsOnPort $Port)
    if ($procIds.Length -eq 0) {
        return "$Label 端口 $Port：未被占用"
    }
    Stop-ProcessesOnPort $Port
    return "$Label 端口 $Port：已强制结束（PID: $($procIds -join '，')）"
}

$stopAll = {
    $script:shuttingDown = $true
    if ($script:djangoRunning) { Stop-DjangoService; Reset-DjangoUi }
    if ($script:opencodeRunning) { Stop-OpenCodeService; Reset-OpenCodeUi }
    Update-BottomButtons
}

[System.Windows.Forms.Application]::EnableVisualStyles()

$form = New-Object System.Windows.Forms.Form
$form.Text = "WebMedic 启动器"
$form.StartPosition = "CenterScreen"
$form.FormBorderStyle = "FixedDialog"
$form.MaximizeBox = $false
$form.ClientSize = New-Object System.Drawing.Size(540, 420)

$grpDb = New-Object System.Windows.Forms.GroupBox
$grpDb.Text = "数据库"
$grpDb.Location = New-Object System.Drawing.Point(12, 12)
$grpDb.Size = New-Object System.Drawing.Size(516, 118)
$form.Controls.Add($grpDb)

$lblDbHost = New-Object System.Windows.Forms.Label
$lblDbHost.Text = "数据库 IP:"
$lblDbHost.Location = New-Object System.Drawing.Point(12, 26)
$lblDbHost.Size = New-Object System.Drawing.Size(72, 20)
$grpDb.Controls.Add($lblDbHost)

$txtDbHost = New-Object System.Windows.Forms.TextBox
$txtDbHost.Text = "127.0.0.1"
$txtDbHost.Location = New-Object System.Drawing.Point(84, 24)
$txtDbHost.Size = New-Object System.Drawing.Size(80, 22)
$grpDb.Controls.Add($txtDbHost)

$lblDbPort = New-Object System.Windows.Forms.Label
$lblDbPort.Text = "数据库端口:"
$lblDbPort.Location = New-Object System.Drawing.Point(172, 26)
$lblDbPort.Size = New-Object System.Drawing.Size(72, 20)
$grpDb.Controls.Add($lblDbPort)

$txtDbPort = New-Object System.Windows.Forms.TextBox
$txtDbPort.Text = "3306"
$txtDbPort.Location = New-Object System.Drawing.Point(248, 24)
$txtDbPort.Size = New-Object System.Drawing.Size(52, 22)
$grpDb.Controls.Add($txtDbPort)

$lblDbName = New-Object System.Windows.Forms.Label
$lblDbName.Text = "数据库名:"
$lblDbName.Location = New-Object System.Drawing.Point(308, 26)
$lblDbName.Size = New-Object System.Drawing.Size(60, 20)
$grpDb.Controls.Add($lblDbName)

$txtDbName = New-Object System.Windows.Forms.TextBox
$txtDbName.Text = "web_medic"
$txtDbName.Location = New-Object System.Drawing.Point(372, 24)
$txtDbName.Size = New-Object System.Drawing.Size(132, 22)
$grpDb.Controls.Add($txtDbName)

$lblDbPassword = New-Object System.Windows.Forms.Label
$lblDbPassword.Text = "数据库密码:"
$lblDbPassword.Location = New-Object System.Drawing.Point(12, 56)
$lblDbPassword.Size = New-Object System.Drawing.Size(72, 20)
$grpDb.Controls.Add($lblDbPassword)

$txtDbPassword = New-Object System.Windows.Forms.TextBox
$txtDbPassword.Location = New-Object System.Drawing.Point(84, 54)
$txtDbPassword.Size = New-Object System.Drawing.Size(120, 22)
$grpDb.Controls.Add($txtDbPassword)

$lblCondaEnv = New-Object System.Windows.Forms.Label
$lblCondaEnv.Text = "Anaconda 环境包名:"
$lblCondaEnv.Location = New-Object System.Drawing.Point(216, 56)
$lblCondaEnv.Size = New-Object System.Drawing.Size(112, 20)
$grpDb.Controls.Add($lblCondaEnv)

$txtCondaEnv = New-Object System.Windows.Forms.TextBox
$txtCondaEnv.Text = "webmedic"
$txtCondaEnv.Location = New-Object System.Drawing.Point(332, 54)
$txtCondaEnv.Size = New-Object System.Drawing.Size(172, 22)
$grpDb.Controls.Add($txtCondaEnv)

$btnCheckDb = New-Object System.Windows.Forms.Button
$btnCheckDb.Text = "检查数据库连接"
$btnCheckDb.Location = New-Object System.Drawing.Point(12, 84)
$btnCheckDb.Size = New-Object System.Drawing.Size(492, 26)
$grpDb.Controls.Add($btnCheckDb)

$grpDjango = New-Object System.Windows.Forms.GroupBox
$grpDjango.Text = "Django 配置"
$grpDjango.Location = New-Object System.Drawing.Point(12, 136)
$grpDjango.Size = New-Object System.Drawing.Size(516, 108)
$form.Controls.Add($grpDjango)

$lblDjangoPort = New-Object System.Windows.Forms.Label
$lblDjangoPort.Text = "Django 端口:"
$lblDjangoPort.Location = New-Object System.Drawing.Point(12, 26)
$lblDjangoPort.Size = New-Object System.Drawing.Size(72, 20)
$grpDjango.Controls.Add($lblDjangoPort)

$txtDjangoPort = New-Object System.Windows.Forms.TextBox
$txtDjangoPort.Text = "8000"
$txtDjangoPort.Location = New-Object System.Drawing.Point(84, 24)
$txtDjangoPort.Size = New-Object System.Drawing.Size(52, 22)
$grpDjango.Controls.Add($txtDjangoPort)

$btnStartDjango = New-Object System.Windows.Forms.Button
$btnStartDjango.Text = "启动网页服务"
$btnStartDjango.Location = New-Object System.Drawing.Point(148, 22)
$btnStartDjango.Size = New-Object System.Drawing.Size(108, 26)
$grpDjango.Controls.Add($btnStartDjango)

$btnReleaseDjangoPort = New-Object System.Windows.Forms.Button
$btnReleaseDjangoPort.Text = "解除端口占用"
$btnReleaseDjangoPort.Location = New-Object System.Drawing.Point(264, 22)
$btnReleaseDjangoPort.Size = New-Object System.Drawing.Size(108, 26)
$grpDjango.Controls.Add($btnReleaseDjangoPort)

$btnCheckDjangoPort = New-Object System.Windows.Forms.Button
$btnCheckDjangoPort.Text = "检查端口"
$btnCheckDjangoPort.Location = New-Object System.Drawing.Point(380, 22)
$btnCheckDjangoPort.Size = New-Object System.Drawing.Size(124, 26)
$grpDjango.Controls.Add($btnCheckDjangoPort)

$progressDjango = New-Object System.Windows.Forms.ProgressBar
$progressDjango.Location = New-Object System.Drawing.Point(12, 56)
$progressDjango.Size = New-Object System.Drawing.Size(492, 18)
$progressDjango.Minimum = 0
$progressDjango.Maximum = 100
$grpDjango.Controls.Add($progressDjango)

$lblDjangoStatus = New-Object System.Windows.Forms.Label
$lblDjangoStatus.Location = New-Object System.Drawing.Point(12, 80)
$lblDjangoStatus.Size = New-Object System.Drawing.Size(492, 20)
$lblDjangoStatus.Text = "网页服务未启动"
$grpDjango.Controls.Add($lblDjangoStatus)

$grpOpenCode = New-Object System.Windows.Forms.GroupBox
$grpOpenCode.Text = "智能体配置"
$grpOpenCode.Location = New-Object System.Drawing.Point(12, 250)
$grpOpenCode.Size = New-Object System.Drawing.Size(516, 108)
$form.Controls.Add($grpOpenCode)

$lblOpenCodePort = New-Object System.Windows.Forms.Label
$lblOpenCodePort.Text = "OpenCode 端口:"
$lblOpenCodePort.Location = New-Object System.Drawing.Point(12, 26)
$lblOpenCodePort.Size = New-Object System.Drawing.Size(88, 20)
$grpOpenCode.Controls.Add($lblOpenCodePort)

$txtOpenCodePort = New-Object System.Windows.Forms.TextBox
$txtOpenCodePort.Text = "4096"
$txtOpenCodePort.Location = New-Object System.Drawing.Point(100, 24)
$txtOpenCodePort.Size = New-Object System.Drawing.Size(52, 22)
$grpOpenCode.Controls.Add($txtOpenCodePort)

$btnStartOpenCode = New-Object System.Windows.Forms.Button
$btnStartOpenCode.Text = "启动智能体服务"
$btnStartOpenCode.Location = New-Object System.Drawing.Point(160, 22)
$btnStartOpenCode.Size = New-Object System.Drawing.Size(108, 26)
$grpOpenCode.Controls.Add($btnStartOpenCode)

$btnReleaseOpenCodePort = New-Object System.Windows.Forms.Button
$btnReleaseOpenCodePort.Text = "解除端口占用"
$btnReleaseOpenCodePort.Location = New-Object System.Drawing.Point(276, 22)
$btnReleaseOpenCodePort.Size = New-Object System.Drawing.Size(108, 26)
$grpOpenCode.Controls.Add($btnReleaseOpenCodePort)

$btnCheckOpenCodePort = New-Object System.Windows.Forms.Button
$btnCheckOpenCodePort.Text = "检查端口"
$btnCheckOpenCodePort.Location = New-Object System.Drawing.Point(392, 22)
$btnCheckOpenCodePort.Size = New-Object System.Drawing.Size(112, 26)
$grpOpenCode.Controls.Add($btnCheckOpenCodePort)

$progressOpenCode = New-Object System.Windows.Forms.ProgressBar
$progressOpenCode.Location = New-Object System.Drawing.Point(12, 56)
$progressOpenCode.Size = New-Object System.Drawing.Size(492, 18)
$progressOpenCode.Minimum = 0
$progressOpenCode.Maximum = 100
$grpOpenCode.Controls.Add($progressOpenCode)

$lblOpenCodeStatus = New-Object System.Windows.Forms.Label
$lblOpenCodeStatus.Location = New-Object System.Drawing.Point(12, 80)
$lblOpenCodeStatus.Size = New-Object System.Drawing.Size(492, 20)
$lblOpenCodeStatus.Text = "智能体服务未启动"
$grpOpenCode.Controls.Add($lblOpenCodeStatus)

$btnOpen = New-Object System.Windows.Forms.Button
$btnOpen.Text = "打开网页"
$btnOpen.Location = New-Object System.Drawing.Point(12, 372)
$btnOpen.Size = New-Object System.Drawing.Size(252, 32)
$btnOpen.Enabled = $false
$form.Controls.Add($btnOpen)

$btnStop = New-Object System.Windows.Forms.Button
$btnStop.Text = "关闭服务"
$btnStop.Location = New-Object System.Drawing.Point(276, 372)
$btnStop.Size = New-Object System.Drawing.Size(252, 32)
$btnStop.Enabled = $false
$form.Controls.Add($btnStop)

$saved = Load-LauncherConfig
if ($saved) {
    if ($saved.dbHost) { $txtDbHost.Text = ([string]$saved.dbHost).Trim() }
    if ($saved.dbPort) { $txtDbPort.Text = ([string]$saved.dbPort).Trim() }
    if ($saved.dbName) { $txtDbName.Text = ([string]$saved.dbName).Trim() }
    if ($null -ne $saved.dbPassword) { $txtDbPassword.Text = ([string]$saved.dbPassword).Trim() }
    if ($saved.condaEnv) { $txtCondaEnv.Text = ([string]$saved.condaEnv).Trim() }
    if ($saved.djangoPort) { $txtDjangoPort.Text = ([string]$saved.djangoPort).Trim() }
    if ($saved.openCodePort) { $txtOpenCodePort.Text = ([string]$saved.openCodePort).Trim() }
}

function Save-CurrentConfig {
    Save-LauncherConfig @{
        dbHost = $txtDbHost.Text.Trim()
        dbPort = $txtDbPort.Text.Trim()
        dbName = $txtDbName.Text.Trim()
        dbPassword = $txtDbPassword.Text
        condaEnv = $txtCondaEnv.Text.Trim()
        djangoPort = $txtDjangoPort.Text.Trim()
        openCodePort = $txtOpenCodePort.Text.Trim()
    }
}

$btnCheckDb.Add_Click({
    try {
        $db = Get-DatabaseSettings $txtDbHost $txtDbPort $txtDbName $txtDbPassword
        $condaEnv = Get-CondaEnvName $txtCondaEnv
        $btnCheckDb.Enabled = $false
        $btnCheckDb.Text = "正在检查..."
        [System.Windows.Forms.Application]::DoEvents()

        $prefix = Get-CondaCmdPrefix $condaEnv
        $py = @"
import pymysql
try:
    conn = pymysql.connect(host='$($db.Host)', port=$($db.Port), user='root', password='$($db.Password)', database='$($db.Name)', connect_timeout=5)
    conn.close()
    print('OK')
except Exception as e:
    print('ERR:' + str(e))
"@
        $pyFile = Join-Path $env:TEMP "webmedic_db_check.py"
        [System.IO.File]::WriteAllText($pyFile, $py, [System.Text.UTF8Encoding]::new($false))
        $cmd = "$prefix && python `"$pyFile`""
        $output = & cmd.exe /c $cmd 2>&1 | Out-String
        Remove-Item $pyFile -Force -ErrorAction SilentlyContinue

        if ($output -match "OK") {
            Save-CurrentConfig
            [System.Windows.Forms.MessageBox]::Show(
                "数据库连接成功。`r`n$($db.Host):$($db.Port) / $($db.Name)",
                "数据库检查",
                "OK",
                "Information"
            ) | Out-Null
        } else {
            $err = if ($output -match "ERR:(.+)") { $Matches[1].Trim() } else { $output.Trim() }
            throw "无法连接数据库：$err"
        }
    } catch {
        Show-ErrorBox $_.Exception.Message
    } finally {
        $btnCheckDb.Enabled = $true
        $btnCheckDb.Text = "检查数据库连接"
    }
})

$btnCheckDjangoPort.Add_Click({
    try {
        $port = Get-SinglePort $txtDjangoPort "Django"
        $line = Get-PortStatusText $port "Django"
        $inUse = Test-PortInUse $port
        $icon = if ($inUse) { "Warning" } else { "Information" }
        $title = if ($inUse) { "Django 端口：存在占用" } else { "Django 端口：可用" }
        [System.Windows.Forms.MessageBox]::Show($line, $title, "OK", $icon) | Out-Null
        $lblDjangoStatus.Text = if ($inUse) { "Django 端口 $port 已被占用" } else { "Django 端口 $port 可用" }
    } catch {
        Show-ErrorBox $_.Exception.Message
    }
})

$btnCheckOpenCodePort.Add_Click({
    try {
        $port = Get-SinglePort $txtOpenCodePort "OpenCode"
        $line = Get-PortStatusText $port "OpenCode"
        $inUse = Test-PortInUse $port
        $icon = if ($inUse) { "Warning" } else { "Information" }
        $title = if ($inUse) { "OpenCode 端口：存在占用" } else { "OpenCode 端口：可用" }
        [System.Windows.Forms.MessageBox]::Show($line, $title, "OK", $icon) | Out-Null
        $lblOpenCodeStatus.Text = if ($inUse) { "OpenCode 端口 $port 已被占用" } else { "OpenCode 端口 $port 可用" }
    } catch {
        Show-ErrorBox $_.Exception.Message
    }
})

$btnReleaseDjangoPort.Add_Click({
    try {
        $port = Get-SinglePort $txtDjangoPort "Django"
        if (-not $script:djangoRunning -and -not (Test-PortInUse $port)) {
            [System.Windows.Forms.MessageBox]::Show(
                "Django 端口 $port 未被占用。",
                "解除占用",
                "OK",
                "Information"
            ) | Out-Null
            return
        }
        $msg = if ($script:djangoRunning) {
            "将停止当前运行的 Django 网页服务（端口 $port），是否继续？"
        } else {
            "将强制结束占用 Django 端口 $port 的进程，是否继续？"
        }
        $confirm = [System.Windows.Forms.MessageBox]::Show(
            $msg,
            "解除占用",
            "YesNo",
            "Warning"
        )
        if ($confirm -ne "Yes") { return }

        $result = Invoke-ReleaseSinglePort $port "Django" $script:djangoRunning {
            Stop-DjangoService
            Reset-DjangoUi
            Update-BottomButtons
        }
        [System.Windows.Forms.MessageBox]::Show($result, "解除占用完成", "OK", "Information") | Out-Null
    } catch {
        Show-ErrorBox $_.Exception.Message
    }
})

$btnReleaseOpenCodePort.Add_Click({
    try {
        $port = Get-SinglePort $txtOpenCodePort "OpenCode"
        if (-not $script:opencodeRunning -and -not (Test-PortInUse $port)) {
            [System.Windows.Forms.MessageBox]::Show(
                "OpenCode 端口 $port 未被占用。",
                "解除占用",
                "OK",
                "Information"
            ) | Out-Null
            return
        }
        $msg = if ($script:opencodeRunning) {
            "将停止当前运行的 OpenCode 智能体服务（端口 $port），是否继续？"
        } else {
            "将强制结束占用 OpenCode 端口 $port 的进程，是否继续？"
        }
        $confirm = [System.Windows.Forms.MessageBox]::Show(
            $msg,
            "解除占用",
            "YesNo",
            "Warning"
        )
        if ($confirm -ne "Yes") { return }

        $result = Invoke-ReleaseSinglePort $port "OpenCode" $script:opencodeRunning {
            Stop-OpenCodeService
            Reset-OpenCodeUi
            Update-BottomButtons
        }
        [System.Windows.Forms.MessageBox]::Show($result, "解除占用完成", "OK", "Information") | Out-Null
    } catch {
        Show-ErrorBox $_.Exception.Message
    }
})

$btnStartDjango.Add_Click({
    if ($script:djangoRunning) { return }

    try {
        $djangoPort = Get-SinglePort $txtDjangoPort "Django"
        $openCodePort = Get-SinglePort $txtOpenCodePort "OpenCode"
        if ($djangoPort -eq $openCodePort) {
            throw "Django 与 OpenCode 端口不能相同。"
        }
        $db = Get-DatabaseSettings $txtDbHost $txtDbPort $txtDbName $txtDbPassword
        $condaEnv = Get-CondaEnvName $txtCondaEnv
    } catch {
        Show-ErrorBox $_.Exception.Message
        return
    }

    $btnStartDjango.Enabled = $false
    $btnCheckDjangoPort.Enabled = $false
    $btnReleaseDjangoPort.Enabled = $false
    $txtDjangoPort.Enabled = $false
    $progressDjango.Value = 0

    try {
        Ensure-LauncherMutex
        Update-Progress $progressDjango $lblDjangoStatus 10 "检查 Django 端口..."
        if (Test-PortInUse $djangoPort) { throw "Django 端口 $djangoPort 已被占用。" }

        if (-not (Test-Path (Join-Path $ProjectRoot "manage.py"))) { throw "未找到 manage.py。" }

        Save-CurrentConfig
        Write-DatabaseDotEnv $db $ProjectRoot
        $prefix = Get-CondaCmdPrefix $condaEnv
        $dbEnv = Get-DatabaseEnvPrefix $db

        Update-Progress $progressDjango $lblDjangoStatus 40 "正在启动 Django (HTTPS)..."
        $djangoCmd = "$prefix && $dbEnv && CD /D `"$ProjectRoot`" && python manage.py runsslserver 0.0.0.0:$djangoPort"
        $script:djangoPid = Start-ServiceCmd "WebMedic-Django" $djangoCmd
        $script:djangoPort = $djangoPort
        $script:djangoRunning = $true
        $script:homeUrl = "https://127.0.0.1:$djangoPort/"

        Update-Progress $progressDjango $lblDjangoStatus 100 "网页服务已启动"
        $lanLines = Get-LanAccessLines $djangoPort
        $lblDjangoStatus.Text = @(
            "网页服务已就绪（HTTPS）"
            "本机：$($script:homeUrl)"
        ) + $lanLines -join " | "
        $btnCheckDjangoPort.Enabled = $true
        $btnReleaseDjangoPort.Enabled = $true
        Update-BottomButtons
    } catch {
        Show-ErrorBox "Django 启动失败：`r`n$($_.Exception.Message)"
        Stop-DjangoService
        Reset-DjangoUi
        Update-BottomButtons
        if ($_.Exception.Message -like "*已在运行*") {
            $form.Close()
        }
    }
})

$btnStartOpenCode.Add_Click({
    if ($script:opencodeRunning) { return }

    try {
        $openCodePort = Get-SinglePort $txtOpenCodePort "OpenCode"
        $djangoPort = Get-SinglePort $txtDjangoPort "Django"
        if ($djangoPort -eq $openCodePort) {
            throw "Django 与 OpenCode 端口不能相同。"
        }
        $condaEnv = Get-CondaEnvName $txtCondaEnv
    } catch {
        Show-ErrorBox $_.Exception.Message
        return
    }

    $btnStartOpenCode.Enabled = $false
    $btnCheckOpenCodePort.Enabled = $false
    $btnReleaseOpenCodePort.Enabled = $false
    $txtOpenCodePort.Enabled = $false
    $progressOpenCode.Value = 0

    try {
        Ensure-LauncherMutex
        Update-Progress $progressOpenCode $lblOpenCodeStatus 10 "检查 OpenCode 端口..."
        if (Test-PortInUse $openCodePort) { throw "OpenCode 端口 $openCodePort 已被占用。" }
        if (-not (Test-Path $AgentDir)) { throw "未找到 main\Agent 目录。" }

        Save-CurrentConfig
        $prefix = Get-CondaCmdPrefix $condaEnv

        Update-Progress $progressOpenCode $lblOpenCodeStatus 50 "正在启动 OpenCode..."
        $openCmd = "$prefix && CD /D `"$AgentDir`" && opencode serve --port $openCodePort"
        $script:opencodePid = Start-ServiceCmd "WebMedic-OpenCode" $openCmd
        $script:openCodePort = $openCodePort
        $script:opencodeRunning = $true

        Update-Progress $progressOpenCode $lblOpenCodeStatus 100 "智能体服务已启动"
        $lblOpenCodeStatus.Text = "智能体服务已就绪：http://127.0.0.1:$openCodePort"
        $btnCheckOpenCodePort.Enabled = $true
        $btnReleaseOpenCodePort.Enabled = $true
        Update-BottomButtons
    } catch {
        Show-ErrorBox "OpenCode 启动失败：`r`n$($_.Exception.Message)"
        Stop-OpenCodeService
        Reset-OpenCodeUi
        Update-BottomButtons
        if ($_.Exception.Message -like "*已在运行*") {
            $form.Close()
        }
    }
})

$form.Add_FormClosing({
    param($sender, $e)
    if ($script:shuttingDown -or -not (Test-AnyServiceRunning)) { return }
    $ans = [System.Windows.Forms.MessageBox]::Show(
        "关闭窗口将停止所有已启动的服务，是否继续？",
        "关闭服务",
        "YesNo",
        "Question"
    )
    if ($ans -ne "Yes") { $e.Cancel = $true; return }
    & $stopAll
})

$btnOpen.Add_Click({
    if ($script:homeUrl) { Start-Process $script:homeUrl }
})

$btnStop.Add_Click({
    & $stopAll
    $form.Close()
})

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
