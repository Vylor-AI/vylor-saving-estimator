# install.ps1 - vylor-estimate installer and runner for Windows
# Usage: irm https://raw.githubusercontent.com/Vylor-AI/vylor-saving-estimator/main/install.ps1 | iex

$ErrorActionPreference = "Stop"

$Repo = "Vylor-AI/vylor-saving-estimator"
$BinaryName = "vylor-estimate.exe"
$InstallDir = Join-Path $env:LOCALAPPDATA "vylor-estimate"
$Dest = Join-Path $InstallDir $BinaryName

Write-Host ""
Write-Host "vylor-estimate" -ForegroundColor Cyan
Write-Host "==============" -ForegroundColor Cyan

# Resolve latest release tag
try {
    $Release = Invoke-RestMethod "https://api.github.com/repos/$Repo/releases/latest"
    $Tag = $Release.tag_name
} catch {
    Write-Host "ERROR: Could not determine latest release. Check your internet connection." -ForegroundColor Red
    exit 1
}

$Url = "https://github.com/$Repo/releases/download/$Tag/$BinaryName"

# Create install directory
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

Write-Host "Downloading vylor-estimate $Tag..." -ForegroundColor Gray

# Fast download without UI bottleneck
$prevProgress = $ProgressPreference
$ProgressPreference = 'SilentlyContinue'
try {
    Invoke-WebRequest -Uri $Url -OutFile $Dest -UseBasicParsing
} catch {
    Write-Host "ERROR: Download failed from $Url" -ForegroundColor Red
    exit 1
} finally {
    $ProgressPreference = $prevProgress
}

# 1. Add to user PATH persistently for all future terminals
$CurrentPath = [Environment]::GetEnvironmentVariable("Path", [EnvironmentVariableTarget]::User)
if ($CurrentPath -notlike "*$InstallDir*") {
    [Environment]::SetEnvironmentVariable(
        "Path",
        "$CurrentPath;$InstallDir",
        [EnvironmentVariableTarget]::User
    )
}

# 2. Add to current session PATH immediately (no restart needed!)
if ($env:Path -notlike "*$InstallDir*") {
    $env:Path = "$InstallDir;$env:Path"
}

Write-Host "Installed to $Dest" -ForegroundColor Green
Write-Host ""

# 3. Automatically run the tool right now!
& $Dest $args
