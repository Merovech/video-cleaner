<#
.SYNOPSIS
    Set up video-cleaner on Windows: Python 3.12 venv, dependencies, ffmpeg, MKVToolNix,
    a private config file, then a self-check.

.DESCRIPTION
    Idempotent: every step checks first and is skipped when already done, so the script can be
    rerun safely. Nothing is downloaded or executed from an arbitrary URL: Python packages come
    from the locked uv.lock (uv sync --frozen) and external tools only through winget.

    Requires uv (https://docs.astral.sh/uv/) and winget (App Installer) on PATH.
    Run from the repository root:  powershell -ExecutionPolicy Bypass -File .\setup.ps1
#>
[CmdletBinding()]
param(
    # Skip the winget installs (for example when ffmpeg and MKVToolNix are managed elsewhere).
    [switch]$SkipTools
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = $PSScriptRoot
$PythonVersion = '3.12'

function Write-Step([string]$Message) {
    Write-Host "==> $Message"
}

function Assert-Command([string]$Name, [string]$Hint) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name was not found on PATH. $Hint"
    }
}

function Invoke-Native([string]$Exe, [string[]]$Arguments) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Exe $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
}

function Install-WingetPackage([string]$Id) {
    # winget list returns non-zero when the package is not installed. Windows PowerShell 5.1
    # turns redirected native stderr into errors, so relax 'Stop' for this probe only.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & winget list --id $Id -e --accept-source-agreements *> $null
        $installed = ($LASTEXITCODE -eq 0)
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($installed) {
        Write-Step "$Id already installed"
        return
    }
    Write-Step "Installing $Id with winget"
    Invoke-Native 'winget' @('install', '--id', $Id, '-e', '--accept-package-agreements', '--accept-source-agreements')
}

Push-Location $RepoRoot
try {
    Assert-Command 'uv' 'Install it with: winget install --id astral-sh.uv -e'

    Write-Step "Python $PythonVersion virtual environment"
    if (-not (Test-Path (Join-Path $RepoRoot '.venv\Scripts\python.exe'))) {
        Invoke-Native 'uv' @('venv', '--python', $PythonVersion)
    } else {
        Write-Step '.venv already exists'
    }

    Write-Step 'Installing locked dependencies (uv sync --frozen)'
    Invoke-Native 'uv' @('sync', '--frozen', '--python', $PythonVersion)

    if (-not $SkipTools) {
        Assert-Command 'winget' 'Install "App Installer" from the Microsoft Store.'
        Install-WingetPackage 'Gyan.FFmpeg'
        Install-WingetPackage 'MoritzBunkus.MKVToolNix'
    }

    if (-not $env:LOCALAPPDATA) {
        throw 'LOCALAPPDATA is not set; cannot place the config file.'
    }
    $ConfigDir = Join-Path $env:LOCALAPPDATA 'video-cleaner'
    $ConfigFile = Join-Path $ConfigDir 'config.toml'
    if (-not (Test-Path $ConfigDir)) {
        New-Item -ItemType Directory -Path $ConfigDir | Out-Null
    }
    if (Test-Path $ConfigFile) {
        Write-Step "Config already present: $ConfigFile (left unchanged)"
    } else {
        Write-Step "Creating $ConfigFile from config.example.toml; edit the paths before the first run"
        Copy-Item -Path (Join-Path $RepoRoot 'config.example.toml') -Destination $ConfigFile
    }

    # Restrict the config to the current user, SYSTEM and Administrators (also on reruns, in
    # case the ACL was loosened). /inheritance:r drops inherited entries such as BUILTIN\Users.
    Write-Step 'Restricting config file permissions'
    $User = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    Invoke-Native 'icacls' @(
        $ConfigFile, '/inheritance:r',
        '/grant:r', "${User}:(F)",
        '/grant:r', '*S-1-5-18:(F)',
        '/grant:r', '*S-1-5-32-544:(F)'
    )

    Write-Step 'Running video-cleaner doctor'
    # A fresh winget install updates PATH only for new shells; the doctor also looks in the
    # default winget and MKVToolNix install locations, so it finds them anyway.
    & uv run --frozen video-cleaner doctor --config $ConfigFile
    $DoctorExit = $LASTEXITCODE
    if ($DoctorExit -ne 0) {
        Write-Warning "doctor reported problems (exit $DoctorExit). Fix them and rerun: uv run video-cleaner doctor"
    }
    exit $DoctorExit
}
finally {
    Pop-Location
}
