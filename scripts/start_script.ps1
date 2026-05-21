<#
start_script.ps1
Purpose: Prepare .venv, ensure on 'main' git branch (if repo), install requirements, and run start.py
Usage: pwsh.exe -File .\scripts\start_script.ps1 [symbol] [-Timeframe <tf>]
Examples:
  .\scripts\start_script.ps1                           # Run with default from config.ini
  .\scripts\start_script.ps1 ETH/USDT                  # Trade ETH/USDT
  .\scripts\start_script.ps1 BTC/USDT -Timeframe 4h    # Trade BTC/USDT on 4h timeframe
  .\scripts\start_script.ps1 -SkipInstall              # Skip pip install step
Parameters:
  -Symbol <string>     Trading symbol (e.g., BTC/USDT, ETH/USDT)
  -Timeframe <string>  Timeframe for trading (e.g., 1h, 4h, 1d)
  -Venv <path>         Path to virtual environment relative to repository root (default: .venv)
  -SkipInstall         Skip pip install step
#>
param(
    [string]$Symbol = "",
    [string]$Timeframe = "",
    [string]$Venv = ".venv",
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $RepoRoot $Venv
$ActivatePath = Join-Path $VenvPath 'Scripts\Activate.ps1'
$RequirementsPath = Join-Path $RepoRoot 'requirements.txt'
$StartPath = Join-Path $RepoRoot 'start.py'
$GitPath = Join-Path $RepoRoot '.git'

Write-Host "== scripts/start_script.ps1 (main) =="
Write-Host "Repository root: $RepoRoot"

# Checkout branch if this is a git repo
if (Test-Path $GitPath) {
    if (Get-Command git -ErrorAction SilentlyContinue) {
        Write-Host "Switching to 'main' branch..."
        git -C $RepoRoot fetch --all --prune
        git -C $RepoRoot checkout main
    }
    else {
        Write-Host "Git command not found; skipping branch checkout."
    }
}
else {
    Write-Host "No .git folder found; skipping branch checkout."
}

# Create virtual environment if missing
if (-not (Test-Path $VenvPath)) {
    Write-Host "Creating virtual environment at '$VenvPath'..."
    python -m venv $VenvPath
}
else {
    Write-Host "Virtual environment '$VenvPath' already exists."
}

if (-not (Test-Path $ActivatePath)) {
    Write-Host "Activation script not found at $ActivatePath"
    Write-Host "If the venv was just created, ensure Python created Scripts/Activate.ps1."
    exit 1
}

Write-Host "Activating virtual environment..."
. $ActivatePath

if (-not $SkipInstall) {
    if (Test-Path $RequirementsPath) {
        Write-Host "Checking installed packages against requirements.txt..."
        $reqs = Get-Content $RequirementsPath | ForEach-Object { $_.Trim() } | Where-Object { $_ -and -not ($_ -match '^(\s*#)') }
        $installed = & pip freeze
        $missing = @()
        foreach ($r in $reqs) {
            $name = ($r -split '[=<>!~]')[0].Trim()
            if ($r -match '==') {
                $pattern = '^' + [regex]::Escape($r) + '$'
                if (-not ($installed -match $pattern)) { $missing += $r }
            }
            else {
                $pattern = '^' + [regex]::Escape($name) + '=='
                if (-not ($installed -match $pattern)) { $missing += $r }
            }
        }

        if ($missing.Count -eq 0) {
            Write-Host "All requirements satisfied; skipping pip install."
        }
        else {
            Write-Host "Missing or mismatched requirements detected:`n$missing"
            Write-Host "Installing/updating dependencies from requirements.txt..."
            pip install --upgrade pip
            pip install -r $RequirementsPath
        }
    }
    else {
        Write-Host "No requirements.txt found; skipping pip install."
    }
}
else {
    Write-Host "Skipping dependency installation (--SkipInstall provided)."
}

if (Test-Path $StartPath) {
    $startArgs = @()
    if ($Symbol) { $startArgs += $Symbol }
    if ($Timeframe) { $startArgs += "-t", $Timeframe }

    if ($startArgs.Count -gt 0) {
        Write-Host "Running start.py with arguments: $($startArgs -join ' ')..."
    }
    else {
        Write-Host "Running start.py with default settings..."
    }

    & python $StartPath @startArgs

    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        Write-Host "`n=== Process exited with error code: $exitCode ===" -ForegroundColor Red
    }
}
else {
    Write-Host "No start.py found in repository root."
}

Write-Host "`n=== Script completed. Press ENTER to close this window... ===" -ForegroundColor Cyan
Read-Host