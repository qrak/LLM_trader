<#
start_script_test.ps1
Purpose: Prepare .venv, ensure on 'development' git branch (if repo), install requirements, and run start.py
Usage: pwsh.exe -File .\start_script_test.ps1 [symbol] [-Timeframe <tf>]
Examples:
  .\start_script_test.ps1                           # Run with default from config.ini
  .\start_script_test.ps1 ETH/USDT                  # Trade ETH/USDT
  .\start_script_test.ps1 BTC/USDT -Timeframe 4h    # Trade BTC/USDT on 4h timeframe
  .\start_script_test.ps1 -SkipInstall              # Skip pip install step
Parameters:
  -Symbol <string>     Trading symbol (e.g., BTC/USDT, ETH/USDT)
  -Timeframe <string>  Timeframe for trading (e.g., 1h, 4h, 1d)
  -Venv <path>         Path to virtual environment (default: .venv)
  -SkipInstall         Skip pip install step
#>
param(
    [string]$Symbol = "",
    [string]$Timeframe = "",
    [string]$Venv = ".venv",
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'

Write-Host "== start_script_test.ps1 (test) =="

# Checkout branch if this is a git repo
if (Test-Path .git) {
    Write-Host "Switching to 'development' branch..."
    git fetch --all --prune
    git checkout development
}
else {
    Write-Host "No .git folder found; skipping branch checkout."
}

# Create virtual environment if missing
if (-not (Test-Path $Venv)) {
    Write-Host "Creating virtual environment at '$Venv'..."
    python -m venv $Venv
}
else {
    Write-Host "Virtual environment '$Venv' already exists."
}

$activate = Join-Path $Venv 'Scripts\Activate.ps1'
if (-not (Test-Path $activate)) {
    Write-Host "Activation script not found at $activate"
    Write-Host "If the venv was just created, ensure Python created the Scripts/Activate.ps1 file."
    exit 1
}

Write-Host "Activating virtual environment..."
. $activate

if (-not $SkipInstall) {
    if (Test-Path 'requirements.txt') {
        Write-Host "Checking installed packages against requirements.txt..."
        $reqs = Get-Content 'requirements.txt' | ForEach-Object { $_.Trim() } | Where-Object { $_ -and -not ($_ -match '^(\s*#)') }
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
            Write-Host "Installing/Updating dependencies from requirements.txt..."
            pip install --upgrade pip
            pip install -r requirements.txt
        }
    }
    else {
        Write-Host "No requirements.txt found; skipping pip install."
    }
}
else {
    Write-Host "Skipping dependency installation (--SkipInstall provided)."
}

# Run the project's start entrypoint
if (Test-Path 'start.py') {
    # Build command with optional arguments
    $startArgs = @()
    if ($Symbol) { $startArgs += $Symbol }
    if ($Timeframe) { $startArgs += "-t", $Timeframe }
    
    if ($startArgs.Count -gt 0) {
        Write-Host "Running start.py with arguments: $($startArgs -join ' ')..."
        python start.py @startArgs
    }
    else {
        Write-Host "Running start.py with default settings..."
        python start.py
    }
    
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        Write-Host "`n=== Process exited with error code: $exitCode ===" -ForegroundColor Red
    }
}
else {
    Write-Host "No start.py found in repository root."
}

# Keep window open to review output
Write-Host "`n=== Script completed. Press ENTER to close this window... ===" -ForegroundColor Cyan
Read-Host
