<#
start_script_release_candidate_rc2.ps1
Purpose: Prepare .venv, ensure on 'release/v1.0-rc2' git branch when available, install requirements, and run start.py.
Usage: pwsh.exe -File .\start_script_release_candidate_rc2.ps1 [symbol] [-Timeframe <tf>]
Examples:
    .\start_script_release_candidate_rc2.ps1
    .\start_script_release_candidate_rc2.ps1 BTC/USDC -Timeframe 4h
Parameters:
  -Symbol <pair>       Optional trading pair override.
  -Timeframe <tf>      Optional timeframe override.
  -Venv <path>         Path to virtual environment (default: .venv).
  -SkipInstall         Skip pip install step.
#>
param(
    [string]$Symbol = "",
    [string]$Timeframe = "",
    [string]$Venv = ".venv",
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'
$ReleaseBranch = 'release/v1.0-rc2'

Write-Host "== start_script_release_candidate_rc2.ps1 ($ReleaseBranch) =="
Write-Host "Graceful stop: use Ctrl+C (app shows confirmation popup)." -ForegroundColor Yellow
Write-Host "Closing the terminal window/tab with X terminates host process immediately." -ForegroundColor Yellow

if (Test-Path .git) {
    Write-Host "Switching to '$ReleaseBranch' branch (Release Candidate 2)..."
    try {
        git fetch --all --prune
        git checkout $ReleaseBranch
    }
    catch {
        Write-Host "Git branch checkout failed; continuing with the current working tree." -ForegroundColor Yellow
        Write-Host $_.Exception.Message -ForegroundColor Yellow
    }
}
else {
    Write-Host "No .git folder found; skipping branch checkout."
}

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
        foreach ($requirement in $reqs) {
            $name = ($requirement -split '[=<>!~]')[0].Trim()
            if ($requirement -match '==') {
                $pattern = '^' + [regex]::Escape($requirement) + '$'
                if (-not ($installed -match $pattern)) { $missing += $requirement }
            }
            else {
                $pattern = '^' + [regex]::Escape($name) + '=='
                if (-not ($installed -match $pattern)) { $missing += $requirement }
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
    Write-Host "Skipping dependency installation (-SkipInstall provided)."
}

if (Test-Path 'start.py') {
    $startArgs = @()
    if ($Symbol) { $startArgs += $Symbol }
    if ($Timeframe) { $startArgs += '-t', $Timeframe }

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

Write-Host "`n=== Script completed. Press ENTER to close this window... ===" -ForegroundColor Cyan
Read-Host
