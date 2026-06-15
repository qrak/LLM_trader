<#
start_script_develop.ps1 (scripts/)
Purpose: Prepare .venv, switch to develop branch (best-effort), install requirements, and run start.py
Usage: pwsh.exe -File .\scripts\start_script_develop.ps1 [symbol] [-Timeframe <tf>]
Examples:
    .\scripts\start_script_develop.ps1
    .\scripts\start_script_develop.ps1 BTC/USDC -Timeframe 4h
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

$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $RepoRoot $Venv
$ActivatePath = Join-Path $VenvPath 'Scripts\Activate.ps1'
$RequirementsPath = Join-Path $RepoRoot 'requirements.txt'
$StartPath = Join-Path $RepoRoot 'start.py'
$GitPath = Join-Path $RepoRoot '.git'

Write-Host "== scripts/start_script_develop.ps1 (develop) =="
Write-Host "Graceful stop: use Ctrl+C (app shows confirmation popup)." -ForegroundColor Yellow
Write-Host "Closing the terminal window/tab with X terminates host process immediately." -ForegroundColor Yellow
Write-Host "Repository root: $RepoRoot"

if (Test-Path $GitPath) {
    if (Get-Command git -ErrorAction SilentlyContinue) {
        Write-Host "Switching to 'develop' branch (best-effort)..."
        git -C $RepoRoot fetch --all --prune
        try {
            git -C $RepoRoot checkout develop
        }
        catch {
            Write-Host "Could not checkout 'develop'; continuing on current branch." -ForegroundColor Yellow
        }
    }
    else {
        Write-Host "Git command not found; skipping branch checkout."
    }
}
else {
    Write-Host "No .git folder found; skipping branch checkout."
}

if (-not (Test-Path $VenvPath)) {
    Write-Host "Creating virtual environment at '$VenvPath'..."
    python -m venv $VenvPath
}
else {
    Write-Host "Virtual environment '$VenvPath' already exists."
}

if (-not (Test-Path $ActivatePath)) {
    Write-Host "Activation script not found at $ActivatePath"
    Write-Host "If the venv was just created, ensure Python created the Scripts/Activate.ps1 file."
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
        foreach ($req in $reqs) {
            $name = ($req -split '[=<>!~]')[0].Trim()
            if ($req -match '==') {
                $pattern = '^' + [regex]::Escape($req) + '$'
                if (-not ($installed -match $pattern)) { $missing += $req }
            }
            else {
                $pattern = '^' + [regex]::Escape($name) + '=='
                if (-not ($installed -match $pattern)) { $missing += $req }
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
    Write-Host "Skipping dependency installation (-SkipInstall provided)."
}

Set-Location $RepoRoot

if (Test-Path $StartPath) {
    $startArgs = @()
    if ($Symbol) { $startArgs += $Symbol }
    if ($Timeframe) { $startArgs += '-t', $Timeframe }

    if ($startArgs.Count -gt 0) {
        Write-Host "Running start.py with arguments: $($startArgs -join ' ')..."
    }
    else {
        Write-Host "Running start.py with default settings..."
    }

    & python $StartPath @startArgs

    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        Write-Host "`n=== Process exited with error code: $exitCode ===`n" -ForegroundColor Red
    }
}
else {
    Write-Host "No start.py found in repository root."
}

Write-Host "`n=== Script completed. Press ENTER to close this window... ===`n" -ForegroundColor Cyan
Read-Host
