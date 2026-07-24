<#
start_script_main.ps1 (scripts/)
Purpose: Prepare .venv, install requirements, and run start.py (main branch)
Usage: pwsh.exe -File .\scripts\start_script_main.ps1 [symbol] [-Timeframe <tf>]
Examples:
    .\scripts\start_script_main.ps1
    .\scripts\start_script_main.ps1 BTC/USDC -Timeframe 4h
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

Write-Output "== scripts/start_script_main.ps1 (main) =="
Write-Output "Graceful stop: use Ctrl+C (app shows confirmation popup)."
Write-Output "Closing the terminal window/tab with X terminates host process immediately."
Write-Output "Repository root: $RepoRoot"

if (-not (Test-Path $VenvPath)) {
    Write-Output "Creating virtual environment at '$VenvPath'..."
    python -m venv $VenvPath
}
else {
    Write-Output "Virtual environment '$VenvPath' already exists."
}

if (-not (Test-Path $ActivatePath)) {
    Write-Output "Activation script not found at $ActivatePath"
    Write-Output "If the venv was just created, ensure Python created the Scripts/Activate.ps1 file."
    exit 1
}

Write-Output "Activating virtual environment..."
. $ActivatePath

if (-not $SkipInstall) {
    if (Test-Path $RequirementsPath) {
        Write-Output "Checking installed packages against requirements.txt..."
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
            Write-Output "All requirements satisfied; skipping pip install."
        }
        else {
            Write-Output "Missing or mismatched requirements detected:`n$missing"
            Write-Output "Installing/updating dependencies from requirements.txt..."
            pip install --upgrade pip
            pip install -r $RequirementsPath
        }
    }
    else {
        Write-Output "No requirements.txt found; skipping pip install."
    }
}
else {
    Write-Output "Skipping dependency installation (-SkipInstall provided)."
}

Set-Location $RepoRoot

if (Test-Path $StartPath) {
    $startArgs = @()
    if ($Symbol) { $startArgs += $Symbol }
    if ($Timeframe) { $startArgs += '-t', $Timeframe }

    if ($startArgs.Count -gt 0) {
        Write-Output "Running start.py with arguments: $($startArgs -join ' ')..."
    }
    else {
        Write-Output "Running start.py with default settings..."
    }

    & python $StartPath @startArgs

    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        Write-Output "`n=== Process exited with error code: $exitCode ===`n"
    }
}
else {
    Write-Output "No start.py found in repository root."
}

Write-Output "`n=== Script completed. Press ENTER to close this window... ===`n"
Read-Host
