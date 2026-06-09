<#
start_website.ps1 (scripts/)
Purpose: Install dependencies (if needed) and start the Astro website in dev, build, or preview mode
Usage: pwsh.exe -File .\scripts\start_website.ps1 [mode]
  mode: dev     - Start Astro dev server with hot reload (default)
        build   - Build static site to dist/
        preview - Preview the built site locally
Examples:
  .\scripts\start_website.ps1          # Start dev server
  .\scripts\start_website.ps1 build    # Build static site
  .\scripts\start_website.ps1 preview  # Preview built site
#>
param(
    [string]$Mode = "dev"
)

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot
$WebsiteDir = Join-Path $RepoRoot "website"
$NodeModules = Join-Path $WebsiteDir "node_modules"

Write-Host "== scripts/start_website.ps1 =="
Write-Host "Website directory: $WebsiteDir"

if (-not (Test-Path $WebsiteDir)) {
    Write-Host "ERROR: Website directory not found at $WebsiteDir" -ForegroundColor Red
    exit 1
}

Set-Location $WebsiteDir

# Install dependencies if node_modules missing
if (-not (Test-Path $NodeModules)) {
    Write-Host "Installing npm dependencies..."
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "npm install failed with exit code: $LASTEXITCODE" -ForegroundColor Red
        exit 1
    }
}
else {
    Write-Host "node_modules exists; skipping npm install."
}

switch ($Mode.ToLower()) {
    "dev" {
        Write-Host "Starting Astro dev server (hot reload)..."
        npm run dev
    }
    "build" {
        Write-Host "Building static site..."
        npm run build
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Build complete. Output in: $WebsiteDir\dist\"
        }
    }
    "preview" {
        Write-Host "Starting Astro preview server..."
        npm run preview
    }
    default {
        Write-Host "Usage: .\scripts\start_website.ps1 [dev|build|preview]" -ForegroundColor Yellow
        Write-Host "  dev     - Start dev server (default)" -ForegroundColor Yellow
        Write-Host "  build   - Build static site to dist/" -ForegroundColor Yellow
        Write-Host "  preview - Preview the built site" -ForegroundColor Yellow
        exit 1
    }
}

# Keep window open if this is an interactive session
if ($Host.Name -eq "ConsoleHost") {
    Write-Host "`n=== Script completed. Press ENTER to close this window... ===" -ForegroundColor Cyan
    Read-Host
}
