#Requires -Version 7.0
$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
$logDirectory = Join-Path $repo 'logs'
$log = Join-Path $logDirectory 'publish_public_numbers.log'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null

$secrets = 'C:\Users\Kuruś\.dsh\secrets\secrets.ps1'
$env:CLOUDFLARE_API_KEY = & $secrets -Action Get -Name cloudflare_global_key -Reveal
$env:CLOUDFLARE_EMAIL = & $secrets -Action Get -Name cloudflare_email -Reveal
$env:CLOUDFLARE_ACCOUNT_ID = & $secrets -Action Get -Name cloudflare_account_id -Reveal

$stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
"[$stamp] publish start" | Add-Content -Path $log -Encoding utf8

$python = Join-Path $repo '.venv\Scripts\python.exe'
$script = Join-Path $repo 'scripts\publish_public_numbers.py'
& $python $script @args *>&1 | Tee-Object -FilePath $log -Append | Out-Null
$code = $LASTEXITCODE

"[$stamp] publish exit $code" | Add-Content -Path $log -Encoding utf8
exit $code
