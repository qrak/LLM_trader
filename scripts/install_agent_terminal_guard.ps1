# Installs a session-local safety net for accidental literal prompt-control text.
# This does not edit the user's PowerShell profile and only affects the current shell.

$ExecutionContext.InvokeCommand.CommandNotFoundAction = {
    param(
        [string] $CommandName,
        [System.Management.Automation.CommandLookupEventArgs] $CommandLookupEventArgs
    )

    if (-not $CommandName.StartsWith('^U')) {
        return
    }

    $cleanCommand = $CommandName.Substring(2)
    if ([string]::IsNullOrWhiteSpace($cleanCommand)) {
        return
    }

    $CommandLookupEventArgs.CommandScriptBlock = {
        & $cleanCommand @args
    }.GetNewClosure()
}

Write-Host "Installed session-local agent terminal guard: literal ^U prefixes will be stripped before command lookup."