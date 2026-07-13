$ErrorActionPreference = "Stop"

function Test-NineRouter {
    $listener = Get-NetTCPConnection -State Listen -LocalPort 20128 -ErrorAction SilentlyContinue
    return $null -ne $listener
}

if (Test-NineRouter) {
    Write-Host "9Router is already running at http://127.0.0.1:20128"
    while (Test-NineRouter) {
        Start-Sleep -Seconds 3
    }
    throw "9Router stopped unexpectedly."
}

$command = Get-Command 9router.cmd -ErrorAction SilentlyContinue
if (-not $command) {
    throw "9Router is not installed. Install it globally so 9router.cmd is available."
}

Write-Host "Starting 9Router at http://127.0.0.1:20128 ..."
& $command.Source --port 20128 --log --skip-update
exit $LASTEXITCODE
