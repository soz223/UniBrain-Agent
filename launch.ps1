param([int]$Port = 8773)
$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONPATH = $PSScriptRoot
Set-Location -LiteralPath $PSScriptRoot
$runtime = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (!(Test-Path -LiteralPath $runtime)) {
    $localSettings = Join-Path $PSScriptRoot 'local.json'
    if (Test-Path -LiteralPath $localSettings) {
        $runtime = (Get-Content -LiteralPath $localSettings -Raw | ConvertFrom-Json).python
    }
}
if (!$runtime -or !(Test-Path -LiteralPath $runtime)) {
    throw 'Create .venv and install requirements.txt. See README.md.'
}
try {
    $existing = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/status" -TimeoutSec 2
    if ($existing.app -eq 'unibrain_agent' -and $existing.version -eq '0.6') {
        Start-Process "http://127.0.0.1:$Port"
        exit 0
    }
} catch { }
& $runtime -m unibrain_agent serve --port $Port --open
