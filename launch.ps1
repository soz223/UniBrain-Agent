param([int]$Port = 8776)
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
    $headers = @{}
    if ($env:UNIBRAIN_ACCESS_TOKEN) { $headers.Authorization = 'Bearer ' + $env:UNIBRAIN_ACCESS_TOKEN }
    $existing = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/status" -Headers $headers -TimeoutSec 2
    if ($existing.app -eq 'unibrain_agent' -and $existing.version -eq '0.8') {
        Start-Process "http://127.0.0.1:$Port"
        exit 0
    }
} catch {
    if ($_.Exception.Response.StatusCode -eq 401) {
        Write-Output 'An authenticated backend is already running. Connect with its token in AI settings.'
        Start-Process "http://127.0.0.1:$Port"
        exit 0
    }
}
& $runtime -m unibrain_agent serve --port $Port --open
