param([string]$Python = '')
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
if (-not $Python) { $Python = Join-Path $root '.venv\Scripts\python.exe' }
& $Python -m src.build_30epoch_notebooks --prepare-bundles
if ($LASTEXITCODE -ne 0) { throw 'Notebook/bundle preparation failed; do not upload partial bundles.' }
