param(
    [string]$PythonPath = '',
    [ValidateSet('E3', 'E4')][string[]]$Models = @('E3', 'E4'),
    [int]$BatchSize = 4,
    [string]$Device = 'auto'
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $root
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
if ($PythonPath) {
    $pythonExecutable = (Resolve-Path -LiteralPath $PythonPath).Path
} else {
    $pythonExecutable = $null
    $candidates = @((Join-Path $root '.venv/Scripts/python.exe'), (Join-Path $root 'tmp/python312_runtime/python.exe'))
    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($command) { $candidates += $command.Source }
    foreach ($candidate in $candidates) {
        if (-not (Test-Path -LiteralPath $candidate)) { continue }
        try {
            & $candidate -c 'import torch, ultralytics, torchvision, yaml, PIL' 2>$null
            if ($LASTEXITCODE -eq 0) { $pythonExecutable = $candidate; break }
        } catch { continue }
    }
    if (-not $pythonExecutable) { throw 'Python/dependencies unavailable. Pass -PythonPath to a working Python with torch, torchvision and ultralytics==8.4.165.' }
}
Write-Host "Python: $pythonExecutable" -ForegroundColor Cyan
& $pythonExecutable -u -m scripts.evaluate_30epoch_outputs --models @Models --batch $BatchSize --device $Device
if ($LASTEXITCODE -ne 0) { throw 'Evaluation stopped. Fix the error above then rerun this command to resume cached predictions.' }
Write-Host (Join-Path $root 'results/30epochs') -ForegroundColor Green
