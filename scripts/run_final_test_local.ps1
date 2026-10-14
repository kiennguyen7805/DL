param(
    [switch]$SkipSlides,
    [string]$PythonPath = ''
)

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$venvPython = Join-Path $root '.venv\Scripts\python.exe'
if ($PythonPath) {
    $python = (Resolve-Path -LiteralPath $PythonPath).Path
} else {
    $python = $null
    if (Test-Path -LiteralPath $venvPython) {
        & $venvPython -c "import sys" 2>$null
        if ($LASTEXITCODE -eq 0) { $python = $venvPython }
    }
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $python -and $null -ne $pythonCommand) { $python = $pythonCommand.Source }
    if ($null -eq $python) {
        throw 'Khong tim thay Python. Truyen -PythonPath C:\duong-dan\python.exe'
    }
}

Set-Location -LiteralPath $root
$env:PYTHONUNBUFFERED = '1'
& $python -c "import sys; print('Python:', sys.executable)"
if ($LASTEXITCODE -ne 0) {
    throw "Python khong khoi dong duoc: $python. Neu .venv da bi di chuyen, truyen -PythonPath den Python dang dung duoc."
}

Write-Host '=== PREFLIGHT: evaluation lock va checkpoint ===' -ForegroundColor Cyan
& $python -c "from app.evaluate_models import verify_lock; verify_lock(); print('PASS: evaluation_lock.json')"
if ($LASTEXITCODE -ne 0) { throw 'Evaluation lock preflight FAIL' }

$folders = @{
    E4 = 'E4_YOLO11s'
    E6 = 'E6_Faster_RCNN'
}
foreach ($model in @('E4', 'E6')) {
    $report = Join-Path $root ("results/{0}/metrics/common_test.json" -f $folders[$model])
    if (Test-Path -LiteralPath $report) {
        Write-Host "=== ${model}: da co bao cao TEST, bo qua inference ===" -ForegroundColor Green
        continue
    }
    Write-Host "=== ${model}: danh gia TEST (tu dong tiep tuc cache neu bi gian doan) ===" -ForegroundColor Yellow
    & $python -m app.evaluate_test --model $model --confirm-locked-test
    if ($LASTEXITCODE -ne 0) { throw "$model TEST evaluation FAIL; cache da co se duoc dung o lan chay sau" }
}

Write-Host '=== VERIFY: E3/E4/E6 locked TEST ===' -ForegroundColor Cyan
& $python scripts/validate_final_test_results.py
if ($LASTEXITCODE -ne 0) { throw 'Final TEST validation FAIL; slide khong duoc cap nhat' }

if (-not $SkipSlides) {
    Write-Host '=== BUILD: cap nhat slide bang TEST ===' -ForegroundColor Cyan
    & (Join-Path $root 'docs/presentation/build_deck.ps1')
    if ($LASTEXITCODE -ne 0) { throw 'Base slide build FAIL' }
    & (Join-Path $root 'docs/presentation/augment_yolo11_architecture.ps1')
    if ($LASTEXITCODE -ne 0) { throw 'Detailed slide build FAIL' }
}

Write-Host '=== HOAN THANH ===' -ForegroundColor Green
Write-Host (Join-Path $root 'results/final_test_summary.json')
if (-not $SkipSlides) {
    Write-Host (Join-Path $root 'docs/presentation/BAO_VE_DO_AN_HOC_SAU_PHAT_HIEN_VAT_CAN_YOLO11_CHI_TIET.pptx')
    Write-Host (Join-Path $root 'BAO_CAO_PHAT_HIEN_VAT_CAN_DAY_DU (1)_SLIDES_YOLO11.pdf')
}
