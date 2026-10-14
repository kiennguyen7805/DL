param()

$ErrorActionPreference = 'Stop'
$reportRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$engine = Get-Command 'xelatex' -ErrorAction SilentlyContinue
if (-not $engine) {
    throw 'Không tìm thấy XeLaTeX trên PATH. Hãy dùng môi trường Prism đã cài TeX.'
}

Push-Location -LiteralPath $reportRoot
try {
    foreach ($pass in 1..2) {
        & $engine.Source '-interaction=nonstopmode' '-halt-on-error' '-file-line-error' 'main.tex'
        if ($LASTEXITCODE -ne 0) { throw "XeLaTeX thất bại ở lượt $pass. Xem main.log." }
    }
    if (-not (Test-Path -LiteralPath 'main.pdf')) { throw 'Không tìm thấy main.pdf sau khi biên dịch.' }
    $logText = Get-Content -LiteralPath 'main.log' -Raw -Encoding UTF8
    if ($logText -match '(?i)font.+not found|missing file|file .+ not found') {
        throw 'Log báo thiếu font hoặc file. Xem main.log.'
    }
    Write-Output 'Đã tạo main.pdf; không thấy mẫu lỗi thiếu font/file trong main.log.'
}
finally {
    Pop-Location
}
