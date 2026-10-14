$ErrorActionPreference = 'Stop'
$docx = (Resolve-Path 'docs\reports\BAO_CAO_PHAT_HIEN_VAT_CAN_DAY_DU\PAPER_PHAT_HIEN_VAT_CAN_NGUYEN_TRAN_KIEN.docx').Path
$pdf = [System.IO.Path]::ChangeExtension($docx, '.pdf')
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
  $doc = $word.Documents.Open($docx, $false, $true)
  try {
    $doc.ExportAsFixedFormat($pdf, 17)
  }
  finally {
    $doc.Close(0)
  }
}
finally {
  $word.Quit()
}
Get-Item -LiteralPath $pdf | Select-Object FullName, Length, LastWriteTime
