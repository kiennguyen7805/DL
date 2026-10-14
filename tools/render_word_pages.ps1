$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$docx = (Resolve-Path 'docs\reports\BAO_CAO_PHAT_HIEN_VAT_CAN_DAY_DU\PAPER_PHAT_HIEN_VAT_CAN_CO_HINH_VA_CONG_THUC.docx').Path
$out = Join-Path (Resolve-Path '.').Path 'tmp\paper_visual_qa'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
  $doc = $word.Documents.Open($docx, $false, $true)
  try {
    $pages = $doc.ComputeStatistics(2)
    for ($i = 1; $i -le $pages; $i++) {
      $sel = $word.Selection
      $null = $sel.GoTo(1, 1, $i)
      $pageRange = $sel.Bookmarks.Item('\Page').Range
      [System.Windows.Forms.Clipboard]::Clear()
      $pageRange.CopyAsPicture()
      Start-Sleep -Milliseconds 300
      $img = [System.Windows.Forms.Clipboard]::GetImage()
      if ($null -eq $img) { throw "Clipboard image unavailable for page $i" }
      $path = Join-Path $out ("page-{0}.png" -f $i)
      $img.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
      $img.Dispose()
    }
  }
  finally { $doc.Close(0) }
}
finally { $word.Quit() }
Get-ChildItem -LiteralPath $out -Filter '*.png' | Select-Object Name, Length
