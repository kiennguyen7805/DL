$ErrorActionPreference = 'Stop'
$docx = (Resolve-Path 'docs\reports\BAO_CAO_PHAT_HIEN_VAT_CAN_DAY_DU\PAPER_PHAT_HIEN_VAT_CAN_NGUYEN_TRAN_KIEN.docx').Path
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
  $doc = $word.Documents.Open($docx, $false, $true)
  try {
    [pscustomobject]@{
      Pages = $doc.ComputeStatistics(2)
      Words = $doc.ComputeStatistics(0)
      Paragraphs = $doc.Paragraphs.Count
      Tables = $doc.Tables.Count
      InlineShapes = $doc.InlineShapes.Count
      Title = $doc.Paragraphs.Item(1).Range.Text.Trim()
      Author = $doc.Paragraphs.Item(2).Range.Text.Trim()
    }
  }
  finally { $doc.Close(0) }
}
finally { $word.Quit() }
