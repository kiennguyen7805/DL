$ErrorActionPreference = 'Stop'
$root = (Resolve-Path '.').Path
$source = Join-Path $root 'docs\reports\BAO_CAO_PHAT_HIEN_VAT_CAN_DAY_DU\PAPER_PHAT_HIEN_VAT_CAN_NGUYEN_TRAN_KIEN.docx'
$output = Join-Path $root 'docs\reports\BAO_CAO_PHAT_HIEN_VAT_CAN_DAY_DU\PAPER_PHAT_HIEN_VAT_CAN_CO_HINH_VA_CONG_THUC.docx'
$pdf = [System.IO.Path]::ChangeExtension($output, '.pdf')

function Find-Range($doc, [string]$text) {
  $range = $doc.Content.Duplicate
  $find = $range.Find
  $find.ClearFormatting()
  $find.Text = $text
  if (-not $find.Execute()) { throw "Khong tim thay: $text" }
  return $range
}

function Add-Marker-After($doc, [string]$needle, [string]$marker) {
  $range = Find-Range $doc $needle
  $range.Collapse(0)
  $range.InsertAfter("`r$marker`r")
}

function Replace-Marker-With-Picture($doc, [string]$marker, [string]$path, [string]$caption, [double]$width) {
  $range = Find-Range $doc $marker
  $range.Text = ''
  $shape = $doc.InlineShapes.AddPicture($path, $false, $true, $range)
  $shape.LockAspectRatio = -1
  $shape.Width = $width
  $shape.Range.ParagraphFormat.Alignment = 1
  $after = $shape.Range.Duplicate
  $after.Collapse(0)
  $after.InsertAfter("`r$caption`r")
  $cap = Find-Range $doc $caption
  $cap.ParagraphFormat.Alignment = 1
  $cap.Font.Name = 'Times New Roman'
  $cap.Font.Size = 10
  $cap.Font.Italic = $true
}

function Replace-Marker-With-Equations($doc, [string]$marker) {
  $range = Find-Range $doc $marker
  $range.Text = ''
  $equations = @(
    'IoU(Bp, Bgt) = |Bp ∩ Bgt| / |Bp ∪ Bgt|',
    'Precision = TP / (TP + FP)',
    'Recall = TP / (TP + FN)',
    'Fβ = (1 + β²)PR / (β²P + R);   F2 = 5PR / (4P + R)',
    'AP = ∫₀¹ Pinterp(R)dR;   mAP50:95 = (1/10) Σ APt,  t ∈ {0,50; 0,55; …; 0,95}'
  )
  foreach ($eq in $equations) {
    $range.InsertAfter("$eq`r")
    $eqRange = Find-Range $doc $eq
    $eqRange.ParagraphFormat.Alignment = 1
    $eqRange.Font.Name = 'Cambria Math'
    $eqRange.Font.Size = 11
    $eqRange.Font.Italic = $true
    $range = $eqRange.Duplicate
    $range.Collapse(0)
  }
}

Copy-Item -LiteralPath $source -Destination $output -Force
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
  $doc = $word.Documents.Open($output, $false, $false)
  try {
    Add-Marker-After $doc 'Mô hình' '[[FIG_YOLO]]'
    Add-Marker-After $doc '[[FIG_YOLO]]' '[[FIG_FRCNN]]'
    Add-Marker-After $doc 'Chỉ số và evaluator' '[[EQUATIONS]]'
    Add-Marker-After $doc 'Ứng dụng ảnh tĩnh' '[[FIG_RESULT_1]]'
    Add-Marker-After $doc '[[FIG_RESULT_1]]' '[[FIG_RESULT_2]]'

    Replace-Marker-With-Picture $doc '[[FIG_YOLO]]' (Join-Path $root 'YOLO11_KIEN_TRUC_CHI_TIET.png') 'Hình 1  Kiến trúc YOLO11 gồm backbone, neck FPN PAN và ba đầu phát hiện đa tỷ lệ.' 455
    Replace-Marker-With-Picture $doc '[[FIG_FRCNN]]' (Join-Path $root 'FASTER_RCNN_RESNET50_FPN_V2_KIEN_TRUC.png') 'Hình 2  Kiến trúc Faster R-CNN ResNet-50 FPN V2 với RPN và RoI Head.' 455
    Replace-Marker-With-Equations $doc '[[EQUATIONS]]'
    Replace-Marker-With-Picture $doc '[[FIG_RESULT_1]]' (Join-Path $root 'app\outputs\rod_55a86e798f7c4b9eb4fcfb3e940cbff9.png') 'Hình 3  Ví dụ phát hiện nắp cống trong cảnh vỉa hè.' 285
    Replace-Marker-With-Picture $doc '[[FIG_RESULT_2]]' (Join-Path $root 'app\outputs\rod_032ec7a6c47a4d28b22973b7c2784583.png') 'Hình 4  Ví dụ phát hiện nhiều vật cản và nhầm lẫn giữa Dustbin với Plant Pot.' 285

    $doc.Save()
    $doc.ExportAsFixedFormat($pdf, 17)
    [pscustomobject]@{
      Pages = $doc.ComputeStatistics(2)
      Words = $doc.ComputeStatistics(0)
      Tables = $doc.Tables.Count
      Images = $doc.InlineShapes.Count
      Output = $output
      PDF = $pdf
    }
  }
  finally { $doc.Close(0) }
}
finally { $word.Quit() }
