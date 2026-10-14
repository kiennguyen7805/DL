$ErrorActionPreference = 'Stop'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$source = Join-Path $PSScriptRoot 'BAO_VE_DO_AN_HOC_SAU_PHAT_HIEN_VAT_CAN.pptx'
$outputPptx = Join-Path $PSScriptRoot 'BAO_VE_DO_AN_HOC_SAU_PHAT_HIEN_VAT_CAN_YOLO11_CHI_TIET.pptx'
$outputPdf = Join-Path $root 'BAO_CAO_PHAT_HIEN_VAT_CAN_DAY_DU (1)_SLIDES_YOLO11.pdf'
$assetDir = Join-Path $PSScriptRoot 'assets\yolo11'
$reviewDir = Join-Path $PSScriptRoot '_final_review'

$msoTextOrientationHorizontal = 1
$msoShapeRectangle = 1
$ppLayoutBlank = 12
$ppSaveAsOpenXMLPresentation = 24
$ppSaveAsPDF = 32
$navy = 0x4A2813
$blue = 0xC77816
$teal = 0xA68A0F
$cyan = 0xE8B43B
$ink = 0x292421
$muted = 0x756B63
$light = 0xF7F4EF
$white = 0xFFFFFF
$font = 'Aptos'

function Add-Text($slide, $text, $x, $y, $w, $h, $size = 20, $color = $ink, $bold = $false, $align = 1) {
    $shape = $slide.Shapes.AddTextbox($msoTextOrientationHorizontal, $x, $y, $w, $h)
    $shape.TextFrame2.TextRange.Text = $text
    $shape.TextFrame2.TextRange.Font.Name = $font
    $shape.TextFrame2.TextRange.Font.Size = $size
    $shape.TextFrame2.TextRange.Font.Fill.ForeColor.RGB = $color
    $shape.TextFrame2.TextRange.Font.Bold = [int]$bold
    $shape.TextFrame2.TextRange.ParagraphFormat.Alignment = $align
    $shape.TextFrame2.MarginLeft = 2
    $shape.TextFrame2.MarginRight = 2
    $shape.TextFrame2.MarginTop = 1
    $shape.TextFrame2.MarginBottom = 1
    return $shape
}

function Add-Title($slide, $number, $title, $subtitle = '') {
    Add-Text $slide ('{0:D2}' -f $number) 42 26 40 28 12 $blue $true 1 | Out-Null
    Add-Text $slide $title 82 19 815 45 28 $navy $true 1 | Out-Null
    $bar = $slide.Shapes.AddShape($msoShapeRectangle, 42, 66, 876, 3)
    $bar.Fill.ForeColor.RGB = $cyan
    $bar.Line.Visible = 0
    if ($subtitle) {
        Add-Text $slide $subtitle 44 73 870 24 12 $muted $false 1 | Out-Null
    }
    Add-Text $slide ('PHÁT HIỆN VẬT CẢN • {0:D2}' -f $number) 740 510 178 16 9 $muted $false 3 | Out-Null
}

function Add-FittedImage($slide, $path, $x, $y, $w, $h) {
    $picture = $slide.Shapes.AddPicture($path, $false, $true, 0, 0, -1, -1)
    $ratio = [Math]::Min($w / $picture.Width, $h / $picture.Height)
    $picture.Width = $picture.Width * $ratio
    $picture.Height = $picture.Height * $ratio
    $picture.Left = $x + (($w - $picture.Width) / 2)
    $picture.Top = $y + (($h - $picture.Height) / 2)
    return $picture
}

function Add-ThreePanelSlide($presentation, $index, $number, $title, $subtitle, $imageNames, $panelLabels, $caption) {
    $slide = $presentation.Slides.Add($index, $ppLayoutBlank)
    $slide.Background.Fill.ForeColor.RGB = $light
    Add-Title $slide $number $title $subtitle
    for ($panel = 0; $panel -lt 3; $panel++) {
        $x = 45 + ($panel * 300)
        Add-FittedImage $slide (Join-Path $assetDir $imageNames[$panel]) $x 105 270 315 | Out-Null
        Add-Text $slide $panelLabels[$panel] $x 424 270 22 13 @($blue, $teal, $cyan)[$panel] $true 2 | Out-Null
    }
    Add-Text $slide $caption 65 454 830 24 15 $navy $true 2 | Out-Null
    Add-Text $slide 'Nguồn: Hidayatullah et al., arXiv:2501.13400, Figure 4.' 65 484 830 18 10 $muted $false 2 | Out-Null
    return $slide
}

foreach ($required in @($source,
        (Join-Path $assetDir 'yolo11_overall.png'),
        (Join-Path $assetDir 'backbone_early.png'),
        (Join-Path $assetDir 'backbone_middle.png'),
        (Join-Path $assetDir 'backbone_deep.png'),
        (Join-Path $assetDir 'neck_context.png'),
        (Join-Path $assetDir 'neck_topdown.png'),
        (Join-Path $assetDir 'neck_bottomup.png'),
        (Join-Path $assetDir 'head_p3.png'),
        (Join-Path $assetDir 'head_p4.png'),
        (Join-Path $assetDir 'head_p5.png'))) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Thiếu file: $required" }
}

$powerPoint = New-Object -ComObject PowerPoint.Application
$powerPoint.Visible = -1
$presentation = $null
try {
    $presentation = $powerPoint.Presentations.Open($source, $false, $false, $false)

    # Replace the former schematic on slide 8 with the published overview.
    $overview = $presentation.Slides.Item(8)
    for ($i = $overview.Shapes.Count; $i -ge 1; $i--) { $overview.Shapes.Item($i).Delete() }
    $overview.Background.Fill.ForeColor.RGB = $light
    Add-Title $overview 8 'Kiến trúc YOLO11' 'Sơ đồ toàn mạng gồm Backbone, Neck và ba nhánh Detect đa tỉ lệ.'
    Add-FittedImage $overview (Join-Path $assetDir 'yolo11_overall.png') 55 103 850 365 | Out-Null
    Add-Text $overview 'Nguồn: Hidayatullah et al., arXiv:2501.13400, Figure 4.' 65 480 830 18 10 $muted $false 2 | Out-Null

    Add-ThreePanelSlide $presentation 9 9 'YOLO11 • Backbone' 'Ba đoạn liên tiếp của phần trích xuất đặc trưng.' @('backbone_early.png','backbone_middle.png','backbone_deep.png') @('P1–P2','P3–P4','P5') 'Conv giảm dần độ phân giải; C3k2 học đặc trưng từ nông đến sâu.' | Out-Null
    Add-ThreePanelSlide $presentation 10 10 'YOLO11 • Neck' 'Các nhánh hợp nhất đặc trưng đa tỉ lệ.' @('neck_context.png','neck_topdown.png','neck_bottomup.png') @('SPPF + C2PSA','Top-down path','Bottom-up path') 'Upsample và Concat truyền thông tin giữa các mức; C3k2 tinh lọc đặc trưng sau hợp nhất.' | Out-Null
    Add-ThreePanelSlide $presentation 11 11 'YOLO11 • Detection Head' 'Ba đầu ra tương ứng ba độ phân giải đặc trưng.' @('head_p3.png','head_p4.png','head_p5.png') @('P3 • vật thể nhỏ','P4 • vật thể trung bình','P5 • vật thể lớn') 'Mỗi nhánh Detect dự đoán lớp và bounding box ở một mức đặc trưng.' | Out-Null

    # Existing slides moved by three positions. Refresh visible slide numbers.
    for ($slideIndex = 1; $slideIndex -le $presentation.Slides.Count; $slideIndex++) {
        $slide = $presentation.Slides.Item($slideIndex)
        for ($shapeIndex = 1; $shapeIndex -le $slide.Shapes.Count; $shapeIndex++) {
            $shape = $slide.Shapes.Item($shapeIndex)
            if ($shape.HasTextFrame -ne -1 -or $shape.TextFrame.HasText -ne -1) { continue }
            $text = $shape.TextFrame.TextRange.Text.Trim()
            if ($text -match '^\d{2}$') {
                $shape.TextFrame.TextRange.Text = ('{0:D2}' -f $slideIndex)
            } elseif ($text -match '^PHÁT HIỆN VẬT CẢN • \d{2}$') {
                $shape.TextFrame.TextRange.Text = ('PHÁT HIỆN VẬT CẢN • {0:D2}' -f $slideIndex)
            }
        }
    }

    if (Test-Path -LiteralPath $outputPptx) { Remove-Item -LiteralPath $outputPptx -Force }
    if (Test-Path -LiteralPath $outputPdf) { Remove-Item -LiteralPath $outputPdf -Force }
    New-Item -ItemType Directory -Force -Path $reviewDir | Out-Null
    $presentation.SaveAs($outputPptx, $ppSaveAsOpenXMLPresentation)
    $presentation.SaveAs($outputPdf, $ppSaveAsPDF)
    for ($slideIndex = 1; $slideIndex -le $presentation.Slides.Count; $slideIndex++) {
        $png = Join-Path $reviewDir ('slide-{0:D2}.png' -f $slideIndex)
        $presentation.Slides.Item($slideIndex).Export($png, 'PNG', 1600, 900)
    }
}
finally {
    if ($null -ne $presentation) { $presentation.Close() }
    $powerPoint.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($presentation) | Out-Null
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($powerPoint) | Out-Null
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

Get-Item -LiteralPath $outputPptx, $outputPdf | Select-Object FullName, Length, LastWriteTime




