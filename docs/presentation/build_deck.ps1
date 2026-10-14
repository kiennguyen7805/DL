$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$outDir = Join-Path $root 'docs\presentation'
$renderDir = Join-Path $outDir '_rendered'
New-Item -ItemType Directory -Force -Path $outDir,$renderDir | Out-Null
$pptxPath = Join-Path $outDir 'BAO_VE_DO_AN_HOC_SAU_PHAT_HIEN_VAT_CAN.pptx'
$pdfPath = Join-Path $outDir 'BAO_VE_DO_AN_HOC_SAU_PHAT_HIEN_VAT_CAN.pdf'
$testSummaryPath = Join-Path $root 'results\final_test_summary.json'
$testComplete = Test-Path -LiteralPath $testSummaryPath
$testSummary = if($testComplete){Get-Content -LiteralPath $testSummaryPath -Raw -Encoding UTF8 | ConvertFrom-Json}else{$null}
if($testComplete -and ($testSummary.status -ne 'FINAL_LOCKED' -or $testSummary.split -ne 'test' -or $testSummary.images -ne 1611)){throw "Invalid final TEST summary: $testSummaryPath"}
$validP=@(0.6455358544,0.6502194642,0.7482529118);$validR=@(0.7733082707,0.8075187970,0.8453007519)
$validM50=@(0.7957589188,0.8048659705,0.8701219838);$validM5095=@(0.5951288215,0.6145455259,0.6604099804)
$classes=@('Bike','Building','Car','Person','Stairs','Traffic sign','Electrical Pole','Road','Motorcycle','Dustbin','Dog','Manhole','Tree','Guard rail','Pedestrian crosswalk','Truck','Bus','Bench','Traffic Cone','Fire hydrant','Traffic Barrel','Plant Pot','Electrical Box','Chair','Bicycle Rack')
if($testComplete){
  $testModels=@($testSummary.models.E3,$testSummary.models.E4,$testSummary.models.E6)
  $resultP=@($testModels|ForEach-Object{[double]$_.precision});$resultR=@($testModels|ForEach-Object{[double]$_.recall})
  $resultM50=@($testModels|ForEach-Object{[double]$_.map50});$resultM5095=@($testModels|ForEach-Object{[double]$_.map50_95})
  $resultSplit='TEST'
}else{
  $resultP=$validP;$resultR=$validR;$resultM50=$validM50;$resultM5095=$validM5095;$resultSplit='VALID'
}

$ppLayoutBlank=12; $msoTextOrientationHorizontal=1; $msoShapeRectangle=1; $msoShapeRoundedRectangle=5; $msoShapeOval=9; $msoConnectorStraight=1; $msoArrowheadTriangle=3
$W=960; $H=540
$navy=0x4A2813; $blue=0xC77816; $teal=0xA68A0F; $cyan=0xE8B43B; $ink=0x292421; $muted=0x756B63; $light=0xF7F4EF; $white=0xFFFFFF; $orange=0x3C78E8; $red=0x3347C9; $green=0x57963C; $yellow=0x55C8F4
$font='Aptos'

function Add-Text($s,$txt,$x,$y,$w,$h,$size=20,$color=$ink,$bold=$false,$align=1){
  $sh=$s.Shapes.AddTextbox($msoTextOrientationHorizontal,$x,$y,$w,$h); $sh.TextFrame2.TextRange.Text=$txt
  $sh.TextFrame2.TextRange.Font.Name=$font; $sh.TextFrame2.TextRange.Font.Size=$size; $sh.TextFrame2.TextRange.Font.Fill.ForeColor.RGB=$color; $sh.TextFrame2.TextRange.Font.Bold=[int]$bold
  $sh.TextFrame2.TextRange.ParagraphFormat.Alignment=$align; $sh.TextFrame2.MarginLeft=2; $sh.TextFrame2.MarginRight=2; $sh.TextFrame2.MarginTop=1; $sh.TextFrame2.MarginBottom=1
  return $sh
}
function Add-Box($s,$x,$y,$w,$h,$fill=$white,$line=$blue,$radius=$true){
  $t=if($radius){$msoShapeRoundedRectangle}else{$msoShapeRectangle}; $sh=$s.Shapes.AddShape($t,$x,$y,$w,$h); $sh.Fill.ForeColor.RGB=$fill; $sh.Line.ForeColor.RGB=$line; $sh.Line.Weight=1.25; return $sh
}
function Add-Line($s,$x1,$y1,$x2,$y2,$color=$blue,$arrow=$true,$weight=2){$l=$s.Shapes.AddConnector($msoConnectorStraight,$x1,$y1,$x2,$y2);$l.Line.ForeColor.RGB=$color;$l.Line.Weight=$weight;if($arrow){$l.Line.EndArrowheadStyle=$msoArrowheadTriangle};return $l}
function Add-Title($s,$n,$title,$sub=''){
  Add-Text $s ("{0:D2}" -f $n) 42 26 40 28 12 $blue $true 1 | Out-Null
  Add-Text $s $title 82 19 815 45 28 $navy $true 1 | Out-Null
  $bar=$s.Shapes.AddShape($msoShapeRectangle,42,66,876,3);$bar.Fill.ForeColor.RGB=$cyan;$bar.Line.Visible=0
  if($sub){Add-Text $s $sub 44 73 870 24 12 $muted $false 1 | Out-Null}
  Add-Text $s ("PHÁT HIỆN VẬT CẢN • {0:D2}" -f $n) 740 510 178 16 9 $muted $false 3 | Out-Null
}
function Add-BulletList($s,$items,$x,$y,$w,$h,$size=18,$color=$ink){$txt=($items|ForEach-Object{"• $_"}) -join "`r"; $t=Add-Text $s $txt $x $y $w $h $size $color $false 1; $t.TextFrame2.TextRange.ParagraphFormat.SpaceAfter=8; return $t}
function Add-Image($s,$path,$x,$y,$w,$h){if(Test-Path -LiteralPath $path){$s.Shapes.AddPicture($path,$false,$true,$x,$y,$w,$h)|Out-Null}}
function Add-Tag($s,$txt,$x,$y,$w,$fill=$blue){$b=Add-Box $s $x $y $w 28 $fill $fill $true;$b.Line.Visible=0;Add-Text $s $txt ($x+6) ($y+4) ($w-12) 20 11 $white $true 2|Out-Null}
function Add-MetricBar($s,$label,$vals,$x,$y,$scale=340){
  Add-Text $s $label $x $y 100 24 15 $ink $true 1|Out-Null; $colors=@($blue,$teal,$orange); $names=@('E3','E4','E6')
  for($i=0;$i -lt 3;$i++){Add-Text $s $names[$i] ($x+103) ($y+$i*25) 30 20 11 $muted $true 1|Out-Null;$bw=$vals[$i]*$scale;$r=$s.Shapes.AddShape($msoShapeRectangle,$x+135,$y+$i*25+2,$bw,15);$r.Fill.ForeColor.RGB=$colors[$i];$r.Line.Visible=0;Add-Text $s ((100*$vals[$i]).ToString('0.0')+'%') ($x+140+$bw) ($y+$i*25) 48 19 11 $ink $true 1|Out-Null}
}
function Add-PerClassSlide($number,$start,$end){
  $s=New-Slide;Add-Title $s $number ("Kết quả TEST theo lớp • {0}–{1}" -f ($start+1),($end+1)) 'AP được tính trên nhiều ngưỡng confidence; P/R tổng dùng threshold vận hành đã khóa.'
  $headers=@('ID','Lớp','GT','E3 AP50 / AP50–95','E4 AP50 / AP50–95','E6 AP50 / AP50–95');$xs=@(45,90,320,385,545,705);$ws=@(45,230,65,160,160,160)
  for($c=0;$c -lt $headers.Count;$c++){Add-Box $s $xs[$c] 110 $ws[$c] 30 $navy $navy $false|Out-Null;Add-Text $s $headers[$c] ($xs[$c]+4) 117 ($ws[$c]-8) 18 11 $white $true 2|Out-Null}
  for($i=$start;$i -le $end;$i++){
    $y=140+($i-$start)*27;$fill=if(($i-$start)%2){0xF0ECE6}else{$white}
    $values=@($i,$classes[$i],$testModels[0].per_class[$i].gt_count,
      ((100*[double]$testModels[0].per_class[$i].ap50).ToString('0.0')+' / '+(100*[double]$testModels[0].per_class[$i].ap50_95).ToString('0.0')),
      ((100*[double]$testModels[1].per_class[$i].ap50).ToString('0.0')+' / '+(100*[double]$testModels[1].per_class[$i].ap50_95).ToString('0.0')),
      ((100*[double]$testModels[2].per_class[$i].ap50).ToString('0.0')+' / '+(100*[double]$testModels[2].per_class[$i].ap50_95).ToString('0.0')))
    for($c=0;$c -lt $values.Count;$c++){Add-Box $s $xs[$c] $y $ws[$c] 27 $fill 0xD8D0C8 $false|Out-Null;Add-Text $s ([string]$values[$c]) ($xs[$c]+4) ($y+5) ($ws[$c]-8) 17 10 $(if($c -eq 1){$navy}else{$ink}) $(if($c -eq 1){$true}else{$false}) $(if($c -eq 1){1}else{2})|Out-Null}
  }
}

$ppt=New-Object -ComObject PowerPoint.Application; $ppt.Visible=-1; $pres=$ppt.Presentations.Add(); $pres.PageSetup.SlideWidth=$W; $pres.PageSetup.SlideHeight=$H
function New-Slide(){ $sl=$pres.Slides.Add($pres.Slides.Count+1,$ppLayoutBlank); $sl.Background.Fill.ForeColor.RGB=$light; return $sl }

# 1 Cover
$s=New-Slide; $bg=$s.Shapes.AddShape($msoShapeRectangle,0,0,$W,$H);$bg.Fill.ForeColor.RGB=$navy;$bg.Line.Visible=0
$logo=Join-Path $root 'docs\reports\BAO_CAO_PHAT_HIEN_VAT_CAN_DAY_DU\_analysis\template_media\image1.jpg'; Add-Image $s $logo 54 38 82 80
Add-Text $s 'TRƯỜNG ĐẠI HỌC SƯ PHẠM KỸ THUẬT HƯNG YÊN' 155 45 720 28 16 $white $true 1|Out-Null
Add-Tag $s 'BÁO CÁO ĐỒ ÁN HỌC SÂU' 155 84 245 $teal
Add-Text $s 'PHÁT HIỆN VẬT CẢN TRONG ẢNH' 54 170 850 48 32 $white $true 1|Out-Null
Add-Text $s 'HƯỚNG TỚI HỖ TRỢ DI CHUYỂN CHO NGƯỜI KHIẾM THỊ' 54 220 820 88 30 $yellow $true 1|Out-Null
$ln=$s.Shapes.AddShape($msoShapeRectangle,54,326,250,4);$ln.Fill.ForeColor.RGB=$teal;$ln.Line.Visible=0
Add-Text $s "Sinh viên: Nguyễn Trần Kiên  |  MSV: 12423018  |  Lớp: 12423TN`rGiảng viên hướng dẫn: PGS.TS. Nguyễn Minh Tiến" 54 354 610 70 16 $white $false 1|Out-Null
Add-Text $s '2026' 54 475 120 25 14 $white $true 1|Out-Null
Add-Text $s 'YOLO11n  •  YOLO11s  •  Faster R-CNN' 478 472 430 25 13 $white $true 3|Out-Null

# 2 Problem
$s=New-Slide;Add-Title $s 2 'Đặt vấn đề' 'Phát hiện vật cản là lớp nhận thức thị giác — không phải một hệ thống dẫn đường hoàn chỉnh.'
Add-Image $s (Join-Path $root 'app\outputs\rod_627e1da0a7564caaaf1da499f0219c34.png') 515 112 395 330
Add-Text $s 'BÀI TOÁN' 48 118 140 22 12 $blue $true 1|Out-Null
Add-Text $s 'Nhận biết nhanh các vật thể có thể cản trở di chuyển trong một ảnh.' 48 148 410 72 24 $navy $true 1|Out-Null
Add-BulletList $s @('Môi trường đô thị có nhiều vật thể, kích thước và khoảng cách khác nhau.','Bỏ sót vật cản làm giảm giá trị hỗ trợ; cảnh báo sai quá nhiều gây nhiễu.','Mô hình phải cân bằng chất lượng phát hiện với thời gian và tài nguyên suy luận.') 48 246 420 150 17|Out-Null
Add-Tag $s 'Không thay thế gậy dò đường hay đánh giá an toàn thực tế' 48 425 420 $orange

# 3 objectives
$s=New-Slide;Add-Title $s 3 'Mục tiêu và phạm vi nghiên cứu'
Add-Text $s 'MỤC TIÊU' 50 106 220 28 13 $blue $true 1|Out-Null
Add-BulletList $s @('Chuẩn bị bộ dữ liệu phát hiện vật cản 25 lớp.','Fine-tune E3, E4 và E6 từ trọng số COCO.','So sánh chất lượng nhận diện và chi phí suy luận.','Xây dựng giao diện demo xử lý ảnh tĩnh.') 50 140 385 190 18|Out-Null
Add-Text $s 'PHẠM VI' 520 106 220 28 13 $teal $true 1|Out-Null
Add-BulletList $s @('Đầu vào: một ảnh JPG/JPEG/PNG.','Đầu ra: bbox, lớp, confidence và số lượng.','Không đo khoảng cách, GPS, âm thanh hay webcam.') 520 140 385 150 18|Out-Null
Add-Box $s 100 365 180 70 $white $blue|Out-Null;Add-Text $s 'ẢNH ĐẦU VÀO' 115 387 150 25 16 $navy $true 2|Out-Null
Add-Line $s 280 400 390 400 $blue $true 3|Out-Null;Add-Box $s 390 350 180 100 $navy $navy|Out-Null;Add-Text $s "MÔ HÌNH`rPHÁT HIỆN" 410 375 140 48 17 $white $true 2|Out-Null
Add-Line $s 570 400 680 400 $blue $true 3|Out-Null;Add-Box $s 680 365 180 70 $white $teal|Out-Null;Add-Text $s 'BBOX • LỚP • ĐỘ TIN CẬY' 695 382 150 38 14 $navy $true 2|Out-Null

# 4 dataset
$s=New-Slide;Add-Title $s 4 'Bộ dữ liệu ROD Final A4' 'Quy mô được xác minh từ manifest đã freeze.'
$vals=@(@('TRAIN','18.151','30.793'),@('VALID','3.318','5.320'),@('TEST','1.611','2.354'))
for($i=0;$i -lt 3;$i++){ $x=48+$i*205;Add-Box $s $x 112 180 92 $white @($blue,$teal,$orange)[$i]|Out-Null;Add-Text $s $vals[$i][0] ($x+12) 124 156 18 12 $muted $true 1|Out-Null;Add-Text $s ($vals[$i][1]+' ảnh') ($x+12) 145 156 28 22 $navy $true 1|Out-Null;Add-Text $s ($vals[$i][2]+' đối tượng') ($x+12) 176 156 18 12 $muted $false 1|Out-Null }
Add-Box $s 678 112 230 92 $navy $navy|Out-Null;Add-Text $s '23.080 ảnh' 696 130 196 28 24 $white $true 2|Out-Null;Add-Text $s '38.467 bbox • 25 lớp' 696 166 196 20 13 $yellow $true 2|Out-Null
$max=18151;for($i=0;$i -lt 3;$i++){$hgt=@(220,40,20)[$i];$x=85+$i*120;$r=$s.Shapes.AddShape($msoShapeRectangle,$x,460-$hgt,70,$hgt);$r.Fill.ForeColor.RGB=@($blue,$teal,$orange)[$i];$r.Line.Visible=0;Add-Text $s $vals[$i][0] ($x-5) 468 80 18 12 $ink $true 2|Out-Null}
Add-Text $s 'SỐ ẢNH THEO SPLIT' 52 250 340 24 13 $blue $true 1|Out-Null
Add-Image $s (Join-Path $root 'results\lam_sach_du_lieu\verify\viz\002_valid_img_22331.jpg') 460 232 210 238
Add-Image $s (Join-Path $root 'results\lam_sach_du_lieu\verify\viz\003_test_img_20274.jpg') 688 232 210 238

# 5 cleaning
$s=New-Slide;Add-Title $s 5 'Làm sạch và chuẩn bị dữ liệu' 'Mục tiêu: giảm lỗi nhãn và ngăn rò rỉ thông tin giữa các split.'
$steps=@('ROD nguồn','Kiểm tra ảnh / nhãn','Rà soát bbox','Kiểm tra trùng & leakage','Chuẩn hóa','ROD Final A4')
for($i=0;$i -lt 6;$i++){ $x=45+$i*148;Add-Box $s $x 150 125 64 $(if($i -eq 5){$navy}else{$white}) $(if($i -eq 5){$navy}else{$blue})|Out-Null;Add-Text $s $steps[$i] ($x+8) 166 109 38 14 $(if($i -eq 5){$white}else{$ink}) $true 2|Out-Null;if($i -lt 5){Add-Line $s ($x+125) 182 ($x+146) 182 $teal $true 2|Out-Null}}
Add-Text $s 'KẾT QUẢ ĐÃ GHI NHẬN' 48 275 280 24 13 $blue $true 1|Out-Null
Add-Box $s 48 310 260 100 $white $orange|Out-Null;Add-Text $s '178 ảnh loại khỏi TRAIN/VALID' 66 326 225 30 21 $navy $true 1|Out-Null;Add-Text $s '146 TRAIN  •  32 VALID' 66 370 225 22 14 $muted $false 1|Out-Null
Add-Box $s 335 310 260 100 $white $green|Out-Null;Add-Text $s '0 nhóm trùng chính xác giữa split' 353 326 225 42 20 $navy $true 1|Out-Null;Add-Text $s 'Kiểm soát leakage' 353 377 225 18 13 $muted $false 1|Out-Null
Add-Box $s 622 310 288 100 $white $yellow|Out-Null;Add-Text $s '271 nhóm trùng trong cùng split' 640 326 252 42 20 $navy $true 1|Out-Null;Add-Text $s 'Được giữ và ghi nhận' 640 377 252 18 13 $muted $false 1|Out-Null
Add-Text $s 'Các đề xuất sửa 18 bbox và nhãn Bike/Motorcycle còn ở trạng thái chờ, không trình bày như đã áp dụng.' 50 446 850 34 14 $red $true 1|Out-Null

# 6 workflow
$s=New-Slide;Add-Title $s 6 'Tổng quan phương pháp nghiên cứu'
Add-Box $s 55 220 150 74 $navy $navy|Out-Null;Add-Text $s 'ROD Final A4' 72 244 116 26 18 $white $true 2|Out-Null
$models=@("E3`rYOLO11n","E4`rYOLO11s","E6`rFaster R-CNN");$cols=@($blue,$teal,$orange)
for($i=0;$i -lt 3;$i++){ $x=280+$i*190;Add-Line $s 205 257 $x 168 $cols[$i] $true 2|Out-Null;Add-Box $s $x 125 155 86 $white $cols[$i]|Out-Null;Add-Text $s $models[$i] ($x+10) 145 135 48 17 $navy $true 2|Out-Null;Add-Line $s ($x+78) 211 ($x+78) 320 $cols[$i] $true 2|Out-Null}
Add-Box $s 280 320 535 65 $white $blue|Out-Null;Add-Text $s 'Fine-tune 10 epoch  →  chọn checkpoint & threshold bằng VALID' 300 340 495 26 17 $navy $true 2|Out-Null
Add-Line $s 548 385 548 430 $blue $true 3|Out-Null;Add-Box $s 320 430 455 55 $navy $navy|Out-Null;Add-Text $s 'Đánh giá chung  →  so sánh  →  ứng dụng ảnh tĩnh' 340 447 415 23 16 $white $true 2|Out-Null
Add-Text $s 'TRAIN' 55 335 100 24 13 $blue $true 2|Out-Null;Add-Text $s 'VALID' 55 370 100 24 13 $teal $true 2|Out-Null;Add-Text $s $(if($testComplete){'TEST: đã đánh giá cuối'}else{'TEST: khóa, chưa chạy'}) 55 405 190 24 13 $(if($testComplete){$green}else{$red}) $true 2|Out-Null

# 7 basics
$s=New-Slide;Add-Title $s 7 'Kiến thức nền: Object Detection'
Add-Box $s 50 122 250 170 $white $blue|Out-Null;Add-Text $s 'CLASSIFICATION' 68 142 214 22 14 $blue $true 2|Out-Null;Add-Text $s 'Ảnh → một nhãn' 68 190 214 38 26 $navy $true 2|Out-Null;Add-Text $s '“Có xe ô tô”' 68 248 214 24 16 $muted $false 2|Out-Null
Add-Box $s 355 122 555 170 $white $teal|Out-Null;Add-Text $s 'OBJECT DETECTION' 375 142 250 22 14 $teal $true 1|Out-Null
$im=Add-Box $s 378 180 170 80 $light $muted $false; $bb=$s.Shapes.AddShape($msoShapeRectangle,415,194,84,50);$bb.Fill.Visible=0;$bb.Line.ForeColor.RGB=$orange;$bb.Line.Weight=3
Add-Text $s 'bbox' 420 198 50 18 12 $orange $true 1|Out-Null;Add-Text $s 'vị trí + lớp + confidence' 580 193 290 46 23 $navy $true 1|Out-Null
Add-Text $s 'ONE-STAGE' 70 345 180 24 13 $blue $true 1|Out-Null;Add-Text $s 'Dự đoán trực tiếp trên feature maps' 70 380 330 60 20 $navy $true 1|Out-Null;Add-Tag $s 'YOLO11' 70 450 115 $blue
Add-Text $s 'TWO-STAGE' 520 345 180 24 13 $orange $true 1|Out-Null;Add-Text $s 'Đề xuất vùng → phân loại và tinh chỉnh bbox' 520 380 360 60 20 $navy $true 1|Out-Null;Add-Tag $s 'Faster R-CNN' 520 450 150 $orange

# 8 YOLO architecture
$s=New-Slide;Add-Title $s 8 'Kiến trúc YOLO11' 'Sơ đồ khái quát theo cấu hình Ultralytics: backbone – neck – detection head.'
Add-Box $s 40 145 105 70 $navy $navy|Out-Null;Add-Text $s "INPUT`r640×640" 53 163 80 38 15 $white $true 2|Out-Null
$blocks=@(@('Conv','Conv','C3k2','SPPF','C2PSA'),@('Upsample','Concat','C3k2','PAN down','C3k2'))
Add-Text $s 'BACKBONE' 175 105 250 22 13 $blue $true 2|Out-Null;for($i=0;$i -lt 5;$i++){ $x=170+$i*75;Add-Box $s $x 145 65 70 $white $blue|Out-Null;Add-Text $s $blocks[0][$i] ($x+4) 169 57 22 12 $navy $true 2|Out-Null;if($i -lt 4){Add-Line $s ($x+65) 180 ($x+75) 180 $blue $true 1|Out-Null}}
Add-Line $s 145 180 170 180 $blue $true 2|Out-Null
Add-Text $s 'NECK: HỢP NHẤT ĐA TỶ LỆ' 575 105 335 22 13 $teal $true 2|Out-Null;for($i=0;$i -lt 5;$i++){ $x=560+$i*75;Add-Box $s $x 145 65 70 $white $teal|Out-Null;Add-Text $s $blocks[1][$i] ($x+3) 164 59 38 11 $navy $true 2|Out-Null;if($i -lt 4){Add-Line $s ($x+65) 180 ($x+75) 180 $teal $true 1|Out-Null}}
Add-Line $s 535 180 560 180 $teal $true 2|Out-Null
$ys=@(290,350,410);$ps=@('P3 • vật thể nhỏ','P4 • trung bình','P5 • vật thể lớn');for($i=0;$i -lt 3;$i++){Add-Box $s 325 $ys[$i] 220 45 $white $cols[$i]|Out-Null;Add-Text $s $ps[$i] 340 ($ys[$i]+12) 190 20 14 $navy $true 2|Out-Null;Add-Line $s 775 215 545 ($ys[$i]+22) $cols[$i] $true 2|Out-Null;Add-Line $s 545 ($ys[$i]+22) 635 ($ys[$i]+22) $cols[$i] $true 2|Out-Null}
Add-Box $s 635 312 130 95 $navy $navy|Out-Null;Add-Text $s "HEAD`rbbox + lớp" 651 336 98 50 16 $white $true 2|Out-Null;Add-Line $s 765 360 810 360 $orange $true 2|Out-Null;Add-Box $s 810 325 110 70 $white $orange|Out-Null;Add-Text $s "NMS`rKết quả" 825 344 80 38 15 $navy $true 2|Out-Null

# 9 scale
$s=New-Slide;Add-Title $s 9 'YOLO11n và YOLO11s khác nhau thế nào?' 'Cùng họ kiến trúc; khác quy mô kênh/khối, số tham số và chi phí tính toán.'
Add-Text $s 'YOLO11n' 70 122 200 28 21 $blue $true 1|Out-Null;Add-Text $s 'YOLO11s' 70 253 200 28 21 $teal $true 1|Out-Null
for($i=0;$i -lt 6;$i++){ $ww=28+$i*4;$r=$s.Shapes.AddShape($msoShapeRectangle,220+$i*50,124,$ww,52);$r.Fill.ForeColor.RGB=$blue;$r.Line.Visible=0;$ww2=38+$i*7;$r2=$s.Shapes.AddShape($msoShapeRectangle,220+$i*70,250,$ww2,70);$r2.Fill.ForeColor.RGB=$teal;$r2.Line.Visible=0}
Add-Text $s '2,59 M tham số' 665 132 210 30 22 $navy $true 3|Out-Null;Add-Text $s '9,44 M tham số' 665 270 210 30 22 $navy $true 3|Out-Null
Add-Box $s 70 365 805 95 $white $blue|Out-Null
Add-Text $s 'QUAN SÁT THỰC NGHIỆM' 90 380 250 20 12 $blue $true 1|Out-Null;Add-Text $s ("Trên ${resultSplit}, E4 tăng mAP50–95 +"+((100*($resultM5095[1]-$resultM5095[0])).ToString('0.00'))+' điểm %, nhưng CPU chậm hơn 2,44× so với E3.') 90 408 750 34 20 $navy $true 1|Out-Null
Add-Text $s 'FLOPs/depth/width không được chèn nếu chưa truy vết chắc chắn từ checkpoint/source cục bộ.' 90 470 750 20 11 $muted $false 1|Out-Null

# 10 Faster architecture
$s=New-Slide;Add-Title $s 10 'Kiến trúc Faster R-CNN' 'Two-stage detector trong Torchvision.'
$nodes=@(@('Ảnh',45,205,85,$navy),@('ResNet-50',155,190,110,$blue),@('FPN',295,190,90,$teal),@('RPN',420,135,105,$orange),@("Region`rProposals",555,135,110,$orange),@('RoI Align',420,285,105,$cyan),@('RoI Head',555,285,110,$blue),@('Detections',715,205,150,$navy))
foreach($n in $nodes){Add-Box $s $n[1] $n[2] $n[3] 70 $(if($n[4]-eq $navy){$navy}else{$white}) $n[4]|Out-Null;Add-Text $s $n[0] ($n[1]+7) ($n[2]+20) ($n[3]-14) 35 15 $(if($n[4]-eq $navy){$white}else{$navy}) $true 2|Out-Null}
Add-Line $s 130 240 155 225 $blue $true 2|Out-Null;Add-Line $s 265 225 295 225 $teal $true 2|Out-Null;Add-Line $s 385 215 420 170 $orange $true 2|Out-Null;Add-Line $s 525 170 555 170 $orange $true 2|Out-Null;Add-Line $s 610 205 472 285 $cyan $true 2|Out-Null;Add-Line $s 525 320 555 320 $blue $true 2|Out-Null;Add-Line $s 665 320 715 250 $navy $true 2|Out-Null
Add-Text $s 'objectness + bbox' 423 110 125 18 11 $orange $true 2|Out-Null;Add-Text $s 'phân loại + tinh chỉnh bbox' 548 366 190 24 11 $blue $true 2|Out-Null
Add-BulletList $s @('ResNet-50 trích xuất đặc trưng; FPN kết hợp nhiều độ phân giải.','RPN sinh vùng ứng viên; RoI Align lấy đặc trưng vùng chính xác.','RoI Head phân loại và hồi quy bbox cho kết quả cuối.') 80 420 800 70 15|Out-Null

# 11 choices
$s=New-Slide;Add-Title $s 11 'Vì sao lựa chọn ba mô hình?'
$cards=@(@('YOLO11n','Nhỏ gọn','Ứng viên tài nguyên hạn chế',$blue),@('YOLO11s','Năng lực lớn hơn','Kiểm tra lợi ích tăng quy mô',$teal),@('Faster R-CNN','Two-stage','Đối chứng khác phương pháp',$orange))
for($i=0;$i -lt 3;$i++){$x=50+$i*300;Add-Box $s $x 125 260 150 $white $cards[$i][3]|Out-Null;Add-Text $s $cards[$i][0] ($x+18) 145 224 30 22 $navy $true 1|Out-Null;Add-Text $s $cards[$i][1] ($x+18) 185 224 24 16 $cards[$i][3] $true 1|Out-Null;Add-Text $s $cards[$i][2] ($x+18) 222 224 38 15 $muted $false 1|Out-Null}
Add-Text $s 'CÂU HỎI NGHIÊN CỨU' 50 330 270 24 13 $blue $true 1|Out-Null
Add-Box $s 50 365 410 95 $navy $navy|Out-Null;Add-Text $s '01' 70 386 45 35 24 $yellow $true 1|Out-Null;Add-Text $s 'Nano → Small có đáng đổi lấy chi phí tính toán?' 120 383 315 52 19 $white $true 1|Out-Null
Add-Box $s 500 365 410 95 $white $orange|Out-Null;Add-Text $s '02' 520 386 45 35 24 $orange $true 1|Out-Null;Add-Text $s 'One-stage và two-stage khác nhau thế nào về chất lượng/tốc độ?' 570 378 315 65 18 $navy $true 1|Out-Null

# 12 training config
$s=New-Slide;Add-Title $s 12 'Cấu hình huấn luyện' 'Cả ba dùng COCO pretrained và hoàn thành 10 epoch; chưa đủ cơ sở kết luận hội tụ tối ưu.'
$headers=@('Thuộc tính','E3 • YOLO11n','E4 • YOLO11s','E6 • Faster R-CNN')
$rows=@(@('Pretrained','COCO','COCO','COCO V2'),@('Epoch','10','10','10 (ghi nhận)'),@('Batch size','8','8','2'),@('Kích thước','640','640','min 640 / max 960'),@('Optimizer','SGD','SGD','SGD'),@('LR / scheduler','.01 → .0001','.01 → .0001','.005; StepLR 3×0,1'),@('Checkpoint','best.pt','best.pt','best weights'))
$x0=45;$y0=120;$cw=@(180,230,230,230);$rh=42
for($c=0;$c -lt 4;$c++){ $x=$x0+($cw[0..($c-1)]|Measure-Object -Sum).Sum;if($c -eq 0){$x=$x0};Add-Box $s $x $y0 $cw[$c] $rh $navy $navy $false|Out-Null;Add-Text $s $headers[$c] ($x+8) ($y0+11) ($cw[$c]-16) 20 13 $white $true 2|Out-Null}
for($r=0;$r -lt $rows.Count;$r++){$yy=$y0+$rh*($r+1);for($c=0;$c -lt 4;$c++){$x=$x0;if($c -gt 0){$x=$x0+($cw[0..($c-1)]|Measure-Object -Sum).Sum};Add-Box $s $x $yy $cw[$c] $rh $(if($r%2){0xF0ECE6}else{$white}) 0xD8D0C8 $false|Out-Null;Add-Text $s $rows[$r][$c] ($x+8) ($yy+10) ($cw[$c]-16) 22 12 $(if($c -eq 0){$navy}else{$ink}) $(if($c -eq 0){$true}else{$false}) 2|Out-Null}}
Add-Text $s 'Khác biệt pipeline và kích thước ảnh cần được cân nhắc khi diễn giải kết quả.' 50 468 850 25 13 $red $true 1|Out-Null

# 13 evaluation
$s=New-Slide;Add-Title $s 13 'Phương pháp đánh giá' $(if($testComplete){'Ba mô hình được đánh giá cuối trên cùng TEST bằng evaluator và cấu hình đã khóa từ VALID.'}else{'Ba mô hình được đánh giá trên cùng VALID bằng evaluator chung; TEST chưa chạy.'})
$metrics=@(@('Precision','TP / (TP + FP)'),@('Recall','TP / (TP + FN)'),@('mAP50','AP tại IoU = 0,50'),@('mAP50–95','Trung bình AP từ 0,50 đến 0,95'),@('Latency / FPS','Chi phí suy luận'))
for($i=0;$i -lt 5;$i++){$x=45+($i%3)*295;$y=115+[math]::Floor($i/3)*105;Add-Box $s $x $y 260 78 $white @($blue,$teal,$orange,$cyan,$green)[$i]|Out-Null;Add-Text $s $metrics[$i][0] ($x+15) ($y+12) 230 24 16 $navy $true 1|Out-Null;Add-Text $s $metrics[$i][1] ($x+15) ($y+43) 230 20 13 $muted $false 1|Out-Null}
Add-Box $s 45 345 560 125 $navy $navy|Out-Null;Add-Text $s 'Vì sao Recall quan trọng?' 65 363 300 26 21 $yellow $true 1|Out-Null;Add-Text $s 'False Negative = vật cản có thật nhưng bị bỏ sót. Vì vậy ngưỡng vận hành được chọn theo F2 để ưu tiên Recall hơn Precision.' 65 399 515 56 17 $white $false 1|Out-Null
Add-Box $s 640 345 270 125 $white $orange|Out-Null;Add-Text $s 'GIAO THỨC' 658 363 230 20 12 $orange $true 1|Out-Null;Add-Text $s "IoU 0,50 • AP 101 điểm`rmax_det AP = 100`rP/R/F2 max_det = 300" 658 393 230 62 14 $navy $true 1|Out-Null

# 14 results
$s=New-Slide;Add-Title $s 14 ("Kết quả thực nghiệm trên $resultSplit") $(if($testComplete){'Đánh giá cuối: 1.611 ảnh, 2.354 đối tượng; checkpoint, threshold và protocol đã khóa trước khi mở TEST.'}else{'Cùng ground truth, cùng protocol COCO-style tùy biến; không phải TEST.'})
Add-MetricBar $s 'Precision' $resultP 50 125 300
Add-MetricBar $s 'Recall' $resultR 500 125 300
Add-MetricBar $s 'mAP50' $resultM50 50 280 300
Add-MetricBar $s 'mAP50–95' $resultM5095 500 280 300
Add-Tag $s 'E3' 280 465 55 $blue;Add-Tag $s 'E4' 350 465 55 $teal;Add-Tag $s 'E6' 420 465 55 $orange
$best=0;for($i=1;$i -lt 3;$i++){if($resultM5095[$i] -gt $resultM5095[$best]){$best=$i}};$bestName=@('E3','E4','E6')[$best]
Add-Text $s ("$bestName có mAP50–95 cao nhất trên ${resultSplit}: "+((100*$resultM5095[$best]).ToString('0.00'))+'%. Kết luận tốc độ vẫn dựa trên benchmark cùng thiết bị.') 520 445 385 48 14 $navy $true 1|Out-Null

if($testComplete){Add-PerClassSlide 15 0 12;Add-PerClassSlide 16 13 24}
$mainOffset=if($testComplete){2}else{0}
$appendixOffset=if($testComplete){3}else{0}

# 15 compute
$s=New-Slide;Add-Title $s (15+$mainOffset) 'Hiệu quả tính toán và tài nguyên' 'Benchmark CPU, 30 ảnh VALID, batch 1, warmup 3; không đại diện cho GPU/thiết bị biên.'
Add-Text $s 'ĐỘ TRỄ TRUNG BÌNH (ms/ảnh)' 50 112 380 22 13 $blue $true 1|Out-Null
$lat=@(76.26,185.83,2511.30);$fps=@(13.11,5.38,.398);$pars=@(2.59,9.44,43.38);$names=@('E3','E4','E6');$cc=@($blue,$teal,$orange)
for($i=0;$i -lt 3;$i++){$y=155+$i*75;Add-Text $s $names[$i] 55 $y 40 20 13 $navy $true 1|Out-Null;$bw=[math]::Max(8,[math]::Log10($lat[$i]+1)*145);$r=$s.Shapes.AddShape($msoShapeRectangle,105,$y,$bw,24);$r.Fill.ForeColor.RGB=$cc[$i];$r.Line.Visible=0;Add-Text $s ($lat[$i].ToString('0.00')+' ms') (115+$bw) ($y+2) 110 20 12 $ink $true 1|Out-Null}
Add-Box $s 515 120 395 245 $white $blue|Out-Null;Add-Text $s 'Accuracy ↔ Latency' 538 140 340 26 20 $navy $true 1|Out-Null
$px=@(570,660,825);$py=@(310,278,190);for($i=0;$i -lt 3;$i++){$o=$s.Shapes.AddShape($msoShapeOval,$px[$i],$py[$i],24,24);$o.Fill.ForeColor.RGB=$cc[$i];$o.Line.Visible=0;Add-Text $s ($names[$i]+"`r"+(($resultM5095[$i]*100).ToString('0.0'))+'% mAP') ($px[$i]-28) ($py[$i]-44) 90 40 12 $navy $true 2|Out-Null}
Add-Line $s 555 330 875 330 $muted $true 1|Out-Null;Add-Line $s 555 330 555 170 $muted $true 1|Out-Null;Add-Text $s 'độ trễ →' 780 337 90 20 11 $muted $false 1|Out-Null;Add-Text $s 'mAP ↑' 520 158 55 20 11 $muted $false 1|Out-Null
Add-Text $s 'Tham số / FPS' 50 390 300 22 13 $blue $true 1|Out-Null;for($i=0;$i -lt 3;$i++){Add-Text $s ($names[$i]+': '+$pars[$i]+' M  •  '+$fps[$i]+' FPS') 55 (420+$i*25) 300 22 14 $ink $true 1|Out-Null}
Add-Tag $s ("E3 ưu tiên tốc độ CPU; chất lượng báo cáo theo $resultSplit.") 515 405 395 $navy

# 16 demo
$s=New-Slide;Add-Title $s (16+$mainOffset) 'Ứng dụng demo phát hiện vật cản' 'Ứng dụng Gradio hiện hỗ trợ E3, E4 và E6 trên ảnh tĩnh.'
Add-Image $s (Join-Path $root 'app\outputs\rod_a1938e5301034a6a8ea2a9d6e165721b.png') 470 108 430 340
$flow=@('Upload ảnh','Chọn mô hình','Nhận diện','Hiển thị kết quả','Tải PNG');for($i=0;$i -lt 5;$i++){$y=115+$i*72;Add-Box $s 55 $y 270 48 $(if($i -eq 2){$navy}else{$white}) $(if($i -eq 2){$navy}else{$blue})|Out-Null;Add-Text $s $flow[$i] 72 ($y+13) 236 22 16 $(if($i -eq 2){$white}else{$navy}) $true 2|Out-Null;if($i -lt 4){Add-Line $s 190 ($y+48) 190 ($y+69) $teal $true 2|Out-Null}}
Add-Text $s 'Đầu ra: bbox • lớp • confidence • số lượng • inference/server time' 465 465 435 28 13 $muted $true 1|Out-Null

# 17 findings
$s=New-Slide;Add-Title $s (17+$mainOffset) 'Kết quả, hạn chế và hướng phát triển'
$colsX=@(45,350,655);$heads=@('ĐÃ THỰC HIỆN','HẠN CHẾ','HƯỚNG PHÁT TRIỂN');$c=@($blue,$orange,$teal)
$lists=@(@('Làm sạch và freeze ROD Final A4','Fine-tune 3 mô hình, 10 epoch',$(if($testComplete){'TEST cuối + benchmark CPU'}else{'VALID chung + benchmark CPU'}),'Demo ảnh tĩnh hỗ trợ cả 3 model'),@('Một lần chạy; chưa chứng minh hội tụ','E6 thiếu log training rời đầy đủ',$(if($testComplete){'TEST chỉ dùng một lần sau khi khóa cấu hình'}else{'TEST chưa chạy'}),'Chưa đánh giá an toàn/real-time trên thiết bị đích'),@('Webcam và tối ưu inference','Edge deployment','Thử thiếu sáng, motion blur','Ước lượng khoảng cách và âm thanh'))
for($i=0;$i -lt 3;$i++){Add-Box $s $colsX[$i] 120 260 335 $white $c[$i]|Out-Null;Add-Text $s $heads[$i] ($colsX[$i]+18) 140 224 25 15 $c[$i] $true 1|Out-Null;Add-BulletList $s $lists[$i] ($colsX[$i]+18) 185 224 235 16|Out-Null}
Add-Text $s 'Các hướng phát triển là đề xuất nghiên cứu tiếp, không phải tính năng đã hoàn thành.' 50 474 850 24 13 $red $true 1|Out-Null

# 18 conclusion
$s=New-Slide;$bg=$s.Shapes.AddShape($msoShapeRectangle,0,0,$W,$H);$bg.Fill.ForeColor.RGB=$navy;$bg.Line.Visible=0
Add-Text $s 'KẾT LUẬN' 55 50 300 30 16 $yellow $true 1|Out-Null
Add-Text $s 'Ba mô hình cho ba điểm vận hành khác nhau.' 55 105 820 48 31 $white $true 1|Out-Null
Add-Text $s ("Ba mô hình đã được so sánh bằng cùng evaluator. $bestName đạt mAP50–95 $resultSplit cao nhất; E3 có chi phí suy luận CPU thấp nhất trong benchmark hiện có.") 55 180 810 80 22 $white $false 1|Out-Null
Add-Box $s 55 305 250 105 $white $white|Out-Null;Add-Text $s 'E3' 75 324 55 28 22 $blue $true 1|Out-Null;Add-Text $s 'ưu tiên tốc độ' 75 364 200 24 16 $ink $true 1|Out-Null
Add-Box $s 355 305 250 105 $white $white|Out-Null;Add-Text $s 'E4' 375 324 55 28 22 $teal $true 1|Out-Null;Add-Text $s 'điểm cân bằng trung gian' 375 360 200 42 16 $ink $true 1|Out-Null
Add-Box $s 655 305 250 105 $white $white|Out-Null;Add-Text $s 'E6' 675 324 55 28 22 $orange $true 1|Out-Null;Add-Text $s ("đối chiếu chất lượng $resultSplit") 675 360 200 42 16 $ink $true 1|Out-Null
Add-Text $s 'XIN CẢM ƠN HỘI ĐỒNG' 55 468 850 30 18 $yellow $true 2|Out-Null

# 19 appendix classes
$s=New-Slide;Add-Title $s (19+$mainOffset) 'Phụ lục A • 25 lớp ROD Final A4'
for($i=0;$i -lt 25;$i++){$col=[math]::Floor($i/9);$row=$i%9;$x=70+$col*290;$y=112+$row*40;Add-Text $s ((($i+1).ToString('00'))+'  '+$classes[$i]) $x $y 250 24 16 $(if($i%2){$ink}else{$navy}) $(if($i%2){$false}else{$true}) 1|Out-Null}
Add-Text $s 'Tên lớp giữ nguyên theo cấu hình dataset để bảo toàn ánh xạ checkpoint.' 70 480 820 22 12 $muted $false 1|Out-Null

if($testComplete){
  $s=New-Slide;Add-Title $s 22 'Phụ lục • Kết quả VALID' 'VALID chỉ dùng chọn checkpoint/ngưỡng và kiểm tra pipeline; bảng kết quả chính dùng TEST đã khóa.'
  Add-MetricBar $s 'Precision' $validP 50 125 300
  Add-MetricBar $s 'Recall' $validR 500 125 300
  Add-MetricBar $s 'mAP50' $validM50 50 280 300
  Add-MetricBar $s 'mAP50–95' $validM5095 500 280 300
  Add-Text $s 'Threshold vận hành chọn bằng F2 trên VALID: E3 = 0,22 • E4 = 0,19 • E6 = 0,59.' 150 458 660 28 14 $navy $true 2|Out-Null
}

# 20 blocks
$s=New-Slide;Add-Title $s (20+$appendixOffset) 'Phụ lục B • C3k2, SPPF và C2PSA trong YOLO11'
Add-Box $s 50 125 250 250 $white $blue|Out-Null;Add-Text $s 'C3k2' 70 145 210 30 23 $blue $true 1|Out-Null;Add-Text $s 'Khối CSP hiệu quả, kết hợp nhánh tắt và các bottleneck/kernels để tái sử dụng đặc trưng với chi phí hợp lý.' 70 195 210 130 17 $ink $false 1|Out-Null
Add-Box $s 355 125 250 250 $white $teal|Out-Null;Add-Text $s 'SPPF' 375 145 210 30 23 $teal $true 1|Out-Null;Add-Text $s 'Spatial Pyramid Pooling Fast mở rộng trường tiếp nhận bằng pooling nối tiếp, gom ngữ cảnh đa tỉ lệ ở cuối backbone.' 375 195 210 130 17 $ink $false 1|Out-Null
Add-Box $s 660 125 250 250 $white $orange|Out-Null;Add-Text $s 'C2PSA' 680 145 210 30 23 $orange $true 1|Out-Null;Add-Text $s 'Khối attention định vị không gian giúp mô hình nhấn mạnh vùng đặc trưng hữu ích trước khi chuyển sang neck.' 680 195 210 130 17 $ink $false 1|Out-Null
Add-Line $s 300 250 355 250 $blue $true 2|Out-Null;Add-Line $s 605 250 660 250 $teal $true 2|Out-Null
Add-Tag $s 'Vai trò chung: biểu diễn đặc trưng tốt hơn trước khi hợp nhất P3/P4/P5.' 190 425 580 $navy

# 21 Faster details
$s=New-Slide;Add-Title $s (21+$appendixOffset) 'Phụ lục C • FPN, RPN và RoI Align'
$items=@(@('FPN','Tạo pyramid P2–P5 từ đặc trưng ResNet; đường top-down đưa ngữ nghĩa mạnh về mức phân giải cao.',$teal),@('RPN','Đánh giá objectness và hồi quy anchor để tạo region proposals dùng chung đặc trưng.',$orange),@('RoI Align','Lấy mẫu đặc trưng vùng bằng nội suy, tránh lượng tử hóa thô của RoI Pooling.',$blue))
for($i=0;$i -lt 3;$i++){$x=55+$i*300;Add-Box $s $x 125 260 275 $white $items[$i][2]|Out-Null;Add-Text $s $items[$i][0] ($x+20) 150 220 32 25 $items[$i][2] $true 1|Out-Null;Add-Text $s $items[$i][1] ($x+20) 210 220 125 17 $ink $false 1|Out-Null}
Add-Box $s 105 435 750 45 $navy $navy|Out-Null;Add-Text $s 'FPN → RPN → proposals → RoI Align → classification + box regression' 125 447 710 22 16 $white $true 2|Out-Null

# 22 metrics details
$s=New-Slide;Add-Title $s (22+$appendixOffset) 'Phụ lục D • TP, FP, FN, IoU và mAP'
$defs=@(@('TP','Đúng lớp, IoU đạt ngưỡng'),@('FP','Dự đoán sai hoặc trùng'),@('FN','Vật thể thật bị bỏ sót'))
for($i=0;$i -lt 3;$i++){$x=55+$i*300;Add-Box $s $x 120 250 90 $white @($green,$orange,$red)[$i]|Out-Null;Add-Text $s $defs[$i][0] ($x+15) 137 50 28 22 @($green,$orange,$red)[$i] $true 1|Out-Null;Add-Text $s $defs[$i][1] ($x+70) 137 160 45 14 $ink $false 1|Out-Null}
Add-Text $s 'IoU = diện tích giao / diện tích hợp' 60 255 390 40 24 $navy $true 1|Out-Null;Add-Text $s 'Precision = TP / (TP + FP)' 60 325 390 35 20 $blue $true 1|Out-Null;Add-Text $s 'Recall = TP / (TP + FN)' 60 380 390 35 20 $teal $true 1|Out-Null
Add-Box $s 520 255 360 170 $navy $navy|Out-Null;Add-Text $s 'mAP50–95' 545 278 310 35 25 $yellow $true 1|Out-Null;Add-Text $s 'Tính AP ở 10 ngưỡng IoU từ 0,50 đến 0,95 (bước 0,05), rồi lấy trung bình qua lớp và ngưỡng.' 545 330 310 75 17 $white $false 1|Out-Null
Add-Text $s 'F2 đặt trọng số cho Recall lớn hơn Precision khi chọn ngưỡng vận hành.' 60 455 820 25 14 $red $true 1|Out-Null

# 23 evidence/source
$s=New-Slide;Add-Title $s (23+$appendixOffset) 'Phụ lục E • Truy vết số liệu và tài liệu tham khảo'
Add-Text $s 'NGUỒN NỘI BỘ' 50 110 250 22 13 $blue $true 1|Out-Null
Add-BulletList $s @('FACTS.md và manifest ROD Final A4','results/E3_YOLO11n, E4_YOLO11s, E6_Faster_RCNN','app/evaluator.py, evaluation_lock.json','app/outputs và báo cáo kiểm thử ứng dụng') 50 145 410 150 16|Out-Null
Add-Text $s 'TÀI LIỆU KỸ THUẬT' 510 110 250 22 13 $teal $true 1|Out-Null
Add-BulletList $s @('Redmon et al. — You Only Look Once (2016)','Ultralytics YOLO11 documentation & source','Ren et al. — Faster R-CNN (2015)','He et al. — ResNet (2016)','Lin et al. — Feature Pyramid Networks (2017)','Torchvision fasterrcnn_resnet50_fpn_v2 documentation') 510 145 405 220 15|Out-Null
Add-Box $s 50 380 860 78 $white $orange|Out-Null;Add-Text $s 'Lưu ý: YOLO11 không được gán một “paper chính thức” không tồn tại; kiến trúc được đối chiếu theo documentation/source Ultralytics.' 70 402 820 38 16 $navy $true 1|Out-Null

# Notes placeholder with source citations in each slide notes if supported
for($i=1;$i -le $pres.Slides.Count;$i++){try{$notes=$pres.Slides.Item($i).NotesPage.Shapes.Placeholders.Item(2);$notes.TextFrame.TextRange.Text="Nguồn: README.md; docs/reports/.../FACTS.md; results/; app/. Xem LOI_THUYET_TRINH.md để có lời trình bày và nguồn chi tiết."}catch{}}

if(Test-Path -LiteralPath $pptxPath){Remove-Item -LiteralPath $pptxPath -Force};if(Test-Path -LiteralPath $pdfPath){Remove-Item -LiteralPath $pdfPath -Force}
$slideCount=$pres.Slides.Count
$pres.SaveAs($pptxPath,24)
$pres.SaveAs($pdfPath,32)
for($i=1;$i -le $pres.Slides.Count;$i++){$png=Join-Path $renderDir ('slide-{0:D2}.png' -f $i);$pres.Slides.Item($i).Export($png,'PNG',1600,900)}
$pres.Close();$ppt.Quit();[System.Runtime.Interopservices.Marshal]::ReleaseComObject($pres)|Out-Null;[System.Runtime.Interopservices.Marshal]::ReleaseComObject($ppt)|Out-Null
Write-Output $pptxPath;Write-Output $pdfPath;Write-Output ("Slides: "+$slideCount)

