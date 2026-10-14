$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Drawing
$out=Join-Path (Split-Path $PSScriptRoot -Parent) 'faster_rcnn_architecture.png'
$bmp=New-Object Drawing.Bitmap 1800,820
$g=[Drawing.Graphics]::FromImage($bmp);$g.SmoothingMode='AntiAlias';$g.Clear([Drawing.Color]::White)
$title=[Drawing.Font]::new('Arial',[single]28,[Drawing.FontStyle]::Bold)
$font=[Drawing.Font]::new('Arial',[single]19,[Drawing.FontStyle]::Bold)
$small=[Drawing.Font]::new('Arial',[single]15)
$note=[Drawing.Font]::new('Arial',[single]13)
$navy=[Drawing.Color]::FromArgb(31,78,121);$blue=[Drawing.Color]::FromArgb(91,155,213);$teal=[Drawing.Color]::FromArgb(42,145,130);$orange=[Drawing.Color]::FromArgb(237,125,49);$light=[Drawing.Color]::FromArgb(235,242,249);$ink=[Drawing.Color]::FromArgb(32,32,32)
$g.DrawString('KIẾN TRÚC FASTER R-CNN RESNET-50 FPN V2',$title,(New-Object Drawing.SolidBrush $navy),355,30)
function Box($x,$y,$w,$h,$fill,$label,$sub=''){
  $g.FillRectangle((New-Object Drawing.SolidBrush $fill),$x,$y,$w,$h);$g.DrawRectangle((New-Object Drawing.Pen $navy,3),$x,$y,$w,$h)
  $sf=New-Object Drawing.StringFormat;$sf.Alignment='Center';$sf.LineAlignment='Center'
  $g.DrawString($label,$font,(New-Object Drawing.SolidBrush ([Drawing.Color]::White)),(New-Object Drawing.RectangleF $x,$y,$w,($h-22)),$sf)
  if($sub){$g.DrawString($sub,$note,(New-Object Drawing.SolidBrush ([Drawing.Color]::White)),(New-Object Drawing.RectangleF $x,($y+$h-34),$w,28),$sf)}
}
function Arrow($x1,$y1,$x2,$y2){$p=New-Object Drawing.Pen $ink,4;$p.CustomEndCap=New-Object Drawing.Drawing2D.AdjustableArrowCap 7,9;$g.DrawLine($p,$x1,$y1,$x2,$y2)}
Box 35 300 170 110 $blue 'ẢNH ĐẦU VÀO' 'RGB'
Box 270 270 230 170 $navy 'RESNET-50' 'Backbone C2-C5'
Box 570 245 230 220 $teal 'FPN' 'P2 - P5 đa tỉ lệ'
Box 890 145 245 150 $orange 'RPN' 'Objectness + box'
Box 890 485 245 130 $blue 'PROPOSALS' 'Vùng ứng viên'
Box 1215 300 220 140 $teal 'ROI ALIGN' 'Đặc trưng cố định'
Box 1510 195 245 130 $navy 'CLASSIFICATION' '26 lớp gồm nền'
Box 1510 455 245 130 $orange 'BOX REGRESSION' 'Tinh chỉnh bbox'
Arrow 205 355 270 355;Arrow 500 355 570 355;Arrow 800 315 890 220;Arrow 800 395 890 550;Arrow 1012 295 1012 485;Arrow 1135 550 1215 385;Arrow 1135 220 1215 350;Arrow 1435 350 1510 260;Arrow 1435 390 1510 520
$g.DrawString('Feature maps',$small,(New-Object Drawing.SolidBrush $ink),805,325)
$g.DrawString('RoI features',$small,(New-Object Drawing.SolidBrush $ink),1300,452)
$g.DrawString('Đầu ra: lớp, confidence và bounding box',$small,(New-Object Drawing.SolidBrush $ink),1220,675)
$g.DrawString('Luồng hai giai đoạn: đề xuất vùng (RPN) rồi phân loại và hồi quy hộp trên từng RoI.',$small,(New-Object Drawing.SolidBrush $ink),400,750)
$bmp.Save($out,[Drawing.Imaging.ImageFormat]::Png);$g.Dispose();$bmp.Dispose();Get-Item $out|Select FullName,Length


