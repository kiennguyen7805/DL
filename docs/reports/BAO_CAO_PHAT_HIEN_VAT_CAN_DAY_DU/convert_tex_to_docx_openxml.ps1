$ErrorActionPreference='Stop'
$here=$PSScriptRoot
$project=(Resolve-Path (Join-Path $here '..\..\..')).Path
$source=Join-Path $here 'PAPER_PHAT_HIEN_VAT_CAN.tex'
$template=Join-Path $here 'BAO_CAO_PHAT_HIEN_VAT_CAN_PAPER.docx'
$output=Join-Path $here 'PAPER_PHAT_HIEN_VAT_CAN_NGUYEN_TRAN_KIEN.docx'
$temp=Join-Path $env:TEMP ('rod_docx_'+[guid]::NewGuid().ToString('N'))

function X([string]$s){[System.Security.SecurityElement]::Escape($s)}
function Para([string]$text,[string]$style='Normal',[string]$align='both'){
  if([string]::IsNullOrWhiteSpace($text)){return ''}
  $size=if($style -eq 'Title'){'32'}elseif($style -eq 'Subtitle'){'24'}elseif($style -eq 'Heading1'){'28'}elseif($style -eq 'Heading2'){'24'}else{'22'}
  $bold=if($style -in @('Title','Heading1','Heading2')){'<w:b/>'}else{''}
  '<w:p><w:pPr><w:pStyle w:val="'+$style+'"/><w:jc w:val="'+$align+'"/><w:spacing w:after="100" w:line="276" w:lineRule="auto"/></w:pPr><w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>'+$bold+'<w:sz w:val="'+$size+'"/><w:szCs w:val="'+$size+'"/></w:rPr><w:t xml:space="preserve">'+(X $text)+'</w:t></w:r></w:p>'
}
function Cell([string]$text,[bool]$head=$false){
  $shade=if($head){'<w:shd w:fill="1F4E78"/>'}else{''};$color=if($head){'<w:color w:val="FFFFFF"/>'}else{''};$bold=if($head){'<w:b/>'}else{''}
  '<w:tc><w:tcPr><w:tcMar><w:top w:w="100" w:type="dxa"/><w:left w:w="100" w:type="dxa"/><w:bottom w:w="100" w:type="dxa"/><w:right w:w="100" w:type="dxa"/></w:tcMar>'+$shade+'</w:tcPr><w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>'+$bold+$color+'<w:sz w:val="20"/></w:rPr><w:t>'+(X $text)+'</w:t></w:r></w:p></w:tc>'
}
function TableXml($headers,$rows){
  $xml='<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/><w:tblBorders><w:top w:val="single" w:sz="4" w:color="D9D9D9"/><w:left w:val="single" w:sz="4" w:color="D9D9D9"/><w:bottom w:val="single" w:sz="4" w:color="D9D9D9"/><w:right w:val="single" w:sz="4" w:color="D9D9D9"/><w:insideH w:val="single" w:sz="4" w:color="D9D9D9"/><w:insideV w:val="single" w:sz="4" w:color="D9D9D9"/></w:tblBorders></w:tblPr><w:tr>'
  foreach($h in $headers){$xml+=Cell $h $true};$xml+='</w:tr>'
  foreach($row in $rows){$xml+='<w:tr>';foreach($v in $row){$xml+=Cell ([string]$v) $false};$xml+='</w:tr>'};$xml+='</w:tbl>';return $xml
}
function Clean([string]$s){
  $s=$s -replace '\\code\{([^}]*)\}','$1' -replace '\\textbf\{([^}]*)\}','$1' -replace '\\textit\{([^}]*)\}','$1'
  $s=$s -replace '\\cite\{ultralytics_yolo11\}','[1]' -replace '\\cite\{ren2015faster\}','[2]' -replace '\\cite\{he2016resnet\}','[3]' -replace '\\cite\{lin2017fpn\}','[4]' -replace '\\cite\{torchvision_fasterrcnn\}','[5]' -replace '\\cite\{rod_dataset\}','[6]' -replace '\\cite\{coco_eval\}','[7]'
  $s=$s -replace '\\ref\{[^}]+\}','' -replace '~',' ' -replace '\\&','&' -replace '\\%','%' -replace '--','–'
  $s=$s -replace '\$F_2\$','F₂' -replace '\$TP\$','TP' -replace '\$FP\$','FP' -replace '\$FN\$','FN'
  $s=$s -replace '\\pending','Chưa có kết quả thực nghiệm hoàn chỉnh.' -replace '\\[a-zA-Z]+\*?(\[[^]]*\])?(\{[^}]*\})?',''
  return ($s -replace '\s+',' ').Trim()
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
New-Item -ItemType Directory -Force -Path $temp|Out-Null
[System.IO.Compression.ZipFile]::ExtractToDirectory($template,$temp)
$tex=Get-Content -LiteralPath $source -Raw -Encoding UTF8
$body=$tex.Substring($tex.IndexOf('\begin{document}')+16);$body=$body.Substring(0,$body.LastIndexOf('\end{document}'))
$xmlBody=''
$title='Phát hiện vật cản trong ảnh hướng tới hỗ trợ di chuyển cho người khiếm thị';$xmlBody+=Para $title 'Title' 'center'
$xmlBody+=Para 'Nguyễn Trần Kiên' 'Subtitle' 'center';$xmlBody+=Para 'Trường Đại học Sư phạm Kỹ thuật Hưng Yên | Mã sinh viên 12423018' 'Normal' 'center'
$abstract=[regex]::Match($body,'(?s)\\begin\{abstract\}(.*?)\\end\{abstract\}').Groups[1].Value
$xmlBody+=Para 'Tóm tắt' 'Heading1' 'left';$xmlBody+=Para (Clean $abstract)
$keywords=[regex]::Match($body,'\\noindent\\textbf\{Từ khóa:\}\s*(.*)').Groups[1].Value;$xmlBody+=Para ('Từ khóa: '+(Clean $keywords)) 'Normal' 'left'
$body=[regex]::Replace($body,'(?s).*?\\section\{Giới thiệu\}',"`n@@H1:Giới thiệu@@`n")
$body=[regex]::Replace($body,'\\section\*?\{([^}]+)\}',"`n@@H1:`$1@@`n")
$body=[regex]::Replace($body,'\\subsection\{([^}]+)\}',"`n@@H2:`$1@@`n")
$body=[regex]::Replace($body,'(?s)\\begin\{table\*?\}.*?\\end\{table\*?\}',"`n[TABLE]`n")
$body=[regex]::Replace($body,'(?s)\\begin\{equation\}.*?\\end\{equation\}',"`nVới TP, FP và FN: P = TP/(TP + FP), R = TP/(TP + FN), F₂ = 5PR/(4P + R).`n")
$body=[regex]::Replace($body,'(?s)\\bibliographystyle\{.*','')
$blocks=$body -split "(`r?`n){2,}"
$tableIndex=0
foreach($block in $blocks){
  $b=$block.Trim();if(-not $b){continue}
  if($b -match '^@@H1:(.+)@@$'){$xmlBody+=Para (Clean $matches[1]) 'Heading1' 'left';continue}
  if($b -match '^@@H2:(.+)@@$'){$xmlBody+=Para (Clean $matches[1]) 'Heading2' 'left';continue}
  if($b -eq '[TABLE]'){
    $tableIndex++
    if($tableIndex -eq 1){$xmlBody+=Para 'Bảng 1  Quy mô ROD Final A4' 'Caption' 'center';$xmlBody+=TableXml @('Tập','Số ảnh','Số đối tượng') @(@('TRAIN','18.151','30.793'),@('VALID','3.318','5.320'),@('TEST','1.611','2.354'),@('Tổng','23.080','38.467'))}
    elseif($tableIndex -eq 2){$xmlBody+=Para 'Bảng 2  Kết quả trên VALID bằng evaluator chung' 'Caption' 'center';$xmlBody+=TableXml @('Mô hình','Threshold','Precision','Recall','mAP50','mAP50–95') @(@('E3 YOLO11n','0,22','0,6455','0,7733','0,7958','0,5951'),@('E4 YOLO11s','0,19','0,6502','0,8075','0,8049','0,6145'),@('E6 Faster R-CNN','0,59','0,7483','0,8453','0,8701','0,6604'))}
    elseif($tableIndex -eq 3){$xmlBody+=Para 'Bảng 3  Chi phí suy luận trong phép đo CPU' 'Caption' 'center';$xmlBody+=TableXml @('Mô hình','Tham số (M)','ms/ảnh','FPS') @(@('E3','2,59','76,26','13,11'),@('E4','9,44','185,83','5,38'),@('E6','43,38','2.511,30','0,40'))}
    elseif($tableIndex -eq 4){$xmlBody+=Para 'Bảng 4  Kết quả cuối trên TEST bằng evaluator chung đã khóa' 'Caption' 'center';$xmlBody+=TableXml @('Mô hình','Precision','Recall','mAP50','mAP50–95','TP','FP','FN') @(@('E3 YOLO11n','0,6551','0,7392','0,7518','0,5548','1.740','916','614'),@('E4 YOLO11s','0,6662','0,7732','0,7774','0,5807','1.820','912','534'),@('E6 Faster R-CNN','0,7570','0,8496','0,8754','0,6782','2.000','642','354'))}
    else{
      $e3=Import-Csv (Join-Path $project 'results\E3_YOLO11n\metrics\common_test_per_class.csv');$e4=Import-Csv (Join-Path $project 'results\E4_YOLO11s\metrics\common_test_per_class.csv');$e6=Import-Csv (Join-Path $project 'results\E6_Faster_RCNN\metrics\common_test_per_class.csv');$rows=@()
      for($j=0;$j -lt 25;$j++){$rows+=,@($e3[$j].class_name,$e3[$j].gt_count,([double]$e3[$j].ap50_95).ToString('0.0000'),([double]$e4[$j].ap50_95).ToString('0.0000'),([double]$e6[$j].ap50_95).ToString('0.0000'))}
      $xmlBody+=Para 'Bảng 5  AP50–95 theo 25 lớp trên TEST' 'Caption' 'center';$xmlBody+=TableXml @('Lớp','GT','E3','E4','E6') $rows
    };continue
  }
  $clean=Clean $b;if($clean){$xmlBody+=Para $clean}
}
$refs=@('[1] Ultralytics, YOLO11 Model Documentation.','[2] S. Ren, K. He, R. Girshick và J. Sun, Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, NeurIPS, 2015.','[3] K. He, X. Zhang, S. Ren và J. Sun, Deep Residual Learning for Image Recognition, CVPR, 2016.','[4] T.-Y. Lin và cộng sự, Feature Pyramid Networks for Object Detection, CVPR, 2017.','[5] Torchvision, Faster R-CNN ResNet-50 FPN V2 Documentation.','[6] Abtinzandi, ROD-Dataset, dataset card.','[7] COCO Consortium, Object Detection Evaluation.')
$xmlBody+=Para 'Tài liệu tham khảo' 'Heading1' 'left';foreach($r in $refs){$xmlBody+=Para $r 'Normal' 'left'}
$sect='<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="1247" w:bottom="1134" w:left="1247" w:header="720" w:footer="720" w:gutter="0"/><w:cols w:space="708"/><w:docGrid w:linePitch="360"/></w:sectPr>'
$document='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'+$xmlBody+$sect+'</w:body></w:document>'
[System.IO.File]::WriteAllText((Join-Path $temp 'word\document.xml'),$document,(New-Object System.Text.UTF8Encoding($false)))
if(Test-Path -LiteralPath $output){Remove-Item -LiteralPath $output -Force}
$stream=[System.IO.File]::Open($output,[System.IO.FileMode]::Create)
$archive=New-Object System.IO.Compression.ZipArchive($stream,[System.IO.Compression.ZipArchiveMode]::Create,$false)
Get-ChildItem -LiteralPath $temp -Recurse -File|ForEach-Object{
  $name=$_.FullName.Substring($temp.Length+1).Replace('\','/')
  $entry=$archive.CreateEntry($name,[System.IO.Compression.CompressionLevel]::Optimal)
  $input=[System.IO.File]::OpenRead($_.FullName);$outputStream=$entry.Open();$input.CopyTo($outputStream);$outputStream.Dispose();$input.Dispose()
}
$archive.Dispose();$stream.Dispose()
Remove-Item -LiteralPath $temp -Recurse -Force
$zip=[System.IO.Compression.ZipFile]::OpenRead($output);$count=$zip.Entries.Count;$docEntry=$zip.GetEntry('word/document.xml');$zip.Dispose()
if($null -eq $docEntry -or $count -lt 5){throw 'DOCX package validation failed'}
Get-Item -LiteralPath $output|Select-Object FullName,Length,LastWriteTime



