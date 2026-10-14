$ErrorActionPreference = 'Stop'

$project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$source = Join-Path $project 'docs\reports\BAO_CAO_PHAT_HIEN_VAT_CAN.tex'
$template = Join-Path $project 'docs\reports\12423TN_12423018_NguyenTranKien (2).docx'
$output = Join-Path $project 'docs\reports\BAO_CAO_PHAT_HIEN_VAT_CAN.docx'
$temp = Join-Path $env:TEMP ('rod_report_docx_' + [guid]::NewGuid().ToString('N'))

function XmlEscape([string]$value) { [Security.SecurityElement]::Escape($value) }

function Clean-Tex([string]$value) {
    if ($null -eq $value) { return '' }
    $value = $value -replace '(?m)^\s*%.*$', ''
    $value = $value -replace '\\_', '_' -replace '\\&', '&' -replace '\\%', '%' -replace '\\#', '#'
    $value = $value -replace '\\textasciitilde', '~' -replace '~', ' '
    for ($i = 0; $i -lt 4; $i++) {
        $value = $value -replace '\\(textbf|textit|emph|texttt|url|underline)\{([^{}]*)\}', '$2'
    }
    $value = $value -replace '\\cite\{([^}]*)\}', ' [$1]'
    $value = $value -replace '\\(label|index)\{[^}]*\}', ''
    $value = $value -replace '\\ref\{([^}]*)\}', '$1'
    $value = $value -replace '\\setcounter\{[^}]*\}\{[^}]*\}', '' -replace '\\pagenumbering\{[^}]*\}', ''
    $value = $value -replace '\\(measured|configured|literature)\{([^}]*)\}', ''
    $value = $value -replace '\\(hypothesis|missing)', ''
    $value = $value -replace '\\(toprule|midrule|bottomrule|endhead|centering|small|footnotesize|normalsize)', ''
    $value = $value -replace '\\(addcontentsline|phantomsection)\{[^}]*\}(\{[^}]*\})?(\{[^}]*\})?', ''
    $value = $value -replace '\\[a-zA-Z]+\*?(\[[^]]*\])?', ''
    $value = $value -replace '[{}$]', ''
    $value = $value -replace '\s+', ' '
    $value.Trim()
}

function Paragraph([string]$text, [string]$style = 'Normal', [string]$align = 'both', [bool]$pageBreak = $false) {
    if ([string]::IsNullOrWhiteSpace($text)) { return '' }
    # The reference report uses 13 pt Times New Roman with 1.5 line spacing.
    # Its chapter headings are 14 pt; lower headings remain 13 pt.
    $size = switch ($style) { 'Title' {'36'} 'Heading1' {'28'} 'Heading2' {'26'} 'Heading3' {'26'} 'Caption' {'22'} default {'26'} }
    $bold = if ($style -in @('Title','Heading1','Heading2','Heading3')) { '<w:b/>' } else { '' }
    $break = if ($pageBreak) { '<w:pageBreakBefore/>' } else { '' }
    $keep = if ($style -in @('Heading1','Heading2','Heading3','Caption')) { '<w:keepNext/><w:keepLines/>' } else { '<w:widowControl/>' }
    $indent = if ($style -eq 'Normal' -and $align -eq 'both') { '<w:ind w:firstLine="709"/>' } else { '' }
    '<w:p><w:pPr><w:pStyle w:val="' + $style + '"/><w:jc w:val="' + $align + '"/><w:spacing w:before="0" w:after="0" w:line="360" w:lineRule="auto"/>' + $indent + $keep + $break + '</w:pPr><w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="Times New Roman" w:cs="Times New Roman"/>' + $bold + '<w:color w:val="000000"/><w:sz w:val="' + $size + '"/><w:szCs w:val="' + $size + '"/></w:rPr><w:t xml:space="preserve">' + (XmlEscape $text) + '</w:t></w:r></w:p>'
}

function CoverParagraph([string]$text, [int]$size, [bool]$bold = $false, [int]$before = 0, [int]$after = 0) {
    if ([string]::IsNullOrWhiteSpace($text)) { return '' }
    $boldXml = if ($bold) { '<w:b/><w:bCs/>' } else { '' }
    '<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="' + $before + '" w:after="' + $after + '" w:line="300" w:lineRule="auto"/></w:pPr><w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="Times New Roman" w:cs="Times New Roman"/>' + $boldXml + '<w:color w:val="000000"/><w:sz w:val="' + $size + '"/><w:szCs w:val="' + $size + '"/></w:rPr><w:t xml:space="preserve">' + (XmlEscape $text) + '</w:t></w:r></w:p>'
}

function PageBreak { '<w:p><w:r><w:br w:type="page"/></w:r></w:p>' }

function TocField([string]$instruction) {
    '<w:p><w:pPr><w:spacing w:after="0" w:line="300" w:lineRule="auto"/></w:pPr><w:r><w:fldChar w:fldCharType="begin" w:dirty="true"/></w:r><w:r><w:instrText xml:space="preserve"> ' + (XmlEscape $instruction) + ' </w:instrText></w:r><w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="26"/><w:szCs w:val="26"/></w:rPr><w:t>Cập nhật trường trong Microsoft Word để tạo danh mục.</w:t></w:r><w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>'
}

function TableCell([string]$text, [bool]$header = $false) {
    $shade = if ($header) { '<w:shd w:val="clear" w:color="auto" w:fill="C6D9F1"/>' } else { '' }
    $color = '<w:color w:val="000000"/>'
    $bold = if ($header) { '<w:b/>' } else { '' }
    '<w:tc><w:tcPr><w:tcMar><w:top w:w="60" w:type="dxa"/><w:left w:w="60" w:type="dxa"/><w:bottom w:w="60" w:type="dxa"/><w:right w:w="60" w:type="dxa"/></w:tcMar>' + $shade + '</w:tcPr><w:p><w:pPr><w:jc w:val="center"/><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr><w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="Times New Roman"/>' + $bold + $color + '<w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr><w:t xml:space="preserve">' + (XmlEscape $text) + '</w:t></w:r></w:p></w:tc>'
}

function WordTable($rows) {
    if ($rows.Count -lt 1) { return '' }
    $max = ($rows | ForEach-Object { $_.Count } | Measure-Object -Maximum).Maximum
    $xml = '<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/><w:jc w:val="center"/><w:tblLayout w:type="autofit"/><w:tblBorders><w:top w:val="single" w:sz="4" w:space="0" w:color="000000"/><w:left w:val="single" w:sz="4" w:space="0" w:color="000000"/><w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/><w:right w:val="single" w:sz="4" w:space="0" w:color="000000"/><w:insideH w:val="single" w:sz="4" w:space="0" w:color="000000"/><w:insideV w:val="single" w:sz="4" w:space="0" w:color="000000"/></w:tblBorders></w:tblPr>'
    for ($r = 0; $r -lt $rows.Count; $r++) {
        $xml += '<w:tr>'
        for ($c = 0; $c -lt $max; $c++) {
            $cell = if ($c -lt $rows[$r].Count) { [string]$rows[$r][$c] } else { '' }
            $xml += TableCell $cell ($r -eq 0)
        }
        $xml += '</w:tr>'
    }
    $xml + '</w:tbl>'
}

function ImageBlock([string]$relationshipId, [string]$name, [long]$cx, [long]$cy, [int]$docId) {
    '<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:drawing><wp:inline xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" distT="0" distB="0" distL="0" distR="0"><wp:extent cx="' + $cx + '" cy="' + $cy + '"/><wp:docPr id="' + $docId + '" name="' + (XmlEscape $name) + '"/><a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr><pic:cNvPr id="0" name="' + (XmlEscape $name) + '"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" r:embed="' + $relationshipId + '"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="' + $cx + '" cy="' + $cy + '"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'
}

function Parse-Table([string]$block) {
    $caption = ''
    $capMatch = [regex]::Match($block, '\\caption\{([^}]*)\}')
    if ($capMatch.Success) { $caption = Clean-Tex $capMatch.Groups[1].Value }
    $tab = [regex]::Match($block, '(?s)\\begin\{(?:tabular|longtable)\}(?:\{[^}]*\})?(.*?)\\end\{(?:tabular|longtable)\}')
    if (-not $tab.Success -and $block -match '\\begin\{longtable\}') {
        $content = $block -replace '(?s)^.*?\\begin\{longtable\}\{[^}]*\}', '' -replace '(?s)\\end\{longtable\}.*$', ''
    } elseif ($tab.Success) { $content = $tab.Groups[1].Value } else { return @{ Caption=$caption; Rows=@() } }
    $content = $content -replace '\\caption\{[^}]*\}', '' -replace '\\label\{[^}]*\}', ''
    $content = $content -replace '\\(toprule|midrule|bottomrule|endfirsthead|endhead|endfoot|endlastfoot)', ''
    $rows = @()
    foreach ($row in [regex]::Split($content, '\\\\')) {
        $cleanRow = $row.Trim()
        if (-not $cleanRow -or $cleanRow -match '^\\') { continue }
        $cells = @($cleanRow -split '(?<!\\)&' | ForEach-Object { Clean-Tex $_ })
        if ($cells.Count -gt 1) { $rows += ,$cells }
    }
    @{ Caption=$caption; Rows=$rows }
}

if (-not (Test-Path -LiteralPath $source)) { throw "Missing source: $source" }
if (-not (Test-Path -LiteralPath $template)) { throw "Missing DOCX template: $template" }

Add-Type -AssemblyName System.IO.Compression.FileSystem
Add-Type -AssemblyName System.Drawing
New-Item -ItemType Directory -Force -Path $temp | Out-Null
[IO.Compression.ZipFile]::ExtractToDirectory($template, $temp)

$tex = Get-Content -LiteralPath $source -Raw -Encoding UTF8
$bodyStart = $tex.IndexOf('\end{titlepage}')
if ($bodyStart -lt 0) { throw 'Missing titlepage terminator.' }
$body = $tex.Substring($bodyStart + '\end{titlepage}'.Length)
$body = $body.Substring(0, $body.LastIndexOf('\end{document}'))
$gap = [Environment]::NewLine + [Environment]::NewLine

$tables = @{}
$tableCounter = 0
$tablePattern = '(?s)\\begin\{table\*?\}.*?\\end\{table\*?\}|\\begin\{longtable\}.*?\\end\{longtable\}'
$body = [regex]::Replace($body, $tablePattern, {
    param($match)
    $script:tableCounter++
    $script:tables[$script:tableCounter] = Parse-Table $match.Value
    ($gap + '@@TABLE:' + $script:tableCounter + '@@' + $gap)
})

$images = [ordered]@{}
$imageCounter = 0
$figureCounter = 0
$figurePattern = '(?s)\\begin\{figure\}.*?\\end\{figure\}'
$figureMatches = [regex]::Matches($body, $figurePattern)
for ($figureIndex = $figureMatches.Count - 1; $figureIndex -ge 0; $figureIndex--) {
    $match = $figureMatches[$figureIndex]
    $figureCounter = $figureIndex + 1
    $caption = ''
    $cm = [regex]::Match($match.Value, '\\caption\{([^}]*)\}')
    if ($cm.Success) { $caption = Clean-Tex $cm.Groups[1].Value }
    $im = [regex]::Match($match.Value, '\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}')
    $replacement = $gap + '@@CAPTION:Hình ' + $figureCounter + '. ' + $caption + '@@' + $gap
    if ($im.Success) {
        $raw = $im.Groups[1].Value
        $candidate = if ($raw -like 'figures/*') { Join-Path (Split-Path $source) $raw } elseif ($raw -like '../../results/*') { Join-Path $project ($raw.Substring(6)) } else { Join-Path (Split-Path $source) $raw }
        $candidate = [IO.Path]::GetFullPath($candidate)
        if (Test-Path -LiteralPath $candidate) {
            $imageCounter++
            $images[[string]$imageCounter] = $candidate
            $replacement = $gap + '@@IMAGE:' + $imageCounter + '@@' + $gap + '@@CAPTION:Hình ' + $figureCounter + '. ' + $caption + '@@' + $gap
        }
    }
    $body = $body.Remove($match.Index, $match.Length).Insert($match.Index, $replacement)
}

$body = [regex]::Replace($body, '\\chapter\*?\{([^}]+)\}', ($gap + '@@H1:$1@@' + $gap))
$body = [regex]::Replace($body, '\\section\*?\{([^}]+)\}', ($gap + '@@H2:$1@@' + $gap))
$body = [regex]::Replace($body, '\\subsection\*?\{([^}]+)\}', ($gap + '@@H3:$1@@' + $gap))
$body = [regex]::Replace($body, '\\paragraph\{([^}]+)\}', ($gap + '@@H4:$1@@ '))
$body = [regex]::Replace($body, '(?s)\\begin\{equation\}(.*?)\\end\{equation\}', { param($m) ($gap + '@@EQ:' + (Clean-Tex $m.Groups[1].Value) + '@@' + $gap) })
$body = $body -replace '\\begin\{(itemize|enumerate)\}', '' -replace '\\end\{(itemize|enumerate)\}', ''
$body = $body -replace '\\item', ($gap + '@@BULLET@@ ')
$body = $body -replace '\\clearpage|\\newpage|\\listoffigures|\\listoftables|\\tableofcontents|\\appendix', [Environment]::NewLine
$body = $body -replace '\\begin\{thebibliography\}\{[^}]*\}|\\end\{thebibliography\}', [Environment]::NewLine
$body = [regex]::Replace($body, '\\bibitem\{([^}]*)\}', ($gap + '@@BIB:$1@@ '))
$body = $body -replace '(?m)^\s*\\IfFileExists\{[^}]+\}\s*$', ''
$body = $body -replace '(?m)^\s*\{\\includegraphics[^\r\n]+$', ''
$body = $body -replace '(?m)^\s*\{\\fbox[^\r\n]+$', ''

$xmlBody = ''
$xmlBody += CoverParagraph 'BỘ GIÁO DỤC VÀ ĐÀO TẠO' 28 $true 0 0
$xmlBody += CoverParagraph 'TRƯỜNG ĐẠI HỌC SƯ PHẠM KỸ THUẬT HƯNG YÊN' 28 $true 0 80
$xmlBody += ImageBlock 'rId8' 'Logo trường' 1586046 1490015 50
$xmlBody += CoverParagraph 'BÀI TẬP LỚN' 36 $true 180 0
$xmlBody += CoverParagraph 'HỌC SÂU' 36 $true 0 220
$xmlBody += CoverParagraph 'PHÁT HIỆN VẬT CẢN TRONG ẢNH' 28 $true 0 0
$xmlBody += CoverParagraph 'HƯỚNG TỚI HỖ TRỢ DI CHUYỂN' 28 $true 0 0
$xmlBody += CoverParagraph 'CHO NGƯỜI KHIẾM THỊ' 28 $true 0 140
$xmlBody += CoverParagraph 'NGÀNH: KHOA HỌC MÁY TÍNH' 26 $false 0 0
$xmlBody += CoverParagraph 'CHUYÊN NGÀNH: TRÍ TUỆ NHÂN TẠO VÀ KHOA HỌC DỮ LIỆU' 26 $false 0 220
$xmlBody += CoverParagraph 'SINH VIÊN: NGUYỄN TRẦN KIÊN' 26 $false 0 0
$xmlBody += CoverParagraph 'MÃ SINH VIÊN: 12423018' 26 $false 0 0
$xmlBody += CoverParagraph 'MÃ LỚP: 12423TN' 26 $false 0 0
$xmlBody += CoverParagraph 'GIẢNG VIÊN HƯỚNG DẪN: PGS.TS. NGUYỄN MINH TIẾN' 26 $false 0 240
$xmlBody += CoverParagraph 'HƯNG YÊN - 2026' 28 $true 0 0
$xmlBody += PageBreak
$xmlBody += Paragraph 'MỤC LỤC' 'Heading1' 'center'
$xmlBody += TocField 'TOC \o "1-3" \h \z \u'

$blocks = [regex]::Split($body, '(\r?\n){2,}')
foreach ($rawBlock in $blocks) {
    $block = $rawBlock.Trim()
    if (-not $block) { continue }
    if ($block -match '^@@H1:(.+)@@$') { $xmlBody += Paragraph (Clean-Tex $matches[1]) 'Heading1' 'center' $true; continue }
    if ($block -match '^@@H2:(.+)@@$') { $xmlBody += Paragraph (Clean-Tex $matches[1]) 'Heading2'; continue }
    if ($block -match '^@@H3:(.+)@@$') { $xmlBody += Paragraph (Clean-Tex $matches[1]) 'Heading3'; continue }
    if ($block -match '^@@H4:(.+)@@\s*(.*)$') { $xmlBody += Paragraph ((Clean-Tex $matches[1]) + '. ' + (Clean-Tex $matches[2])); continue }
    if ($block -match '^@@TABLE:(\d+)@@$') {
        $item = $tables[[int]$matches[1]]
        if ($item.Caption) { $xmlBody += Paragraph ('Bảng: ' + $item.Caption) 'Caption' 'center' }
        $xmlBody += WordTable $item.Rows
        continue
    }
    if ($block -match '^@@IMAGE:(\d+)@@$') {
        $idx = [int]$matches[1]
        $file = $images[[string]$idx]
        $bitmap = [System.Drawing.Image]::FromFile($file)
        $maxCx = 5486400L
        $maxCy = 5943600L
        $ratio = [double]$bitmap.Width / [double]$bitmap.Height
        $cx = $maxCx; $cy = [long]($cx / $ratio)
        if ($cy -gt $maxCy) { $cy = $maxCy; $cx = [long]($cy * $ratio) }
        $bitmap.Dispose()
        $xmlBody += ImageBlock ('rIdImg' + $idx) ([IO.Path]::GetFileName($file)) $cx $cy (1000 + $idx)
        continue
    }
    if ($block -match '^@@CAPTION:(.+)@@$') { $xmlBody += Paragraph (Clean-Tex $matches[1]) 'Caption' 'center'; continue }
    if ($block -match '^@@EQ:(.+)@@$') { $xmlBody += Paragraph (Clean-Tex $matches[1]) 'Normal' 'center'; continue }
    if ($block -match '^@@BIB:([^@]+)@@\s*(.*)$') { $xmlBody += Paragraph ('[' + $matches[1] + '] ' + (Clean-Tex $matches[2])); continue }
    if ($block.StartsWith('@@BULLET@@')) { $xmlBody += Paragraph ('• ' + (Clean-Tex $block.Substring(10))); continue }
    $clean = Clean-Tex $block
    if ($clean) { $xmlBody += Paragraph $clean }
}

$section = '<w:sectPr><w:headerReference w:type="default" r:id="rId22" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/><w:footerReference w:type="default" r:id="rId23" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="1420" w:bottom="346" w:left="1701" w:header="454" w:footer="794" w:gutter="0"/><w:pgBorders w:offsetFrom="page" w:display="firstPage"><w:top w:val="single" w:sz="18" w:space="26" w:color="000000"/><w:left w:val="single" w:sz="18" w:space="26" w:color="000000"/><w:bottom w:val="single" w:sz="18" w:space="26" w:color="000000"/><w:right w:val="single" w:sz="18" w:space="26" w:color="000000"/></w:pgBorders><w:pgNumType w:start="1"/><w:cols w:space="720"/><w:titlePg/><w:docGrid w:linePitch="326"/></w:sectPr>'
$document = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>' + $xmlBody + $section + '</w:body></w:document>'
[IO.File]::WriteAllText((Join-Path $temp 'word\document.xml'), $document, (New-Object Text.UTF8Encoding($false)))

$relsPath = Join-Path $temp 'word\_rels\document.xml.rels'
$rels = Get-Content -LiteralPath $relsPath -Raw
foreach ($entry in $images.GetEnumerator()) {
    $idx = [int]$entry.Key
    $extension = [IO.Path]::GetExtension($entry.Value).ToLowerInvariant()
    $targetName = 'report_image_' + $idx + $extension
    Copy-Item -LiteralPath $entry.Value -Destination (Join-Path $temp ('word\media\' + $targetName)) -Force
    $relationship = '<Relationship Id="rIdImg' + $idx + '" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/' + $targetName + '"/>'
    $rels = $rels.Replace('</Relationships>', $relationship + '</Relationships>')
}
[IO.File]::WriteAllText($relsPath, $rels, (New-Object Text.UTF8Encoding($false)))

# Keep the template header styling but replace its unrelated historic title.
$headerPath = Join-Path $temp 'word\header1.xml'
if (Test-Path -LiteralPath $headerPath) {
    [xml]$headerXml = Get-Content -LiteralPath $headerPath -Raw
    $headerNs = New-Object Xml.XmlNamespaceManager($headerXml.NameTable)
    $headerNs.AddNamespace('w','http://schemas.openxmlformats.org/wordprocessingml/2006/main')
    $headerTextNodes = $headerXml.SelectNodes('//w:t', $headerNs)
    if ($headerTextNodes.Count -gt 0) {
        $headerTextNodes[0].InnerText = 'Phát hiện vật cản trong ảnh hướng tới hỗ trợ di chuyển cho người khiếm thị'
        for ($i = 1; $i -lt $headerTextNodes.Count; $i++) { $headerTextNodes[$i].InnerText = '' }
    }
    $headerXml.Save($headerPath)
}

$settingsPath = Join-Path $temp 'word\settings.xml'
if (Test-Path -LiteralPath $settingsPath) {
    $settings = Get-Content -LiteralPath $settingsPath -Raw
    if ($settings -notmatch 'updateFields') { $settings = $settings.Replace('</w:settings>', '<w:updateFields w:val="true"/></w:settings>') }
    [IO.File]::WriteAllText($settingsPath, $settings, (New-Object Text.UTF8Encoding($false)))
}

if (Test-Path -LiteralPath $output) { Remove-Item -LiteralPath $output -Force }
$stream = [IO.File]::Open($output, [IO.FileMode]::Create)
$archive = New-Object IO.Compression.ZipArchive($stream, [IO.Compression.ZipArchiveMode]::Create, $false)
Get-ChildItem -LiteralPath $temp -Recurse -File | ForEach-Object {
    $name = $_.FullName.Substring($temp.Length + 1).Replace('\','/')
    $zipEntry = $archive.CreateEntry($name, [IO.Compression.CompressionLevel]::Optimal)
    $input = [IO.File]::OpenRead($_.FullName)
    $target = $zipEntry.Open()
    $input.CopyTo($target)
    $target.Dispose(); $input.Dispose()
}
$archive.Dispose(); $stream.Dispose()
Remove-Item -LiteralPath $temp -Recurse -Force

Get-Item -LiteralPath $output | Select-Object FullName, Length, LastWriteTime
