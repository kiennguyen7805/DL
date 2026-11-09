# Deletes only an explicit list of generated temporary artifacts within this workspace.
# This does not delete source data, source code, environments, checkpoints or reports.
$ErrorActionPreference = 'Stop'
$workspaceRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path.TrimEnd('\')
$prefix = $workspaceRoot + '\'
$targets = [System.Collections.Generic.List[string]]::new()

foreach ($item in (Get-ChildItem -LiteralPath (Join-Path $workspaceRoot 'tmp') -Directory)) {
    if ($item.Name.StartsWith('pytest_') -or $item.Name.StartsWith('demo_')) {
        $targets.Add($item.FullName)
    }
}
foreach ($relative in @('tmp/e6_archive_audit', 'tmp/render_ppt', 'tmp/render_report',
                        'tmp/python-3.12.6-embed-amd64.zip', 'tmp/__pycache__',
                        'app/outputs', 'app/__pycache__', 'src/__pycache__',
                        'scripts/__pycache__', 'tests/__pycache__',
                        'output/handoff_E6_30epochs_web', '.codex_exec_probe')) {
    $candidate = Join-Path $workspaceRoot $relative
    if (Test-Path -LiteralPath $candidate) { $targets.Add($candidate) }
}

# Validate ALL resolved targets before the first removal. Reject reparse points
# in each target tree so recursive removal cannot traverse an unexpected link.
$verified = [System.Collections.Generic.List[object]]::new()
foreach ($target in $targets) {
    $resolved = (Resolve-Path -LiteralPath $target).Path
    if (-not $resolved.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing path outside workspace: $resolved"
    }
    $item = Get-Item -LiteralPath $resolved -Force
    if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
        throw "Refusing reparse point: $resolved"
    }
    $bytes = 0L
    $count = 0L
    if ($item.PSIsContainer) {
        foreach ($child in (Get-ChildItem -LiteralPath $resolved -Recurse -Force)) {
            if ($child.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
                throw "Refusing linked descendant: $($child.FullName)"
            }
            if (-not $child.PSIsContainer) { $bytes += $child.Length; $count++ }
        }
    } else { $bytes = $item.Length; $count = 1 }
    $verified.Add([PSCustomObject]@{
        path = $resolved.Substring($prefix.Length).Replace('\','/')
        absolute_path = $resolved
        bytes = $bytes
        file_count = $count
    })
}

$receiptPath = Join-Path $workspaceRoot 'output/CLEANUP_RECEIPT_20261028.json'
$totalBytes = 0L
foreach ($entry in $verified) { $totalBytes += $entry.bytes }
$receipt = [ordered]@{
    scope = $workspaceRoot
    status = 'validated_before_removal'
    logical_bytes_selected = $totalBytes
    items = $verified
}
$receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $receiptPath -Encoding UTF8

foreach ($entry in $verified) {
    Remove-Item -LiteralPath $entry.absolute_path -Recurse -Force
    if (Test-Path -LiteralPath $entry.absolute_path) { throw "Deletion incomplete: $($entry.absolute_path)" }
}
$receipt.status = 'complete'
$receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $receiptPath -Encoding UTF8
Write-Output ("Deleted {0} temporary items; logical total {1:N2} MiB. Receipt: {2}" -f $verified.Count, ($totalBytes / 1MB), $receiptPath)
