param(
    [string]$Root
)

$ErrorActionPreference = "Stop"

if (-not $Root) {
    throw "Missing -Root"
}

$rows = @()
Get-ChildItem $Root -Directory | ForEach-Object {
    $summary = Join-Path $_.FullName "summary.json"
    if (Test-Path $summary) {
        $json = Get-Content $summary -Raw | ConvertFrom-Json
        $ov = $json.analysis.all_pairs_overview
        $rows += [PSCustomObject]@{
            name = $_.Name
            clip_style = [double]$ov.clip_style
            clip_content = [double]$ov.clip_content
            content_lpips = [double]$ov.content_lpips
            clip_dir = [double]$ov.clip_dir
        }
    }
}

$rows | Sort-Object clip_style -Descending | ConvertTo-Json -Depth 3
