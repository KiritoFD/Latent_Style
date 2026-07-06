# Aggregate metrics.csv for wikiarts-15 baselines and WD-VF
# CSV header: src_style,tgt_style,src_image,gen_image,content_lpips,clip_dir,clip_style,clip_s_delta_idt,clip_t,clip_content,clip_image_vector
# CLIP-S = clip_style (col 6, 0-indexed)
# LPIPS  = content_lpips (col 4, 0-indexed)
$REPO = "I:\Github\Latent_Style\SchrodingerBridge"

function Agg-Metrics($path) {
    if (-not (Test-Path $path)) { return "MISSING" }
    # Use Import-Csv with explicit header (skip actual header row by reading all lines)
    $lines = Get-Content $path
    if ($lines.Count -lt 2) { return "EMPTY" }
    # Skip header (first line), parse data lines
    $n = 0
    $clipSum = 0.0
    $lpipsSum = 0.0
    $srcStyles = @{}
    $tgtStyles = @{}
    for ($i = 1; $i -lt $lines.Count; $i++) {
        $fields = $lines[$i] -split ','
        if ($fields.Count -lt 9) { continue }
        $lpips = 0.0
        $clip = 0.0
        if (-not [double]::TryParse($fields[4], [ref]$lpips)) { continue }
        if (-not [double]::TryParse($fields[6], [ref]$clip)) { continue }
        $lpipsSum += $lpips
        $clipSum += $clip
        $srcStyles[$fields[0]] = 1
        $tgtStyles[$fields[1]] = 1
        $n++
    }
    if ($n -eq 0) { return "NO_VALID_ROWS" }
    $clipAvg = $clipSum / $n
    $lpipsAvg = $lpipsSum / $n
    return "n=$n CLIP-S=$($clipAvg.ToString('F4')) LPIPS=$($lpipsAvg.ToString('F4')) src_styles=$($srcStyles.Count) tgt_styles=$($tgtStyles.Count)"
}

Write-Output "=== Baselines ==="
foreach ($m in @("identity","adain","wct")) {
    $p = "$REPO\exp\baseline_wikiarts15\$m\metrics.csv"
    Write-Output "${m}: $(Agg-Metrics $p)"
}

Write-Output ""
Write-Output "=== WD-VF ==="
$p = "$REPO\exp\wikiarts15_eval\metrics.csv"
Write-Output "WD-VF: $(Agg-Metrics $p)"
