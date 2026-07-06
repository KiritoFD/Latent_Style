# Aggregate metrics.csv for a method (correct column indices)
# CSV columns: src_style,tgt_style,src_image,gen_image,content_lpips,clip_dir,clip_style,clip_s_delta_idt,clip_t,clip_content,clip_image_vector
# LPIPS = col 4 (content_lpips), CLIP-S = col 6 (clip_style)

param(
    [string]$Method = "identity"
)

$csv = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20\$Method\metrics.csv"
if (-not (Test-Path $csv)) {
    "$Method csv NOT FOUND"
    exit 1
}

$fi = Get-Item $csv
"$Method csv size=$($fi.Length) mtime=$($fi.LastWriteTime)"

$lines = Get-Content $csv
"total lines: $($lines.Count)"
"header: $($lines[0])"

$clip_vals = @()
$lpips_vals = @()
$bad = 0
for ($i = 1; $i -lt $lines.Count; $i++) {
    $cols = $lines[$i].Split(',')
    if ($cols.Count -ge 7) {
        $lpipsStr = $cols[4].Trim()
        $clipStr = $cols[6].Trim()
        $lp = 0.0
        $cl = 0.0
        if ([double]::TryParse($lpipsStr, [ref]$lp) -and [double]::TryParse($clipStr, [ref]$cl)) {
            $lpips_vals += $lp
            $clip_vals += $cl
        } else {
            $bad++
        }
    }
}

if ($clip_vals.Count -gt 0) {
    $avg_clip = ($clip_vals | Measure-Object -Average).Average
    $avg_lpips = ($lpips_vals | Measure-Object -Average).Average
    "$Method 20-style SUMMARY: n=$($clip_vals.Count) CLIP-S=$avg_clip LPIPS=$avg_lpips (bad=$bad)"
} else {
    "$Method no valid data (bad=$bad)"
}
