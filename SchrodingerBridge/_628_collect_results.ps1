# Collect all completed 628 destructive ablation results into a summary
$ErrorActionPreference = 'Continue'
$root = 'I:/Github/Latent_Style/SchrodingerBridge'
$expDir = "$root\exp\628_ablation\destructive"

$results = @()
$dirs = Get-ChildItem $expDir -Directory -ErrorAction SilentlyContinue | Sort-Object Name
foreach ($d in $dirs) {
    $name = $d.Name
    # Try ep10 first, then ep9, ep8
    foreach ($ep in @('epoch_0010','epoch_0009','epoch_0008')) {
        $sumPath = Join-Path $d.FullName "full_eval\$ep\summary.json"
        if (Test-Path $sumPath) {
            try {
                $j = Get-Content $sumPath -Raw | ConvertFrom-Json
                $ap = $j.analysis.all_pairs_overview
                $tr = $j.analysis.style_transfer_ability
                $results += [PSCustomObject]@{
                    name = $name
                    epoch = $ep
                    ap_clip = [math]::Round($ap.clip_style, 4)
                    ap_lpips = [math]::Round($ap.content_lpips, 4)
                    tr_clip = [math]::Round($tr.clip_style, 4)
                    tr_lpips = [math]::Round($tr.content_lpips, 4)
                }
            } catch {
                $results += [PSCustomObject]@{ name = $name; epoch = $ep; ap_clip = 'ERR'; ap_lpips = 'ERR'; tr_clip = 'ERR'; tr_lpips = 'ERR' }
            }
            break
        }
    }
}

Write-Host "=== Completed experiments: $($results.Count) ==="
Write-Host ""
$results | Format-Table -AutoSize | Out-String -Width 200 | Write-Host

# Also save to CSV for easy analysis
$csvPath = "$root\exp\628_ablation\destructive_logs\results_summary.csv"
$results | Export-Csv -Path $csvPath -NoTypeInformation -Encoding UTF8
Write-Host "`nSaved to: $csvPath"

# Baseline reference
Write-Host "`n=== Baseline (T5 ep7) ==="
Write-Host "ap_clip=0.7307 ap_lpips=0.3403"

# Highlight significant deviations
Write-Host "`n=== Significant deviations (|dclip|>0.005 or |dlpips|>0.01) ==="
$baselineClip = 0.7307
$baselineLpips = 0.3403
foreach ($r in $results) {
    if ($r.ap_clip -is [double] -and $r.ap_lpips -is [double]) {
        $dclip = $r.ap_clip - $baselineClip
        $dlpips = $r.ap_lpips - $baselineLpips
        if ([math]::Abs($dclip) -gt 0.005 -or [math]::Abs($dlpips) -gt 0.01) {
            Write-Host ("  {0,-35} clip={1} (d={2:+0.0000;-0.0000;0}) lpips={3} (d={4:+0.0000;-0.0000;0})" -f $r.name, $r.ap_clip, $dclip, $r.ap_lpips, $dlpips)
        }
    }
}
