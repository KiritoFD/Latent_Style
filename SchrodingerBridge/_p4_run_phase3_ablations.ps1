# Phase 3 fine-tuning: V/U/scale grid on T5 baseline (best Pareto)
# T5 checkpoint: t5_b2v2_d2_d4/epoch_XXXX.pt
# T5 config: p4_t5_b2v2_d2_d4.json
# Best Pareto baseline: T5_D4_u01_v3 (ep7, U=0.1, V=k16) -> clip=0.7323, lpips=0.3534

# Each inference ablation takes ~80-100s. Total ~20 ablations = ~30 min.

$base = "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONPATH = ""
$env:P4_BASELINE_CLIP = "0.7323"
$env:P4_BASELINE_LPIPS = "0.3534"

# Default ep7 (best Pareto baseline)
$env:P4_CKPT_PATH = "$base\exp\p4_fusion_breakout\t5_b2v2_d2_d4\epoch_0007.pt"
$env:P4_CONFIG_PATH = "$base\configs\p4_t5_b2v2_d2_d4.json"

# Inference ablation CLI args:
# <exp_name> <lowpass_mode> [style_extrap_alpha] [patch_adain_kernel] [multiband_adain_mode] [tri_band_inference_lock] [mid_adain_scale] [hh_adain_scale]

$ablations = @(
    # === Task 3.2: V direction k grid (k=8, 32, 48), U=0.1, ep7 ===
    @{name="P3_V_k08"; ep="0007"; args=@("P3_V_k08", "dwt_haar", "0.1", "8", "single", "0", "0.3", "0.3")},
    @{name="P3_V_k32"; ep="0007"; args=@("P3_V_k32", "dwt_haar", "0.1", "32", "single", "0", "0.3", "0.3")},
    @{name="P3_V_k48"; ep="0007"; args=@("P3_V_k48", "dwt_haar", "0.1", "48", "single", "0", "0.3", "0.3")},
    # === Task 3.1: U direction alpha grid (gaps: 0.15, 0.3, 0.2), V=k16, ep7 ===
    @{name="P3_U_a015"; ep="0007"; args=@("P3_U_a015", "dwt_haar", "0.15", "16", "single", "0", "0.3", "0.3")},
    @{name="P3_U_a020"; ep="0007"; args=@("P3_U_a020", "dwt_haar", "0.2", "16", "single", "0", "0.3", "0.3")},
    @{name="P3_U_a030"; ep="0007"; args=@("P3_U_a030", "dwt_haar", "0.3", "16", "single", "0", "0.3", "0.3")},
    # === Task 3.5: Early stopping - U4+V3 on different epochs (ep1, ep4, ep8) ===
    # ep1 baseline (no U/V): clip=0.7226, lpips=0.3278 (LPIPS already < 0.35)
    @{name="P3_ep1_u01_v3"; ep="0001"; args=@("P3_ep1_u01_v3", "dwt_haar", "0.1", "16", "single", "0", "0.3", "0.3")},
    @{name="P3_ep4_u01_v3"; ep="0004"; args=@("P3_ep4_u01_v3", "dwt_haar", "0.1", "16", "single", "0", "0.3", "0.3")},
    @{name="P3_ep8_u01_v3"; ep="0008"; args=@("P3_ep8_u01_v3", "dwt_haar", "0.1", "16", "single", "0", "0.3", "0.3")},
    # === Combined: V=8 with U variations ===
    @{name="P3_V08_U005"; ep="0007"; args=@("P3_V08_U005", "dwt_haar", "0.05", "8", "single", "0", "0.3", "0.3")},
    @{name="P3_V08_U015"; ep="0007"; args=@("P3_V08_U015", "dwt_haar", "0.15", "8", "single", "0", "0.3", "0.3")},
    # === mid/hh adain scale fine-tuning ===
    @{name="P3_mid05_hh05"; ep="0007"; args=@("P3_mid05_hh05", "dwt_haar", "0.1", "16", "single", "0", "0.5", "0.5")},
    @{name="P3_mid01_hh01"; ep="0007"; args=@("P3_mid01_hh01", "dwt_haar", "0.1", "16", "single", "0", "0.1", "0.1")},
    @{name="P3_mid05_hh01"; ep="0007"; args=@("P3_mid05_hh01", "dwt_haar", "0.1", "16", "single", "0", "0.5", "0.1")}
)

Write-Host "=== Phase 3 Fine-tuning Ablations ==="
Write-Host "Total ablations: $($ablations.Count)"
Write-Host "Baseline: T5_D4_u01_v3 ep7 (clip=0.7323, lpips=0.3534)"
Write-Host "Target: clip > 0.74, lpips < 0.35"
Write-Host ""

# Track results
$results = @()
$startTime = Get-Date

for ($i = 0; $i -lt $ablations.Count; $i++) {
    $abl = $ablations[$i]
    $name = $abl.name
    $ep = $abl.ep
    $args = $abl.args

    $elapsed = ((Get-Date) - $startTime).TotalMinutes
    Write-Host "[$($i+1)/$($ablations.Count)] Running $name (ep=$ep, elapsed: $([math]::Round($elapsed, 1)) min)..."

    # Update checkpoint path per epoch
    $env:P4_CKPT_PATH = "$base\exp\p4_fusion_breakout\t5_b2v2_d2_d4\epoch_$ep.pt"

    # Run inference ablation (sequential, not parallel, to avoid OOM)
    & "C:\Program Files\Python312\python.exe" "$base\_p4_infer_ablation.py" $args 2>&1 | Out-Null
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        Write-Host "  -> FAILED (exit code $exitCode)"
        continue
    }

    # Read result
    $resultFile = "$base\exp\p4_fusion_breakout\infer_ablation\$name.json"
    if (Test-Path $resultFile) {
        try {
            $data = Get-Content $resultFile -Raw | ConvertFrom-Json
            $metrics = $data.metrics
            $clip = [math]::Round($metrics.allpairs_clip_style, 4)
            $lpips = [math]::Round($metrics.allpairs_content_lpips, 4)
            $trClip = [math]::Round($metrics.transfer_clip_style, 4)
            $trLpips = [math]::Round($metrics.transfer_content_lpips, 4)
            $dist = [math]::Round(0.74 - $clip, 4)
            Write-Host "  -> clip=$clip lpips=$lpips transfer_clip=$trClip transfer_lpips=$trLpips dist_to_0.74=$dist"

            $results += [PSCustomObject]@{
                Name = $name
                Epoch = $ep
                ClipStyle = $clip
                Lpips = $lpips
                TransferClip = $trClip
                TransferLpips = $trLpips
                DistTo074 = $dist
            }
        } catch {
            Write-Host "  -> FAILED to parse result: $_"
        }
    } else {
        Write-Host "  -> FAILED: result file not found"
    }
}

Write-Host ""
Write-Host "=== Phase 3 Results Summary (sorted by clip_style desc) ==="
$results | Sort-Object ClipStyle -Descending | Format-Table -AutoSize

Write-Host ""
Write-Host "=== Phase 3 Results Summary (sorted by lpips asc) ==="
$results | Sort-Object Lpips | Format-Table -AutoSize

Write-Host ""
Write-Host "=== Phase 3 vs Baseline ==="
Write-Host "Baseline T5_D4_u01_v3 ep7: clip=0.7323, lpips=0.3534"
Write-Host "Target: clip > 0.74, lpips < 0.35"
Write-Host ""
$bestClip = ($results | Sort-Object ClipStyle -Descending | Select-Object -First 1)
$bestLpips = ($results | Sort-Object Lpips | Select-Object -First 1)
Write-Host "Best clip: $($bestClip.Name) = $($bestClip.ClipStyle) / $($bestClip.Lpips)"
Write-Host "Best lpips: $($bestLpips.Name) = $($bestLpips.ClipStyle) / $($bestLpips.Lpips)"

# Find Pareto-optimal points (include baseline)
Write-Host ""
Write-Host "=== Pareto Frontier (including baseline) ==="
$allPoints = @()
$allPoints += [PSCustomObject]@{Name="T5_D4_u01_v3_baseline"; ClipStyle=0.7323; Lpips=0.3534; Epoch="0007"}
foreach ($r in $results) {
    $allPoints += [PSCustomObject]@{Name=$r.Name; ClipStyle=$r.ClipStyle; Lpips=$r.Lpips; Epoch=$r.Epoch}
}

$pareto = @()
foreach ($p in $allPoints) {
    $dominated = $false
    foreach ($p2 in $allPoints) {
        if ($p.Name -ne $p2.Name) {
            if ($p2.ClipStyle -ge $p.ClipStyle -and $p2.Lpips -le $p.Lpips -and
                ($p2.ClipStyle -gt $p.ClipStyle -or $p2.Lpips -lt $p.Lpips)) {
                $dominated = $true
                break
            }
        }
    }
    if (-not $dominated) {
        $pareto += $p
    }
}
$pareto | Sort-Object ClipStyle -Descending | Format-Table -AutoSize

Write-Host ""
Write-Host "=== Total elapsed: $([math]::Round(((Get-Date) - $startTime).TotalMinutes, 1)) min ==="

# Save summary to file
$summaryFile = "$base\exp\p4_fusion_breakout\infer_ablation\_phase3_summary.txt"
"=== Phase 3 Fine-tuning Summary ===" | Out-File $summaryFile
"Generated: $(Get-Date)" | Out-File $summaryFile -Append
"Baseline: T5_D4_u01_v3 ep7 (clip=0.7323, lpips=0.3534)" | Out-File $summaryFile -Append
"Target: clip > 0.74, lpips < 0.35" | Out-File $summaryFile -Append
"" | Out-File $summaryFile -Append
"=== All Results (sorted by clip desc) ===" | Out-File $summaryFile -Append
$results | Sort-Object ClipStyle -Descending | Format-Table -AutoSize | Out-String | Out-File $summaryFile -Append
"" | Out-File $summaryFile -Append
"=== Pareto Frontier ===" | Out-File $summaryFile -Append
$pareto | Sort-Object ClipStyle -Descending | Format-Table -AutoSize | Out-String | Out-File $summaryFile -Append
Write-Host "Summary saved to: $summaryFile"
