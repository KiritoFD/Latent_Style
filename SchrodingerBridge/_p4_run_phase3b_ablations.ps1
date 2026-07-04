# Phase 3b: 突破 0.74 的最后一轮精调 - 基于 P3_V_k08 (clip=0.7348)
# 策略：V=8 在 ep7 上能到 0.7348, 但 LPIPS 太高 (0.3868)
# 尝试：1) V=8 + ep1/ep4/ep8 (控制 LPIPS)  2) V=8 + 更小 alpha  3) V=8 + mid/hh scale 控制 LPIPS
# 目标：clip > 0.74 或 LPIPS < 0.35

$base = "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONPATH = ""
$env:P4_BASELINE_CLIP = "0.7348"
$env:P4_BASELINE_LPIPS = "0.3868"
$env:P4_CONFIG_PATH = "$base\configs\p4_t5_b2v2_d2_d4.json"

$ablations = @(
    # V=8 + 不同 epoch (LPIPS 控制)
    @{name="P3b_V08_ep1"; ep="0001"; args=@("P3b_V08_ep1", "dwt_haar", "0.1", "8", "single", "0", "0.3", "0.3")},
    @{name="P3b_V08_ep4"; ep="0004"; args=@("P3b_V08_ep4", "dwt_haar", "0.1", "8", "single", "0", "0.3", "0.3")},
    @{name="P3b_V08_ep8"; ep="0008"; args=@("P3b_V08_ep8", "dwt_haar", "0.1", "8", "single", "0", "0.3", "0.3")},
    @{name="P3b_V08_ep10"; ep="0010"; args=@("P3b_V08_ep10", "dwt_haar", "0.1", "8", "single", "0", "0.3", "0.3")},
    # V=8 + 更小 alpha (减小 LPIPS)
    @{name="P3b_V08_a005"; ep="0007"; args=@("P3b_V08_a005", "dwt_haar", "0.05", "8", "single", "0", "0.3", "0.3")},
    @{name="P3b_V08_a002"; ep="0007"; args=@("P3b_V08_a002", "dwt_haar", "0.02", "8", "single", "0", "0.3", "0.3")},
    # V=8 + mid/hh scale 调节 (压低 LPIPS)
    @{name="P3b_V08_mid01"; ep="0007"; args=@("P3b_V08_mid01", "dwt_haar", "0.1", "8", "single", "0", "0.1", "0.1")},
    @{name="P3b_V08_mid02"; ep="0007"; args=@("P3b_V08_mid02", "dwt_haar", "0.1", "8", "single", "0", "0.2", "0.2")},
    # V=8 + alpha 0.15 + ep1 (最强 clip 组合)
    @{name="P3b_V08_a015_ep1"; ep="0001"; args=@("P3b_V08_a015_ep1", "dwt_haar", "0.15", "8", "single", "0", "0.3", "0.3")},
    @{name="P3b_V08_a015_ep4"; ep="0004"; args=@("P3b_V08_a015_ep4", "dwt_haar", "0.15", "8", "single", "0", "0.3", "0.3")},
    # V=8 + alpha 0.2 + ep4 (推 clip 上限)
    @{name="P3b_V08_a02_ep4"; ep="0004"; args=@("P3b_V08_a02_ep4", "dwt_haar", "0.2", "8", "single", "0", "0.3", "0.3")}
)

Write-Host "=== Phase 3b Breakout Ablations ==="
Write-Host "Total ablations: $($ablations.Count)"
Write-Host "Baseline: P3_V_k08 ep7 (clip=0.7348, lpips=0.3868)"
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

    $env:P4_CKPT_PATH = "$base\exp\p4_fusion_breakout\t5_b2v2_d2_d4\epoch_$ep.pt"

    & "C:\Program Files\Python312\python.exe" "$base\_p4_infer_ablation.py" $args 2>&1 | Out-Null
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        Write-Host "  -> FAILED (exit code $exitCode)"
        continue
    }

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
            $hitClip = if ($clip -gt 0.74) { "***HIT***" } else { "" }
            $hitLpips = if ($lpips -lt 0.35) { "lpips_OK" } else { "" }
            Write-Host "  -> clip=$clip lpips=$lpips dist=$dist $hitClip $hitLpips"

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
Write-Host "=== Phase 3b Results Summary (sorted by clip_style desc) ==="
$results | Sort-Object ClipStyle -Descending | Format-Table -AutoSize

Write-Host ""
Write-Host "=== Phase 3b vs Baseline ==="
Write-Host "Baseline P3_V_k08 ep7: clip=0.7348, lpips=0.3868"
Write-Host "Target: clip > 0.74, lpips < 0.35"
Write-Host ""

# Pareto frontier including all P3 + P3b results
Write-Host ""
Write-Host "=== Combined Pareto Frontier (P3 + P3b) ==="
$dir = "$base\exp\p4_fusion_breakout\infer_ablation"
$allResults = @()
# Add P3 results
Get-ChildItem $dir -Filter "P3_*.json" | Where-Object { $_.Name -notmatch "_override" } | ForEach-Object {
    try {
        $data = Get-Content $_.FullName -Raw | ConvertFrom-Json
        $metrics = $data.metrics
        if ($metrics.allpairs_clip_style -ne $null) {
            $allResults += [PSCustomObject]@{
                Name = $data.exp_name
                ClipStyle = [math]::Round($metrics.allpairs_clip_style, 4)
                Lpips = [math]::Round($metrics.allpairs_content_lpips, 4)
            }
        }
    } catch {}
}
# Add P3b results
foreach ($r in $results) {
    $allResults += [PSCustomObject]@{
        Name = $r.Name
        ClipStyle = $r.ClipStyle
        Lpips = $r.Lpips
    }
}

$pareto = @()
foreach ($p in $allResults) {
    $dominated = $false
    foreach ($p2 in $allResults) {
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
$pareto | Sort-Object ClipStyle -Descending | Select-Object -First 15 | Format-Table -AutoSize

Write-Host ""
Write-Host "=== Total elapsed: $([math]::Round(((Get-Date) - $startTime).TotalMinutes, 1)) min ==="

# Save summary
$summaryFile = "$base\exp\p4_fusion_breakout\infer_ablation\_phase3b_summary.txt"
"=== Phase 3b Breakout Summary ===" | Out-File $summaryFile
"Generated: $(Get-Date)" | Out-File $summaryFile -Append
"Baseline: P3_V_k08 ep7 (clip=0.7348, lpips=0.3868)" | Out-File $summaryFile -Append
"Target: clip > 0.74, lpips < 0.35" | Out-File $summaryFile -Append
"" | Out-File $summaryFile -Append
"=== All P3b Results (sorted by clip desc) ===" | Out-File $summaryFile -Append
$results | Sort-Object ClipStyle -Descending | Format-Table -AutoSize | Out-String | Out-File $summaryFile -Append
"" | Out-File $summaryFile -Append
"=== Combined Pareto Frontier ===" | Out-File $summaryFile -Append
$pareto | Sort-Object ClipStyle -Descending | Format-Table -AutoSize | Out-String | Out-File $summaryFile -Append
Write-Host "Summary saved to: $summaryFile"
