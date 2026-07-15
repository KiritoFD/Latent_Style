Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"

# All B0-B8 experiments with existing eval images
$experiments = @(
    @("b0_t11",          "710_b0_t11"),
    @("b1_no_dwt_route", "710_b1_no_dwt_route"),
    @("b2_det_route",    "710_b2_det_route"),
    @("b3_p05",          "710_b3_p05"),
    @("b4_no_wct",       "710_b4_no_wct"),
    @("b5_strong_ll",    "710_b5_strong_ll"),
    @("b6_no_ll",        "710_b6_no_ll"),
    @("b7_2res",         "710_b7_2res"),
    @("b8_dim32",        "710_b8_dim32")
)

$resultsFile = "exp\710_canonical_dino_results.txt"
"run,n_all,n_off,all_clip_s,all_lpips,all_dino_s,all_dino_c,all_dino_structure,off_clip_s,off_lpips,off_dino_s,off_dino_c,off_dino_structure" | Out-File $resultsFile -Encoding utf8

foreach ($exp in $experiments) {
    $name = $exp[0]
    $saveDir = $exp[1]
    $evalDir = "exp\$saveDir\full_eval\epoch_0005"

    Write-Host "`n========== $name =========="

    if (-not (Test-Path "$evalDir\metrics.csv")) {
        Write-Host "SKIP: metrics.csv not found"
        continue
    }
    if (-not (Test-Path "$evalDir\images")) {
        Write-Host "SKIP: images/ dir not found"
        continue
    }

    # Run canonical DINO metrics
    $dinoStart = Get-Date
    python -u src\utils\compute_dino_metrics.py `
        --eval_dir $evalDir `
        --test_dir $testDir `
        --batch_size 4 --max_refs_per_style 30 `
        --exclude_source_from_style_refs `
        --allow_network `
        *> "exp\${saveDir}_canonical_dino_log.txt" 2>&1
    $dinoMin = [math]::Round(((Get-Date) - $dinoStart).TotalMinutes, 1)
    Write-Host "DINO done: ${dinoMin}min"

    # Extract from metrics.csv + dino_summary.json
    $summaryPath = "$evalDir\dino_summary.json"
    if (Test-Path $summaryPath) {
        python -u C:\Users\Administrator\_710_extract_run.py $evalDir $name | Out-File $resultsFile -Encoding utf8 -Append
    } else {
        Write-Host "ERROR: dino_summary.json not found"
        "$name,ERROR,0,0,0,0,0,0,0,0,0,0,0" | Out-File $resultsFile -Encoding utf8 -Append
    }
}

Write-Host "`n========== ALL DONE =========="
Get-Content $resultsFile
