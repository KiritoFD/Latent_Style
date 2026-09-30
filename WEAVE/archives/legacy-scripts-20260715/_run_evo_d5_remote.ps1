$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache\hf"

# D5 baseline ref: 20-style eval_r5 clip=0.7213 lpips=0.2728
# Target: clip->0.74, lpips<0.30
$exps = @(
    @{name="baseline"; config="remote_evo_d5_baseline.json"; subdir="evo_d5_baseline"; ep="epoch_0005"},
    @{name="adain10"; config="remote_evo_d5_adain10.json"; subdir="evo_d5_adain10"; ep="epoch_0005"},
    @{name="extrap02"; config="remote_evo_d5_extrap02.json"; subdir="evo_d5_extrap02"; ep="epoch_0005"},
    @{name="long10"; config="remote_evo_d5_long10.json"; subdir="evo_d5_long10"; ep="epoch_0010"},
    @{name="combo"; config="remote_evo_d5_combo.json"; subdir="evo_d5_combo"; ep="epoch_0010"}
)

$resultsFile = "I:\Github\Latent_Style\SchrodingerBridge\exp\evo_d5_timing_results.txt"
"=== Evo D5 Remote Start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $resultsFile

foreach ($entry in $exps) {
    $name = $entry.name
    $config = "configs\$($entry.config)"
    $subdir = $entry.subdir
    $finalEp = $entry.ep
    $ckpt = "exp\$subdir\$finalEp.pt"
    $evalDir = "exp\$subdir\full_eval\$finalEp"

    "=== EXP=$name START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $resultsFile -Append

    # Skip training if checkpoint exists
    if (Test-Path $ckpt) {
        "TRAIN: skipped (checkpoint exists)" | Out-File $resultsFile -Append
    } else {
        $trainStart = Get-Date
        python -u src\run.py --config $config 2>&1 | Tee-Object -FilePath "exp\${subdir}_log.txt"
        $trainEc = $LASTEXITCODE
        $trainMin = [math]::Round(((Get-Date) - $trainStart).TotalMinutes, 1)
        "TRAIN: exit=$trainEc dur=${trainMin}min" | Out-File $resultsFile -Append
    }
    if (-not (Test-Path $ckpt)) {
        "SKIP EVAL: checkpoint missing" | Out-File $resultsFile -Append
        continue
    }

    # Clean stale eval dir (in-process eval may have left empty images dir)
    if (Test-Path $evalDir) { Remove-Item -Recurse -Force $evalDir }

    $evalStart = Get-Date
    python -u src\utils\run_evaluation.py `
        --checkpoint $ckpt `
        --output $evalDir `
        --test_dir $testDir `
        --cache_dir $cacheDir `
        --clip_hf_cache_dir $hfCache `
        --batch_size 2 --generation_batch_size 2 --metric_batch_size 2 `
        --target_chunk_size 5 --vae_decode_batch_size 8 `
        --eval_only_lpips_clip_style --eval_lpips_chunk_size 4 --clip_allow_network 2>&1 | Tee-Object -FilePath "exp\${subdir}_eval_log.txt"
    $evalMin = [math]::Round(((Get-Date) - $evalStart).TotalMinutes, 1)
    "EVAL: exit=$LASTEXITCODE dur=${evalMin}min" | Out-File $resultsFile -Append

    $summary = Join-Path $evalDir "summary.json"
    if (Test-Path $summary) {
        $m = python -c "import json; d=json.load(open(r'$summary',encoding='utf-8')); apo=d.get('analysis',{}).get('all_pairs_overview',{}); sta=d.get('analysis',{}).get('style_transfer_ability',{}); print(f'all_clip={apo.get(\"clip_style\",0):.4f} all_lpips={apo.get(\"content_lpips\",0):.4f} sta_clip={sta.get(\"clip_style\",0):.4f} sta_lpips={sta.get(\"content_lpips\",0):.4f}')"
        "METRICS: $m" | Out-File $resultsFile -Append
    } else {
        "METRICS: summary not found" | Out-File $resultsFile -Append
    }
}

"=== ALL EVO D5 COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $resultsFile -Append
Get-Content $resultsFile
