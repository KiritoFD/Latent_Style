$ErrorActionPreference = "Continue"
Set-Location "G:\GitHub\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$testDir = "G:\GitHub\Latent_Style\Dataset\wikiart_random20_512\wikiart_random20_512\images\test"
$cacheDir = "G:\GitHub\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "G:\GitHub\Latent_Style\SchrodingerBridge\exp\eval_cache\hf"

# exp name -> (save_subdir, final_epoch)
# save_subdir matches checkpoint.save_dir basename in config
$exps = @(
    @{name="evo_a1_adain10"; subdir="evo_r20_a1_adain10"; ep="epoch_0005"},
    @{name="evo_a2_long10"; subdir="evo_r20_a2_long10"; ep="epoch_0010"},
    @{name="evo_a3_extrap02"; subdir="evo_r20_a3_extrap02"; ep="epoch_0005"},
    @{name="evo_a4_combo"; subdir="evo_r20_a4_combo"; ep="epoch_0010"}
)

$resultsFile = "exp\evo_r20_timing_results.txt"
"=== Evo R20 Start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $resultsFile

foreach ($entry in $exps) {
    $exp = $entry.name
    $subdir = $entry.subdir
    $finalEp = $entry.ep
    $config = "configs\$exp.json"
    $ckpt = "exp\$subdir\$finalEp.pt"
    $evalDir = "exp\$subdir\full_eval\$finalEp"

    "=== EXP=$exp START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $resultsFile -Append

    # Skip training if checkpoint already exists
    if (Test-Path $ckpt) {
        "TRAIN: skipped (checkpoint exists)" | Out-File $resultsFile -Append
    } else {
        $trainStart = Get-Date
        python -u src\run.py --config $config 2>&1 | Tee-Object -FilePath "exp\${subdir}_log.txt"
        $trainEc = $LASTEXITCODE
        $trainMin = [math]::Round(((Get-Date) - $trainStart).TotalMinutes, 1)
        "TRAIN: exit=$trainEc dur=${trainMin}min" | Out-File $resultsFile -Append
        if ($trainEc -ne 0 -or -not (Test-Path $ckpt)) {
            "SKIP EVAL: checkpoint missing" | Out-File $resultsFile -Append
            continue
        }
    }

    $evalStart = Get-Date
    python -u src\utils\run_evaluation.py `
        --checkpoint $ckpt `
        --output $evalDir `
        --test_dir $testDir `
        --cache_dir $cacheDir `
        --clip_hf_cache_dir $hfCache `
        --batch_size 2 --generation_batch_size 2 --metric_batch_size 2 `
        --target_chunk_size 2 --vae_decode_batch_size 16 `
        --eval_only_lpips_clip_style --eval_lpips_chunk_size 4 2>&1 | Tee-Object -FilePath "exp\${subdir}_log.txt" -Append
    $evalMin = [math]::Round(((Get-Date) - $evalStart).TotalMinutes, 1)
    "EVAL: exit=$LASTEXITCODE dur=${evalMin}min" | Out-File $resultsFile -Append

    $summary = Join-Path $evalDir "summary.json"
    if (Test-Path $summary) {
        python -c "import json; d=json.load(open(r'$summary','r',encoding='utf-8')); apo=d.get('analysis',{}).get('all_pairs_overview',{}); print(f'METRICS: clip={apo.get(chr(34)+chr(99)+chr(108)+chr(105)+chr(112)+chr(95)+chr(115)+chr(116)+chr(121)+chr(108)+chr(101)+chr(34),0):.4f} lpips={apo.get(chr(34)+chr(99)+chr(111)+chr(110)+chr(116)+chr(101)+chr(110)+chr(116)+chr(95)+chr(108)+chr(112)+chr(105)+chr(112)+chr(115)+chr(34),0):.4f}')" | Out-File $resultsFile -Append
    } else {
        "METRICS: summary not found" | Out-File $resultsFile -Append
    }
}

"=== ALL EVO R20 COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $resultsFile -Append
Get-Content $resultsFile
