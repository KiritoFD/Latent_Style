# T11 evolution Round 2: isolate single-parameter effects
# All configs keep scale=0.5 (T11 baseline), change only one param
$ErrorActionPreference = "Continue"
Set-Location "G:\GitHub\Latent_Style\SchrodingerBridge"
$env:PYTHONPATH = "."
$env:PYTHONIOENCODING = "utf-8"

$testDir = "G:\GitHub\Latent_Style\Dataset\distinct5_512\test"
$cacheDir = "G:\GitHub\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "G:\GitHub\Latent_Style\SchrodingerBridge\exp\eval_cache\hf"

$exps = @(
    @{name="t11e_f_alpha05"; epochs=10; finalEp="epoch_0010"},
    @{name="t11e_g_alpha06"; epochs=10; finalEp="epoch_0010"},
    @{name="t11e_h_hh06"; epochs=10; finalEp="epoch_0010"},
    @{name="t11e_i_p10"; epochs=10; finalEp="epoch_0010"},
    @{name="t11e_j_long20"; epochs=20; finalEp="epoch_0020"}
)

$resultsFile = "G:\GitHub\Latent_Style\SchrodingerBridge\exp\t11evo\timing_results_r2.txt"
"=== T11 Evo R2 Start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $resultsFile -Encoding utf8

foreach ($expInfo in $exps) {
    $exp = $expInfo.name
    $config = "configs\$exp.json"
    $ckpt = "exp\t11evo\$exp\$($expInfo.finalEp).pt"
    $evalDir = "exp\t11evo\$exp\full_eval\$($expInfo.finalEp)"
    $logOut = "G:\GitHub\Latent_Style\SchrodingerBridge\exp\t11evo\${exp}_train_eval.log"

    Write-Output "=== EXP=$exp START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
    "=== EXP=$exp START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $resultsFile -Append -Encoding utf8

    # TRAIN
    $trainStart = Get-Date
    python -u src\run.py --config $config 2>&1 | Tee-Object -FilePath $logOut
    $trainEc = $LASTEXITCODE
    $trainEnd = Get-Date
    $trainMin = [math]::Round(($trainEnd - $trainStart).TotalMinutes, 1)
    "TRAIN: exit=$trainEc duration=${trainMin}min" | Out-File $resultsFile -Append -Encoding utf8
    Write-Output "TRAIN DONE exit=$trainEc dur=${trainMin}min"

    if ($trainEc -ne 0 -or -not (Test-Path $ckpt)) {
        $anyCkpt = Get-ChildItem "exp\t11evo\$exp" -Filter "*.pt" -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
        if ($anyCkpt) {
            $ckpt = $anyCkpt.FullName
            $evalDir = "exp\t11evo\$exp\full_eval\$($anyCkpt.BaseName)"
        } else {
            "FATAL: no checkpoint. SKIP." | Out-File $resultsFile -Append -Encoding utf8
            continue
        }
    }

    # EVAL
    $evalStart = Get-Date
    python -u src\utils\run_evaluation.py `
        --checkpoint $ckpt `
        --output $evalDir `
        --test_dir $testDir `
        --cache_dir $cacheDir `
        --clip_hf_cache_dir $hfCache `
        --batch_size 2 --generation_batch_size 2 --metric_batch_size 2 `
        --target_chunk_size 2 --vae_decode_batch_size 16 `
        --eval_only_lpips_clip_style --eval_lpips_chunk_size 4 2>&1 | Tee-Object -FilePath $logOut -Append
    $evalEc = $LASTEXITCODE
    $evalEnd = Get-Date
    $evalMin = [math]::Round(($evalEnd - $evalStart).TotalMinutes, 1)
    "EVAL: exit=$evalEc duration=${evalMin}min" | Out-File $resultsFile -Append -Encoding utf8
    Write-Output "EVAL DONE exit=$evalEc dur=${evalMin}min"

    # Extract metrics
    $summary = Join-Path $evalDir "summary.json"
    if (Test-Path $summary) {
        $metrics = python -c "import json; d=json.load(open(r'$summary','r',encoding='utf-8')); apo=d.get('analysis',{}).get('all_pairs_overview',{}); print(f'clip={apo.get(\"clip_style\",\"N/A\"):.4f} lpips={apo.get(\"content_lpips\",\"N/A\"):.4f}')"
        "METRICS: $metrics" | Out-File $resultsFile -Append -Encoding utf8
        Write-Output "METRICS: $metrics"
    }
    "" | Out-File $resultsFile -Append -Encoding utf8
}
"=== ALL R2 COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $resultsFile -Append -Encoding utf8
Get-Content $resultsFile
