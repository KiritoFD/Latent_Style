$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache\hf"

# 710 Phase B ablation matrix (B0-B8)
$exps = @(
    @{name="b0_weave";      config="710_b0_weave_d5.json";     subdir="710_b0_weave";       ep="epoch_0010"},
    @{name="b1_no_dwt";     config="710_b1_no_dwt_route.json";  subdir="710_b1_no_dwt_route"; ep="epoch_0005"},
    @{name="b2_det";        config="710_b2_det_route.json";     subdir="710_b2_det_route";    ep="epoch_0005"},
    @{name="b3_p05";        config="710_b3_p05.json";           subdir="710_b3_p05";          ep="epoch_0005"},
    @{name="b4_no_wct";     config="710_b4_no_wct.json";        subdir="710_b4_no_wct";       ep="epoch_0005"},
    @{name="b5_strong_ll";  config="710_b5_strong_ll.json";     subdir="710_b5_strong_ll";    ep="epoch_0005"},
    @{name="b6_no_ll";      config="710_b6_no_ll.json";         subdir="710_b6_no_ll";        ep="epoch_0005"},
    @{name="b7_2res";       config="710_b7_2res.json";          subdir="710_b7_2res";         ep="epoch_0005"},
    @{name="b8_dim32";      config="710_b8_dim32.json";         subdir="710_b8_dim32";        ep="epoch_0005"}
)

$resultsFile = "I:\Github\Latent_Style\SchrodingerBridge\exp\710_results.txt"
"=== 710 Phase B Start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $resultsFile

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
        $trainMin = [math]::Round(((Get-Date) - $trainStart).TotalMinutes, 1)
        "TRAIN: dur=${trainMin}min" | Out-File $resultsFile -Append
    }
    if (-not (Test-Path $ckpt)) {
        "SKIP EVAL: checkpoint missing" | Out-File $resultsFile -Append
        continue
    }

    # Clean stale eval dir
    if (Test-Path $evalDir) { Remove-Item -Recurse -Force $evalDir }

    # Run eval (CLIP-S + LPIPS only, fast)
    $evalStart = Get-Date
    python -u src\utils\run_evaluation.py `
        --checkpoint $ckpt `
        --output $evalDir `
        --test_dir $testDir `
        --cache_dir $cacheDir `
        --clip_hf_cache_dir $hfCache `
        --batch_size 2 --generation_batch_size 2 --metric_batch_size 2 `
        --target_chunk_size 5 --vae_decode_batch_size 8 `
        --eval_only_lpips_clip_style --eval_lpips_chunk_size 4 --clip_allow_network `
        --no-save_summary_grid `
        *> "exp\${subdir}_eval_log.txt"
    $evalMin = [math]::Round(((Get-Date) - $evalStart).TotalMinutes, 1)
    "EVAL: dur=${evalMin}min" | Out-File $resultsFile -Append

    # Run DINO metrics (DINO-S + DINO-C)
    $dinoStart = Get-Date
    python -u src\utils\compute_dino_metrics.py `
        --eval_dir $evalDir `
        --test_dir $testDir `
        --batch_size 4 --max_refs_per_style 30 `
        --allow_network `
        *> "exp\${subdir}_dino_log.txt"
    $dinoMin = [math]::Round(((Get-Date) - $dinoStart).TotalMinutes, 1)
    "DINO: dur=${dinoMin}min" | Out-File $resultsFile -Append

    # Extract metrics from CSVs
    $metricsCsv = Join-Path $evalDir "metrics.csv"
    $dinoCsv = Join-Path $evalDir "dino_metrics.csv"
    if ((Test-Path $metricsCsv) -and (Test-Path $dinoCsv)) {
        $m = python -c "import csv,math; rows=list(csv.DictReader(open(r'$metricsCsv',encoding='utf-8'))); dino=list(csv.DictReader(open(r'$dinoCsv',encoding='utf-8'))); n=len(rows); off=[i for i,r in enumerate(rows) if r['src_style']!=r['tgt_style']]; cs=sum(float(r['clip_style']) for r in rows)/n; lp=sum(float(r['content_lpips']) for r in rows)/n; ds=sum(float(dino[i]['dino_s']) for i in range(n))/n; dc=sum(float(dino[i]['dino_c']) for i in range(n))/n; ocs=sum(float(rows[i]['clip_style']) for i in off)/len(off); olp=sum(float(rows[i]['content_lpips']) for i in off)/len(off); ods=sum(float(dino[i]['dino_s']) for i in off)/len(off); odc=sum(float(dino[i]['dino_c']) for i in off)/len(off); print(f'all_clip={cs:.4f} all_lpips={lp:.4f} all_dino_s={ds:.4f} all_dino_c={dc:.4f} off_clip={ocs:.4f} off_lpips={olp:.4f} off_dino_s={ods:.4f} off_dino_c={odc:.4f}')"
        "METRICS: $m" | Out-File $resultsFile -Append
    } else {
        "METRICS: CSV missing" | Out-File $resultsFile -Append
    }
}

"=== ALL 710 COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $resultsFile -Append
Get-Content $resultsFile
