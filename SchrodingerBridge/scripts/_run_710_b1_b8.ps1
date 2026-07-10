Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache\hf"

# B1-B8 experiment definitions: name, config, save_dir
$experiments = @(
    @("b1_no_dwt_route", "710_b1_no_dwt_route.json", "710_b1_no_dwt_route"),
    @("b2_det_route",    "710_b2_det_route.json",    "710_b2_det_route"),
    @("b3_p05",          "710_b3_p05.json",          "710_b3_p05"),
    @("b4_no_wct",       "710_b4_no_wct.json",       "710_b4_no_wct"),
    @("b5_strong_ll",    "710_b5_strong_ll.json",    "710_b5_strong_ll"),
    @("b6_no_ll",        "710_b6_no_ll.json",        "710_b6_no_ll"),
    @("b7_2res",         "710_b7_2res.json",         "710_b7_2res"),
    @("b8_dim32",        "710_b8_dim32.json",        "710_b8_dim32")
)

$resultsFile = "exp\710_results.txt"
"run,train_min,eval_min,dino_min,all_clip_s,all_lpips,all_dino_s,all_dino_c,off_clip_s,off_lpips,off_dino_s,off_dino_c,wall_total_sec,lancet_gen_sec" | Out-File $resultsFile -Encoding utf8

foreach ($exp in $experiments) {
    $name = $exp[0]
    $config = $exp[1]
    $saveDir = $exp[2]
    $ckpt = "exp\$saveDir\epoch_0005.pt"
    $evalDir = "exp\$saveDir\full_eval\epoch_0005"

    Write-Host "`n========== $name =========="

    # Step 1: Train (skip if checkpoint exists)
    if (-not (Test-Path $ckpt)) {
        Write-Host "TRAIN: $name"
        $trainStart = Get-Date
        python -u src\run.py --config "configs\$config" *> "exp\${saveDir}_log.txt" 2>&1
        $trainMin = [math]::Round(((Get-Date) - $trainStart).TotalMinutes, 1)
        Write-Host "TRAIN done: ${trainMin}min"
        if (-not (Test-Path $ckpt)) {
            Write-Host "ERROR: checkpoint not found after training: $ckpt"
            "$name,ERROR,0,0,0,0,0,0,0,0,0,0,0" | Out-File $resultsFile -Encoding utf8 -Append
            continue
        }
    } else {
        Write-Host "SKIP TRAIN (checkpoint exists): $ckpt"
        $trainMin = 0
    }

    # Step 2: Eval (always re-run to save images)
    if (Test-Path $evalDir) { Remove-Item -Recurse -Force $evalDir }
    Write-Host "EVAL: $name"
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
        --save_generated_images --no-save_summary_grid `
        *> "exp\${saveDir}_eval_log.txt" 2>&1
    $evalMin = [math]::Round(((Get-Date) - $evalStart).TotalMinutes, 1)
    Write-Host "EVAL done: ${evalMin}min"

    if (-not (Test-Path "$evalDir\metrics.csv")) {
        Write-Host "ERROR: metrics.csv not found after eval"
        "$name,$trainMin,$evalMin,0,0,0,0,0,0,0,0,0,0" | Out-File $resultsFile -Encoding utf8 -Append
        continue
    }

    # Step 3: DINO metrics
    Write-Host "DINO: $name"
    $dinoStart = Get-Date
    python -u src\utils\compute_dino_metrics.py `
        --eval_dir $evalDir `
        --test_dir $testDir `
        --batch_size 4 --max_refs_per_style 30 `
        --allow_network `
        *> "exp\${saveDir}_dino_log.txt" 2>&1
    $dinoMin = [math]::Round(((Get-Date) - $dinoStart).TotalMinutes, 1)
    Write-Host "DINO done: ${dinoMin}min"

    # Step 4: Extract metrics
    python -c "import csv,json; rows=list(csv.DictReader(open(r'$evalDir\metrics.csv',encoding='utf-8'))); dino=list(csv.DictReader(open(r'$evalDir\dino_metrics.csv',encoding='utf-8'))); n=len(rows); off=[i for i,r in enumerate(rows) if r['src_style']!=r['tgt_style']]; no=len(off); ac=sum(float(r['clip_style']) for r in rows)/n; al=sum(float(r['content_lpips']) for r in rows)/n; ads=sum(float(dino[i]['dino_s']) for i in range(n))/n; adc=sum(float(dino[i]['dino_c']) for i in range(n))/n; oc=sum(float(rows[i]['clip_style']) for i in off)/no; ol=sum(float(rows[i]['content_lpips']) for i in off)/no; ods=sum(float(dino[i]['dino_s']) for i in off)/no; odc=sum(float(dino[i]['dino_c']) for i in off)/no; s=json.load(open(r'$evalDir\summary.json')); t=s.get('timings_sec',{}); wt=t.get('wall_total',0); lg=t.get('lancet_generation',0); print(f'$name,$trainMin,$evalMin,$dinoMin,{ac:.4f},{al:.4f},{ads:.4f},{adc:.4f},{oc:.4f},{ol:.4f},{ods:.4f},{odc:.4f},{wt:.1f},{lg:.1f}')" | Out-File $resultsFile -Encoding utf8 -Append
}

Write-Host "`n========== ALL DONE =========="
Get-Content $resultsFile
