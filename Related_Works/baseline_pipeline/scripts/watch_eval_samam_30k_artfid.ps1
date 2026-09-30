$ErrorActionPreference = "Stop"

$Repo = "G:\GitHub\Latent_Style"
$TrainRoot = Join-Path $Repo "Related_Works\baseline_pipeline\results\samam_wsl_mamba_b2_30k_continue"
$TrainLog = Join-Path $TrainRoot "train.log"
$OutRoot = Join-Path $TrainRoot "curve_eval_sb_artfid_5src"
$SbSrc = Join-Path $Repo "SchrodingerBridge\src"
$TestDir = Join-Path $Repo "style_data\overfit50"
$Clip = Join-Path $Repo "eval_cache\manual_clip\openai-clip-vit-base-patch32"

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$WatchLog = Join-Path $OutRoot "watch.log"
"WATCH_START $(Get-Date -Format o)" | Tee-Object -FilePath $WatchLog -Append

while ($true) {
    if (Test-Path $TrainLog) {
        $tail = Get-Content $TrainLog -Tail 30 -ErrorAction SilentlyContinue
        if ($tail -match "END .* STATUS=0") {
            break
        }
    }
    Start-Sleep -Seconds 30
}

"TRAIN_DONE $(Get-Date -Format o)" | Tee-Object -FilePath $WatchLog -Append

$GenCmd = @'
cd /mnt/g/GitHub/Latent_Style
source /root/venvs/samam/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="/usr/local/cuda-12.8/bin:$PATH"
export PYTHONPATH=/mnt/g/GitHub/Latent_Style/Related_Works/repos/SaMam:/mnt/g/GitHub/Latent_Style
python Related_Works/baseline_pipeline/scripts/eval_samam_checkpoint_curve.py --ckpt-dir /mnt/g/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samam_wsl_mamba_b2_30k_continue/step_checkpoints --image-root /mnt/g/GitHub/Latent_Style/style_data/overfit50 --output-root /mnt/g/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/samam_wsl_mamba_b2_30k_continue/curve_eval_sb_artfid_5src --image-size 256 --max-src-per-style 5 --generate-only
'@
$GenCmd = $GenCmd -replace "`r`n", "`n"
wsl bash -lc $GenCmd *> (Join-Path $OutRoot "generate.log")
if ($LASTEXITCODE -ne 0) {
    throw "generate failed with exit code $LASTEXITCODE"
}
"GENERATE_DONE $(Get-Date -Format o)" | Tee-Object -FilePath $WatchLog -Append

$EvalLog = Join-Path $OutRoot "sb_artfid_eval_all.log"
"" | Set-Content $EvalLog
foreach ($d in Get-ChildItem $OutRoot -Directory | Where-Object { $_.Name -match '^step_\d+$' } | Sort-Object Name) {
    "[EVAL_ARTFID] $($d.Name)" | Tee-Object -FilePath $EvalLog -Append
    py -3 -m utils.run_evaluation `
        --output $d.FullName `
        --test_dir $TestDir `
        --style_subdirs photo,monet,vangogh,cezanne,Hayao `
        --reuse_generated `
        --force_regen `
        --eval_only_lpips_clip_style `
        --eval_enable_art_fid `
        --eval_art_fid_max_gen 5 `
        --eval_art_fid_max_ref 80 `
        --eval_art_fid_batch_size 4 `
        --no-eval_enable_kid `
        --max_src_samples 5 `
        --batch_size 8 `
        --eval_lpips_chunk_size 2 `
        --clip_model_name $Clip 2>&1 | Tee-Object -FilePath $EvalLog -Append
    if ($LASTEXITCODE -ne 0) {
        throw "SB ArtFID eval failed for $($d.Name)"
    }
}

py -3 (Join-Path $Repo "Related_Works\baseline_pipeline\scripts\collect_sb_curve_metrics.py") `
    --root $OutRoot `
    --title "SaMAM 256 16k-30k SB eval" 2>&1 | Tee-Object -FilePath (Join-Path $OutRoot "collect.log")
if ($LASTEXITCODE -ne 0) {
    throw "collect failed with exit code $LASTEXITCODE"
}

"EVAL_DONE $(Get-Date -Format o)" | Tee-Object -FilePath $WatchLog -Append
