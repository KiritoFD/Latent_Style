$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"

Get-ChildItem -Path "src" -Filter "__pycache__" -Directory -Recurse | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$hfCache = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache\hf"

# === Phase 1: Training experiments (14 configs) ===
$trainExps = @(
    "a01_wo_endpoint_adain", "a02_wo_cross_attn", "a03_wo_flow",
    "b01_wll_0", "b02_wll_20",
    "b03_sigma_0", "b04_sigma_02",
    "b05_gate_001", "b06_gate_10",
    "b07_whh_0", "b08_whh_4",
    "b09_lr_5e5", "b10_lr_5e4",
    "b11_loss_huber"
)

foreach ($name in $trainExps) {
    $config = "configs\ablation_v2\$name.json"
    Write-Output "`n=== TRAIN: $name ==="
    $logFile = "C:\Users\Administrator\logs\v2_train_$name.log"
    python run.py --config $config *>&1 | Tee-Object -FilePath $logFile
    Write-Output "TRAIN_EXIT_$name=$LASTEXITCODE"

    if ($LASTEXITCODE -eq 0) {
        $ckptDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\ablation_v2\$name"
        $ckptFile = Join-Path $ckptDir "epoch_0005.pt"
        if (Test-Path $ckptFile) {
            Write-Output "=== EVAL: $name ==="
            $outDir = "$ckptDir\eval"
            $evalLog = "C:\Users\Administrator\logs\v2_eval_$name.log"
            python run_evaluation.py --checkpoint $ckptFile --output $outDir --batch_size 2 --ref_feature_batch_size 2 --vae_decode_batch_size 16 --test_dir $testDir --force_regen *>&1 | Tee-Object -FilePath $evalLog
            Write-Output "EVAL_EXIT_$name=$LASTEXITCODE"

            if ($LASTEXITCODE -eq 0) {
                $imgDir = "$outDir\images"
                if (Test-Path $imgDir) {
                    $dinoOut = "$outDir\dino_summary.json"
                    $dinoLog = "C:\Users\Administrator\logs\v2_dino_$name.log"
                    python _compute_dino.py --images_dir $imgDir --test_dir $testDir --dataset wikiart --output $dinoOut --hf_cache $hfCache --max_refs 30 *>&1 | Tee-Object -FilePath $dinoLog
                    Write-Output "DINO_EXIT_$name=$LASTEXITCODE"
                }
            }
        } else {
            Write-Output "CKPT_MISSING: $ckptFile"
        }
    }
    # Cleanup __pycache__ between runs
    Get-ChildItem -Path "src" -Filter "__pycache__" -Directory -Recurse | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}

# === Phase 2: Inference experiments (7 configs, use baseline ckpt) ===
$baselineCkpt = "I:\Github\Latent_Style\SchrodingerBridge\exp\refactor_clean_baseline\epoch_0005.pt"
$inferExps = @(
    @{name="d01_adain_0"; override='{"model":{"endpoint_adain_scale":0.0}}'},
    @{name="d02_adain_05"; override='{"model":{"endpoint_adain_scale":0.5}}'},
    @{name="d03_adain_20"; override='{"model":{"endpoint_adain_scale":2.0}}'},
    @{name="d04_extrap_00"; override='{"model":{"style_extrap_alpha":0.0}}'},
    @{name="d05_extrap_10"; override='{"model":{"style_extrap_alpha":1.0}}'},
    @{name="d06_steps_1"; override='{"full_eval":{"num_steps":1}}'},
    @{name="d07_steps_32"; override='{"full_eval":{"num_steps":32}}'}
)

foreach ($exp in $inferExps) {
    $name = $exp.name
    $override = $exp.override
    $overrideFile = "I:\Github\Latent_Style\SchrodingerBridge\exp\ablation_v2\_override_$name.json"
    [System.IO.File]::WriteAllText($overrideFile, $override, [System.Text.UTF8Encoding]::new($false))
    $outDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\ablation_v2\$name"
    Write-Output "`n=== INFER EVAL: $name ==="
    $logFile = "C:\Users\Administrator\logs\v2_infer_$name.log"
    python run_evaluation.py --checkpoint $baselineCkpt --output $outDir --batch_size 2 --ref_feature_batch_size 2 --vae_decode_batch_size 16 --test_dir $testDir --force_regen --config_override $overrideFile *>&1 | Tee-Object -FilePath $logFile
    Write-Output "INFER_EXIT_$name=$LASTEXITCODE"

    if ($LASTEXITCODE -eq 0) {
        $imgDir = "$outDir\images"
        if (Test-Path $imgDir) {
            $dinoOut = "$outDir\dino_summary.json"
            $dinoLog = "C:\Users\Administrator\logs\v2_dino_$name.log"
            python _compute_dino.py --images_dir $imgDir --test_dir $testDir --dataset wikiart --output $dinoOut --hf_cache $hfCache --max_refs 30 *>&1 | Tee-Object -FilePath $dinoLog
            Write-Output "DINO_EXIT_$name=$LASTEXITCODE"
        }
    }
    Get-ChildItem -Path "src" -Filter "__pycache__" -Directory -Recurse | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Output "`n=== ALL ABLATION V2 DONE ==="
