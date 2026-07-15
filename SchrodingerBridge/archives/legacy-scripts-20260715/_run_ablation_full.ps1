$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"

Get-ChildItem -Path "src" -Filter "__pycache__" -Directory -Recurse | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$hfCache = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache\hf"

# Training-time sweeps (need training) — 9 configs
$trainSweeps = @(
    @{name="wll_01"; config="configs\abl_wll_01.json"},
    @{name="wll_10"; config="configs\abl_wll_10.json"},
    @{name="wes_02"; config="configs\abl_wes_02.json"},
    @{name="wes_16"; config="configs\abl_wes_16.json"},
    @{name="wes_00"; config="configs\abl_wes_00.json"},
    @{name="wec_0"; config="configs\abl_wec_0.json"},
    @{name="sigma_005"; config="configs\abl_sigma_005.json"},
    @{name="gate_03"; config="configs\abl_gate_03.json"},
    @{name="gate_001"; config="configs\abl_gate_001.json"}
)

# Inference-time sweeps (use baseline checkpoint, only re-eval) — 6 configs
$ckpt = "I:\Github\Latent_Style\SchrodingerBridge\exp\refactor_clean_baseline\epoch_0005.pt"
$inferSweeps = @(
    @{name="adain_05"; override='{"model":{"endpoint_adain_scale":0.5}}'},
    @{name="adain_20"; override='{"model":{"endpoint_adain_scale":2.0}}'},
    @{name="extrap_00"; override='{"model":{"style_extrap_alpha":0.0}}'},
    @{name="extrap_05"; override='{"model":{"style_extrap_alpha":0.5}}'},
    @{name="steps_1"; override='{"full_eval":{"num_steps":1}}'},
    @{name="steps_16"; override='{"full_eval":{"num_steps":16}}'}
)

# === Phase 1: Training sweeps ===
foreach ($exp in $trainSweeps) {
    $name = $exp.name
    $config = $exp.config
    Write-Output "`n=== TRAIN: $name ==="
    $logFile = "C:\Users\Administrator\logs\abl_train_$name.log"
    python run.py --config $config *>&1 | Tee-Object -FilePath $logFile
    Write-Output "TRAIN_EXIT_$name=$LASTEXITCODE"

    if ($LASTEXITCODE -eq 0) {
        $ckptDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\ablation_full\$name"
        $ckptFile = Join-Path $ckptDir "epoch_0005.pt"
        if (Test-Path $ckptFile) {
            Write-Output "=== EVAL: $name ==="
            $outDir = "$ckptDir\eval"
            python run_evaluation.py --checkpoint $ckptFile --output $outDir --batch_size 2 --ref_feature_batch_size 2 --vae_decode_batch_size 16 --test_dir $testDir --force_regen *>&1 | Tee-Object -FilePath "C:\Users\Administrator\logs\abl_eval_$name.log"
            Write-Output "EVAL_EXIT_$name=$LASTEXITCODE"

            if ($LASTEXITCODE -eq 0) {
                $imgDir = "$outDir\images"
                if (Test-Path $imgDir) {
                    $dinoOut = "$outDir\dino_summary.json"
                    python _compute_dino.py --images_dir $imgDir --test_dir $testDir --dataset wikiart --output $dinoOut --hf_cache $hfCache --max_refs 30 *>&1 | Tee-Object -FilePath "C:\Users\Administrator\logs\abl_dino_$name.log"
                    Write-Output "DINO_EXIT_$name=$LASTEXITCODE"
                }
            }
        } else {
            Write-Output "CKPT_MISSING: $ckptFile"
        }
    }
}

# === Phase 2: Inference sweeps ===
foreach ($exp in $inferSweeps) {
    $name = $exp.name
    $override = $exp.override
    $overrideFile = "I:\Github\Latent_Style\SchrodingerBridge\exp\ablation_full\_override_$name.json"
    Set-Content -Path $overrideFile -Value $override -Encoding UTF8
    $outDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\ablation_full\$name"
    Write-Output "`n=== INFER EVAL: $name ==="
    python run_evaluation.py --checkpoint $ckpt --output $outDir --batch_size 2 --ref_feature_batch_size 2 --vae_decode_batch_size 16 --test_dir $testDir --force_regen --config_override $overrideFile *>&1 | Tee-Object -FilePath "C:\Users\Administrator\logs\abl_infer_$name.log"
    Write-Output "INFER_EXIT_$name=$LASTEXITCODE"

    if ($LASTEXITCODE -eq 0) {
        $imgDir = "$outDir\images"
        if (Test-Path $imgDir) {
            $dinoOut = "$outDir\dino_summary.json"
            python _compute_dino.py --images_dir $imgDir --test_dir $testDir --dataset wikiart --output $dinoOut --hf_cache $hfCache --max_refs 30 *>&1 | Tee-Object -FilePath "C:\Users\Administrator\logs\abl_dino_$name.log"
            Write-Output "DINO_EXIT_$name=$LASTEXITCODE"
        }
    }
}

Write-Output "`n=== ALL ABLATION FULL DONE ==="
