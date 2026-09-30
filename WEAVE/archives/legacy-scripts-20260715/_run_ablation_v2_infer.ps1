$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"

$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$hfCache = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache\hf"
$baselineCkpt = "I:\Github\Latent_Style\SchrodingerBridge\exp\ablation_v2\b03_sigma_0\epoch_0005.pt"

# Use b03_sigma_0 checkpoint as baseline (contract_family=weave, same as baseline since sigma=0 is close to default 0.02)
# Actually, use a02_wo_cross_attn which has contract_family=weave in checkpoint
# No - better to use the override approach with contract_family fix

# Use refactor_clean_baseline checkpoint but override contract_family
$inferExps = @(
    @{name="d01_adain_0"; override='{"model":{"endpoint_adain_scale":0.0,"contract_family":"weave"}}'},
    @{name="d02_adain_05"; override='{"model":{"endpoint_adain_scale":0.5,"contract_family":"weave"}}'},
    @{name="d03_adain_20"; override='{"model":{"endpoint_adain_scale":2.0,"contract_family":"weave"}}'},
    @{name="d04_extrap_00"; override='{"model":{"style_extrap_alpha":0.0,"contract_family":"weave"}}'},
    @{name="d05_extrap_10"; override='{"model":{"style_extrap_alpha":1.0,"contract_family":"weave"}}'},
    @{name="d06_steps_1"; override='{"model":{"contract_family":"weave"},"full_eval":{"num_steps":1}}'},
    @{name="d07_steps_32"; override='{"model":{"contract_family":"weave"},"full_eval":{"num_steps":32}}'}
)

$baselineCkpt = "I:\Github\Latent_Style\SchrodingerBridge\exp\refactor_clean_baseline\epoch_0005.pt"

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

Write-Output "`n=== ALL INFER ABLATION V2 DONE ==="
