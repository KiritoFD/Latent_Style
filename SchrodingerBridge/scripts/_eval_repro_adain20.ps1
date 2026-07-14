# Re-eval bf16 checkpoint with correct AdaIN=2.0 override (no retraining)
$ErrorActionPreference = "Continue"
$Root = "I:\Github\Latent_Style\SchrodingerBridge"
$Py = "C:\Program Files\Python312\python.exe"
Set-Location $Root
$env:PYTHONIOENCODING = "utf-8"

$TestDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$HfCache = "C:\Users\Administrator\.cache\huggingface\hub"
$Ckpt = "$Root\exp\repro\bf16\epoch_0010.pt"
$EvalDir = "$Root\exp\repro\bf16_adain20"
$LogDir = "$EvalDir\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Write-Output "=== [bf16_adain20] EVAL START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
& $Py -u "$Root\src\utils\run_evaluation.py" `
    --config_override "$Root\configs\eval_adain_20.json" `
    --checkpoint $Ckpt `
    --output $EvalDir `
    --save_generated_images `
    --batch_size 2 `
    --ref_feature_batch_size 2 `
    --clip_hf_cache_dir $HfCache 2>&1 > "$LogDir\eval.log"
if ($LASTEXITCODE -ne 0) { Write-Output "=== EVAL FAILED ==="; exit 1 }
Write-Output "=== [bf16_adain20] EVAL DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

Write-Output "=== [bf16_adain20] DINO START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
& $Py -u "$Root\_compute_dino.py" `
    --images_dir "$EvalDir\images" --test_dir $TestDir --dataset wikiart `
    --output "$EvalDir\dino.json" --hf_cache $HfCache --max_refs 30 2>&1 > "$LogDir\dino.log"
if ($LASTEXITCODE -ne 0) { Write-Output "=== DINO FAILED ==="; exit 1 }
Write-Output "=== [bf16_adain20] DINO DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Write-Output "=== ALL COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
