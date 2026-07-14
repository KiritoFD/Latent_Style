# Eval brk_a at different AdaIN scales (no training needed)
# FIX: use --config_override (not --config) to properly override endpoint_adain_scale
$ErrorActionPreference = "Continue"
$Root = "I:\Github\Latent_Style\SchrodingerBridge"
$Py = "C:\Program Files\Python312\python.exe"
Set-Location $Root
$env:PYTHONIOENCODING = "utf-8"

$TestDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$HfCache = "C:\Users\Administrator\.cache\huggingface\hub"
$Ckpt = "I:\Github\Latent_Style\SchrodingerBridge\exp\dino_s_break\brk_a_ll03_10ep\epoch_0010.pt"

# Evaluate at AdaIN=0.0 (no post-processing)
$Name0 = "brk_a_adain00"
$EvalDir0 = "$Root\exp\model_probe\brk_a_adain_sweep_v2\adain00"
$LogDir0 = "$Root\exp\model_probe\brk_a_adain_sweep_v2\adain00\logs"
New-Item -ItemType Directory -Force -Path $LogDir0 | Out-Null
Write-Output "=== [$Name0] EVAL START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
& $Py -u "$Root\src\utils\run_evaluation.py" `
    --config_override "$Root\configs\eval_adain_00.json" `
    --checkpoint $Ckpt `
    --output $EvalDir0 `
    --save_generated_images `
    --batch_size 2 `
    --clip_hf_cache_dir $HfCache 2>&1 > "$LogDir0\eval.log"
if ($LASTEXITCODE -ne 0) { Write-Output "=== [$Name0] EVAL FAILED ===" }
Write-Output "=== [$Name0] EVAL DONE ==="
& $Py -u "$Root\_compute_dino.py" `
    --images_dir "$EvalDir0\images" --test_dir $TestDir --dataset wikiart `
    --output "$EvalDir0\dino.json" --hf_cache $HfCache --max_refs 30 2>&1 > "$LogDir0\dino.log"
if ($LASTEXITCODE -ne 0) { Write-Output "=== [$Name0] DINO FAILED ===" }
Write-Output "=== [$Name0] DINO DONE ==="

# Evaluate at AdaIN=3.0 (stronger post-processing)
$Name3 = "brk_a_adain30"
$EvalDir3 = "$Root\exp\model_probe\brk_a_adain_sweep_v2\adain30"
$LogDir3 = "$Root\exp\model_probe\brk_a_adain_sweep_v2\adain30\logs"
New-Item -ItemType Directory -Force -Path $LogDir3 | Out-Null
Write-Output "=== [$Name3] EVAL START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
& $Py -u "$Root\src\utils\run_evaluation.py" `
    --config_override "$Root\configs\eval_adain_30.json" `
    --checkpoint $Ckpt `
    --output $EvalDir3 `
    --save_generated_images `
    --batch_size 2 `
    --clip_hf_cache_dir $HfCache 2>&1 > "$LogDir3\eval.log"
if ($LASTEXITCODE -ne 0) { Write-Output "=== [$Name3] EVAL FAILED ===" }
Write-Output "=== [$Name3] EVAL DONE ==="
& $Py -u "$Root\_compute_dino.py" `
    --images_dir "$EvalDir3\images" --test_dir $TestDir --dataset wikiart `
    --output "$EvalDir3\dino.json" --hf_cache $HfCache --max_refs 30 2>&1 > "$LogDir3\dino.log"
if ($LASTEXITCODE -ne 0) { Write-Output "=== [$Name3] DINO FAILED ===" }
Write-Output "=== [$Name3] DINO DONE ==="

Write-Output "=== SWEEP COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

# Also re-evaluate AdaIN=2.0 with --config_override for fair comparison
$Name2 = "brk_a_adain20"
$EvalDir2 = "$Root\exp\model_probe\brk_a_adain_sweep_v2\adain20"
$LogDir2 = "$Root\exp\model_probe\brk_a_adain_sweep_v2\adain20\logs"
New-Item -ItemType Directory -Force -Path $LogDir2 | Out-Null
Write-Output "=== [$Name2] EVAL START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
& $Py -u "$Root\src\utils\run_evaluation.py" `
    --config_override "$Root\configs\eval_adain_20.json" `
    --checkpoint $Ckpt `
    --output $EvalDir2 `
    --save_generated_images `
    --batch_size 2 `
    --clip_hf_cache_dir $HfCache 2>&1 > "$LogDir2\eval.log"
if ($LASTEXITCODE -ne 0) { Write-Output "=== [$Name2] EVAL FAILED ===" }
Write-Output "=== [$Name2] EVAL DONE ==="
& $Py -u "$Root\_compute_dino.py" `
    --images_dir "$EvalDir2\images" --test_dir $TestDir --dataset wikiart `
    --output "$EvalDir2\dino.json" --hf_cache $HfCache --max_refs 30 2>&1 > "$LogDir2\dino.log"
if ($LASTEXITCODE -ne 0) { Write-Output "=== [$Name2] DINO FAILED ===" }
Write-Output "=== [$Name2] DINO DONE ==="
Write-Output "=== ALL SWEEP COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
