# Pipeline: Round7 Step B — phase-anchored HF transport training + AdaIN=2.0 eval
$ErrorActionPreference = "Continue"
$Root = "I:\Github\Latent_Style\SchrodingerBridge"
$Py = "C:\Program Files\Python312\python.exe"
Set-Location $Root
$env:PYTHONIOENCODING = "utf-8"

$TestDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$CacheDir = "$Root\exp\eval_cache"
$HfCache = "C:\Users\Administrator\.cache\huggingface\hub"

$Name = "phase_anchored_hf"
$Config = "$Root\configs\exp_phase_anchored_hf.json"
$RunDir = "$Root\exp\model_probe\phase_anchored_hf"
$LogDir = "$RunDir\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Write-Output "=== [$Name] TRAIN START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
$TrainLog = "$LogDir\train.log"
& $Py -u "$Root\src\run.py" --config $Config 2>&1 > $TrainLog
if ($LASTEXITCODE -ne 0) {
    Write-Output "=== [$Name] TRAIN FAILED exit=$LASTEXITCODE ==="
    exit 1
}
Write-Output "=== [$Name] TRAIN DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

# Find best checkpoint (last epoch)
$Ckpt = "$RunDir\epoch_0010.pt"
if (-not (Test-Path $Ckpt)) {
    $Ckpt = Get-ChildItem $RunDir -Filter "epoch_*.pt" | Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName
}
if (-not $Ckpt) {
    Write-Output "=== [$Name] NO CHECKPOINT FOUND ==="
    exit 1
}
Write-Output "=== [$Name] Using checkpoint: $Ckpt ==="

# Eval with AdaIN=2.0 protocol (eval_adain_20.json) — main-table standard
$EvalDir = "$RunDir\full_eval\adain20"
Write-Output "=== [$Name] EVAL START (AdaIN=2.0) $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
$EvalLog = "$LogDir\eval_adain20.log"
& $Py -u "$Root\src\utils\run_evaluation.py" `
    --config "$Root\configs\eval_adain_20.json" `
    --checkpoint $Ckpt `
    --output $EvalDir `
    --save_generated_images `
    --batch_size 2 `
    --clip_hf_cache_dir $HfCache 2>&1 > $EvalLog
if ($LASTEXITCODE -ne 0) {
    Write-Output "=== [$Name] EVAL FAILED exit=$LASTEXITCODE ==="
    exit 1
}
Write-Output "=== [$Name] EVAL DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

# DINO evaluation
Write-Output "=== [$Name] DINO START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
$DinoLog = "$LogDir\dino.log"
& $Py -u "$Root\_compute_dino.py" `
    --images_dir "$EvalDir\images" `
    --test_dir $TestDir `
    --dataset wikiart `
    --output "$EvalDir\dino.json" `
    --hf_cache $HfCache `
    --max_refs 30 2>&1 > $DinoLog
if ($LASTEXITCODE -ne 0) {
    Write-Output "=== [$Name] DINO FAILED exit=$LASTEXITCODE ==="
    exit 1
}
Write-Output "=== [$Name] DINO DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Write-Output "=== [$Name] COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
