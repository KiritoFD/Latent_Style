# Pipeline: Round 3 structural re-parameterization probes
# Each: train 6ep from brk_a_ll03_10ep, evaluate with AdaIN 1.5, compute DINO
$ErrorActionPreference = "Continue"
$Root = "I:\Github\Latent_Style\SchrodingerBridge"
$Py = "C:\Program Files\Python312\python.exe"
Set-Location $Root
$env:PYTHONIOENCODING = "utf-8"

$TestDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$CacheDir = "$Root\exp\eval_cache"
$HfCache = "C:\Users\Administrator\.cache\huggingface\hub"

$Experiments = @(
    @{Name="isot_mrsc"; Config="$Root\configs\exp_probe_isot_mrsc_ft6.json"},
    @{Name="gated"; Config="$Root\configs\exp_probe_gated_ft6.json"},
    @{Name="dynamic_pw"; Config="$Root\configs\exp_probe_dynamic_pw_ft6.json"}
)

foreach ($Exp in $Experiments) {
    $Name = $Exp.Name
    $Config = $Exp.Config
    $RunDir = "$Root\exp\model_probe\target_hf_subband_${Name}_ft6"
    $LogDir = "$RunDir\logs"
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

    Write-Output "=== [$Name] TRAIN START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
    $TrainLog = "$LogDir\train.log"
    & $Py -u "$Root\src\run.py" --config $Config 2>&1 > $TrainLog
    if ($LASTEXITCODE -ne 0) {
        Write-Output "=== [$Name] TRAIN FAILED exit=$LASTEXITCODE ==="
        continue
    }
    Write-Output "=== [$Name] TRAIN DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

    $Ckpt = "$RunDir\epoch_0006.pt"
    if (-not (Test-Path $Ckpt)) {
        $Ckpt = Get-ChildItem $RunDir -Filter "epoch_*.pt" | Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName
    }
    if (-not $Ckpt) {
        Write-Output "=== [$Name] NO CHECKPOINT FOUND ==="
        continue
    }
    Write-Output "=== [$Name] Using checkpoint: $Ckpt ==="

    $EvalDir = "$RunDir\full_eval\adain15"
    Write-Output "=== [$Name] EVAL START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
    $EvalLog = "$LogDir\eval_adain15.log"
    & $Py -u "$Root\src\utils\run_evaluation.py" `
        --config "$Root\configs\eval_adain_15.json" `
        --checkpoint $Ckpt `
        --output $EvalDir `
        --save_generated_images `
        --batch_size 2 `
        --clip_hf_cache_dir $HfCache 2>&1 > $EvalLog
    if ($LASTEXITCODE -ne 0) {
        Write-Output "=== [$Name] EVAL FAILED exit=$LASTEXITCODE ==="
        continue
    }
    Write-Output "=== [$Name] EVAL DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

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
        continue
    }
    Write-Output "=== [$Name] DINO DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
    Write-Output "=== [$Name] COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
}

Write-Output "=== ALL EXPERIMENTS COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
