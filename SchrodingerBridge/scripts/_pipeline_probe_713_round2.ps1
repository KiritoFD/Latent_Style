# Pipeline: Run all 4 probe experiments (TASM, LDB, ISST, MRSC)
# Each: train 6ep from brk_a_ll03_10ep, evaluate with AdaIN 1.5, compute DINO
$ErrorActionPreference = "Continue"
$Root = "I:\Github\Latent_Style\SchrodingerBridge"
$Py = "C:\Users\Administrator\miniconda3\envs\torch\python.exe"
Set-Location $Root
$env:PYTHONPATH = "$Root\src"
$env:PYTHONIOENCODING = "utf-8"

$TestDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$CacheDir = "$Root\exp\eval_cache"
$HfCache = "C:\Users\Administrator\.cache\huggingface\hub"

$Experiments = @(
    @{Name="tasm"; Config="configs\exp_probe_tasm_ft6.json"},
    @{Name="ldb"; Config="configs\exp_probe_ldb_ft6.json"},
    @{Name="isot"; Config="configs\exp_probe_isot_ft6.json"},
    @{Name="mrsc"; Config="configs\exp_probe_mrsc_ft6.json"}
)

foreach ($Exp in $Experiments) {
    $Name = $Exp.Name
    $Config = $Exp.Config
    $RunDir = "$Root\exp\model_probe\target_hf_subband_${Name}_ft6"
    $LogDir = "$RunDir\logs"
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

    Write-Output "=== [$Name] TRAIN START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
    $TrainLog = "$LogDir\train.log"
    & $Py -u src\run.py --config $Config 2>&1 > $TrainLog
    if ($LASTEXITCODE -ne 0) {
        Write-Output "=== [$Name] TRAIN FAILED exit=$LASTEXITCODE ==="
        continue
    }
    Write-Output "=== [$Name] TRAIN DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

    # Find best checkpoint (epoch_0006)
    $Ckpt = "$RunDir\epoch_0006.pt"
    if (-not (Test-Path $Ckpt)) {
        $Ckpt = Get-ChildItem $RunDir -Filter "epoch_*.pt" | Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName
    }
    if (-not $Ckpt) {
        Write-Output "=== [$Name] NO CHECKPOINT FOUND ==="
        continue
    }
    Write-Output "=== [$Name] Using checkpoint: $Ckpt ==="

    # Eval AdaIN 1.5
    $EvalDir = "$RunDir\full_eval\adain15"
    Write-Output "=== [$Name] EVAL START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
    $EvalLog = "$LogDir\eval_adain15.log"
    & $Py -u "$Root\src\utils\run_evaluation.py" `
        --config "$Root\configs\eval_adain_15.json" `
        --checkpoint $Ckpt `
        --output $EvalDir `
        --save_generated_images `
        --batch_size 2 2>&1 > $EvalLog
    if ($LASTEXITCODE -ne 0) {
        Write-Output "=== [$Name] EVAL FAILED exit=$LASTEXITCODE ==="
        continue
    }
    Write-Output "=== [$Name] EVAL DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

    # DINO
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

    # Extract results
    Write-Output "=== [$Name] RESULTS ==="
    & $Py -c "import json; d=json.load(open('$EvalDir\dino.json')); print(f'DINO-C: {d[\"dino_content\"][\"mean\"]:.4f}'); print(f'DINO-S: {d[\"dino_style\"][\"mean\"]:.4f}')"
    & $Py -c "import json; d=json.load(open('$EvalDir\summary.json')); s=d['analysis']['all_pairs_overview']; print(f'CLIP-S: {s[\"mean_clip_style\"]:.4f}'); print(f'LPIPS: {s[\"mean_lpips\"]:.4f}')"
    Write-Output "=== [$Name] COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
}

Write-Output "=== ALL EXPERIMENTS COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="