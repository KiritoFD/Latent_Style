param(
    [switch]$EvalOnly,
    [switch]$AllowNetwork
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = (Get-Command python).Source
$Config = Join-Path $Root "src\default_config.json"
$InferenceConfig = Join-Path $Root "configs\eval_adain_20.json"
$RunDir = Join-Path $Root "runs\submission\repro_brk_a_15ep"
$LogDir = Join-Path $RunDir "logs"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Location $Root
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

if (-not $EvalOnly) {
    $TrainLog = Join-Path $LogDir "train.log"
    & $Python -u "src\run.py" --config $Config 2>&1 | Tee-Object -FilePath $TrainLog
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed with exit code $LASTEXITCODE. See $TrainLog"
    }
}

$EvalLog = Join-Path $LogDir "paper_eval_adain20.log"
$EvalArgs = @(
    "scripts\batch_eval_all.py",
    "--checkpoint_dir", $RunDir,
    "--config", $Config,
    "--config_override", $InferenceConfig,
    "--output_subdir", "paper_eval_adain20"
)
if ($AllowNetwork) {
    $EvalArgs += "--allow_network"
}

& $Python -u @EvalArgs 2>&1 | Tee-Object -FilePath $EvalLog
if ($LASTEXITCODE -ne 0) {
    throw "Evaluation failed with exit code $LASTEXITCODE. See $EvalLog"
}
