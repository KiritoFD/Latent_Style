param(
    [string]$Root = "I:\Github\Latent_Style\SchrodingerBridge",
    [string]$ConfigRoot = "exp\remote_swd_ablation",
    [switch]$Force,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$rootPath = (Resolve-Path -LiteralPath $Root).Path
$configPath = Join-Path $rootPath $ConfigRoot
$logDir = Join-Path $rootPath "logs\remote_swd_ablation"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$env:PYTHONPATH = "$rootPath\src;$env:PYTHONPATH"
Set-Location -LiteralPath $rootPath

$configs = Get-ChildItem -LiteralPath $configPath -Recurse -Filter config.json |
    Sort-Object FullName

if ($configs.Count -eq 0) {
    throw "No config.json files found under $configPath"
}

if ($DryRun) {
    "DRY_RUN root=$rootPath configPath=$configPath count=$($configs.Count)"
    foreach ($cfg in $configs) {
        $name = Split-Path -Leaf (Split-Path -Parent $cfg.FullName)
        "CONFIG $name $($cfg.FullName)"
    }
    exit 0
}

foreach ($cfg in $configs) {
    $name = Split-Path -Leaf (Split-Path -Parent $cfg.FullName)
    $runLog = Join-Path $logDir "$name.train.log"
    $statusFile = Join-Path $logDir "$name.status.txt"
    $summary = Join-Path $rootPath "exp\remote_swd_ablation\$name\full_eval\epoch_0005\summary.json"

    if ((Test-Path -LiteralPath $summary) -and -not $Force) {
        "SKIP $name existing summary=$summary" | Tee-Object -FilePath $statusFile
        continue
    }

    "START $name $(Get-Date -Format o)" | Tee-Object -FilePath $statusFile
    $cmdLine = "python -u `"$rootPath\src\run.py`" --config `"$($cfg.FullName)`" 1> `"$runLog`" 2>&1"
    & cmd.exe /c $cmdLine
    $code = $LASTEXITCODE
    "END $name exit=$code $(Get-Date -Format o)" | Tee-Object -FilePath $statusFile -Append
    if ($code -ne 0) {
        throw "Run failed: $name exit=$code log=$runLog"
    }
}

"ALL_DONE $(Get-Date -Format o)" | Tee-Object -FilePath (Join-Path $logDir "launcher.status.txt")
