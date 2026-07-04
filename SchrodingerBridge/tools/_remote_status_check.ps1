# Remote status check - kill GPU processes, verify sync, check dataset
$ErrorActionPreference = "SilentlyContinue"

Write-Host "=== 1. GPU STATUS BEFORE ==="
nvidia-smi
Write-Host ""

Write-Host "=== 2. KILLING PYTHON PROCESSES ==="
$pyProcs = Get-Process python -ErrorAction SilentlyContinue
if ($pyProcs) {
    $pyProcs | ForEach-Object { Write-Host "Killing PID $($_.Id) ($($_.WorkingSet64/1MB) MB)"; Stop-Process -Id $_.Id -Force }
    Start-Sleep -Seconds 2
} else {
    Write-Host "No python processes running"
}
Write-Host ""

Write-Host "=== 3. GPU STATUS AFTER ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
Write-Host ""

Write-Host "=== 4. SRC FILES CHECK ==="
$srcFiles = @(
    "spectral_losses620.py",
    "config_schema.py",
    "trainer.py",
    "run.py",
    "spectral_bridge620.py",
    "style_encoder620.py",
    "blocks620.py"
)
foreach ($f in $srcFiles) {
    $path = "I:\Github\Latent_Style\SchrodingerBridge\src\$f"
    if (Test-Path $path) {
        $info = Get-Item $path
        Write-Host "OK  $f  size=$($info.Length)  modified=$($info.LastWriteTime.ToString('yyyy-MM-dd HH:mm'))"
    } else {
        Write-Host "MISSING  $f"
    }
}
Write-Host ""

Write-Host "=== 5. CONFIGS CHECK ==="
$cfgDir = "I:\Github\Latent_Style\SchrodingerBridge\configs"
if (Test-Path $cfgDir) {
    Get-ChildItem $cfgDir -Filter "630_*" | ForEach-Object { Write-Host "OK  $($_.Name)  size=$($_.Length)" }
} else {
    Write-Host "MISSING configs dir"
}
Write-Host ""

Write-Host "=== 6. STATE FILES CHECK ==="
$stateDir = "I:\Github\Latent_Style\SchrodingerBridge\docs\630\state"
if (Test-Path $stateDir) {
    Get-ChildItem $stateDir | ForEach-Object { Write-Host "OK  $($_.Name)  size=$($_.Length)" }
} else {
    Write-Host "MISSING state dir"
}
Write-Host ""

Write-Host "=== 7. DATASET CHECK ==="
$dsPath = "I:\wikiart_distinct5_samam_512_classview"
if (Test-Path $dsPath) {
    Write-Host "Dataset root EXISTS"
    Get-ChildItem $dsPath | ForEach-Object { Write-Host "  $($_.Name)  [$($_.Mode)]" }
} else {
    Write-Host "MISSING dataset: $dsPath"
}
Write-Host ""

Write-Host "=== 8. EXP DIR CHECK ==="
$expDir = "I:\Github\Latent_Style\SchrodingerBridge\exp"
if (Test-Path $expDir) {
    Get-ChildItem $expDir | ForEach-Object { Write-Host "  $($_.Name)" }
} else {
    Write-Host "MISSING exp dir"
}
Write-Host ""

Write-Host "=== 9. BASELINE EVAL CHECK ==="
$baselinePath = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\eval\samam\summary.json"
if (Test-Path $baselinePath) {
    Write-Host "SaMam baseline EXISTS"
    Get-Content $baselinePath
} else {
    Write-Host "MISSING baseline summary"
}
Write-Host ""

Write-Host "=== 10. PYTHON ENV ==="
python --version
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
Write-Host ""
Write-Host "=== DONE ==="
