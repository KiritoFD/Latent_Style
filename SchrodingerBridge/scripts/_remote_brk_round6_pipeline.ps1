# Round 6 pipeline: 2 structural pivot experiments
# brk_y_ml_dwt (multi-level DWT), brk_z_ll_wct (LL WCT)
# Run sequentially on remote RTX 3060 12GB
$ErrorActionPreference = "Continue"
$base = "I:\Github\Latent_Style\SchrodingerBridge"
$exp_names = @("brk_y_ml_dwt", "brk_z_ll_wct")

# Ensure logs directory exists
if (-not (Test-Path "$base\logs")) {
    New-Item -ItemType Directory -Path "$base\logs" -Force | Out-Null
}

foreach ($exp in $exp_names) {
    Write-Host "=========================================="
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START $exp"
    Write-Host "=========================================="

    $config = "$base\configs\exp_$exp.json"
    if (-not (Test-Path $config)) {
        Write-Host "ERROR: config not found: $config"
        continue
    }

    # Run training
    python "$base\src\run.py" --config "$config" 2>&1 | Tee-Object -FilePath "$base\logs\${exp}_train.log"

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: $exp training failed with exit code $LASTEXITCODE"
        continue
    }

    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] DONE $exp"
}

Write-Host "=========================================="
Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ALL ROUND 6 EXPERIMENTS COMPLETE"
Write-Host "=========================================="
