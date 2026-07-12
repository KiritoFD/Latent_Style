# Round 7 pipeline: 2 alpha augmentation experiments
# brk_aa_aug24 (alpha~U(0.2,0.4)), brk_aa_aug35 (alpha~U(0.3,0.5))
# Run sequentially on remote RTX 3060 12GB
$ErrorActionPreference = "Continue"
$base = "I:\Github\Latent_Style\SchrodingerBridge"
$exp_names = @("brk_aa_aug24", "brk_aa_aug35")

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
Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ALL ROUND 7 EXPERIMENTS COMPLETE"
Write-Host "=========================================="
