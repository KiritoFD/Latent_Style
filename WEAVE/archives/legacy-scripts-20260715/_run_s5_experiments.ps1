$env:PYTHONIOENCODING = "utf-8"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"

# Run s5_cov07 first, then s5_cov05 — continue on failure
$experiments = @(
    @{ name = "s5_cov07"; config = "I:\Github\Latent_Style\SchrodingerBridge\configs\_s5_overrides\s5_cov07_wct_ll05.json" },
    @{ name = "s5_cov05"; config = "I:\Github\Latent_Style\SchrodingerBridge\configs\_s5_overrides\s5_cov05_wct_ll05.json" }
)

foreach ($exp in $experiments) {
    $name = $exp.name
    $config = $exp.config
    $logFile = "I:\Github\Latent_Style\SchrodingerBridge\logs\$name.log"

    Write-Host "========== START $name =========="
    $expStart = Get-Date

    try {
        & powershell -ExecutionPolicy Bypass -File "I:\Github\Latent_Style\SchrodingerBridge\scripts\_run_s1_single.ps1" `
            -RunName $name `
            -OverrideConfig $config *>&1 | Tee-Object -FilePath $logFile
    } catch {
        Write-Host "ERROR in $name : $_"
    }

    $expMin = [math]::Round(((Get-Date) - $expStart).TotalMinutes, 1)
    Write-Host "========== DONE $name (${expMin}min) =========="
}

Write-Host "ALL_S5_EXPERIMENTS_DONE"
