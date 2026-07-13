# Round 11: StyleAdaIN sweep — batch training 3 experiments
$ErrorActionPreference = "Continue"
$base = "I:\Github\Latent_Style\SchrodingerBridge"
Set-Location $base

$exp_names = @(
    "brk_ae_adain_02",
    "brk_ae_adain_05",
    "brk_ae_adain_10"
)

foreach ($name in $exp_names) {
    $logFile = "$base\logs\${name}_train.log"
    Write-Output "[$(Get-Date)] Starting $name ..."
    python "$base\src\run.py" --config "$base\configs\exp_${name}.json" *>&1 | Tee-Object -FilePath $logFile
    $exitCode = $LASTEXITCODE
    Write-Output "[$(Get-Date)] $name complete. Exit code: $exitCode" | Out-File -Append -FilePath $logFile -Encoding utf8
    if ($exitCode -ne 0) {
        Write-Output "[$(Get-Date)] WARNING: $name failed with exit $exitCode, continuing to next..." | Out-File -Append -FilePath $logFile -Encoding utf8
    }
}

Write-Output "[$(Get-Date)] Round 11 batch complete."