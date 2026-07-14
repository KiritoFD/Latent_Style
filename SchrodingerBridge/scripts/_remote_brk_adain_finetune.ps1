# Adain finetune: brk_s (1.6) + brk_t (1.7)
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$experiments = @(
    @{ name = "brk_s_adain16";  config = "configs\exp_brk_s_adain16.json" },
    @{ name = "brk_t_adain17";  config = "configs\exp_brk_t_adain17.json" }
)

foreach ($exp in $experiments) {
    $name = $exp.name
    $cfg = $exp.config
    $logFile = "C:\Users\Administrator\logs\${name}_train.out"

    Write-Output "=== START $name $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
    "=== START $name $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $logFile -Encoding utf8

    & python -u run.py --config $cfg *>&1 | Out-File $logFile -Append -Encoding utf8
    $exitCode = $LASTEXITCODE

    "=== DONE $name exit=$exitCode $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $logFile -Append -Encoding utf8
    Write-Output "=== DONE $name exit=$exitCode $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

    if ($exitCode -ne 0) {
        Write-Output "WARNING: $name failed with exit $exitCode, continuing to next experiment..."
    }
}

Write-Output "=== ADAIN FINETUNE DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
