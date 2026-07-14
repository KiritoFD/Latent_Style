# Stage11-13 pipeline: sequential training on remote RTX 3060
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$experiments = @(
    @{ name = "stage11_ll_wct";       config = "configs\exp_sty_stage11_ll_wct.json" },
    @{ name = "stage12_ll_adain_a10"; config = "configs\exp_sty_stage12_ll_a10.json" },
    @{ name = "stage13_cfg_ll_adain"; config = "configs\exp_sty_stage13_cfg_ll.json" }
)

foreach ($exp in $experiments) {
    $name = $exp.name
    $cfg = $exp.config
    $logFile = "C:\Users\Administrator\logs\${name}_train.out"

    Write-Output "=== START $name $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
    "=== START $name $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $logFile -Encoding utf8

    & python -u src\run.py --config $cfg *>&1 | Out-File $logFile -Append -Encoding utf8
    $exitCode = $LASTEXITCODE

    "=== DONE $name exit=$exitCode $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $logFile -Append -Encoding utf8
    Write-Output "=== DONE $name exit=$exitCode $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

    if ($exitCode -ne 0) {
        Write-Output "WARNING: $name failed with exit $exitCode, continuing to next experiment..."
    }
}

Write-Output "=== ALL PIPELINE DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
