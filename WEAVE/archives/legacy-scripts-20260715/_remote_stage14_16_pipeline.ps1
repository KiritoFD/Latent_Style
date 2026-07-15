# Stage14-16 pipeline: sequential training on remote RTX 3060
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$experiments = @(
    @{ name = "stage14_huber_hh";          config = "configs\exp_sty_stage14_huber_hh.json" },
    @{ name = "stage15_huber_hh_hfwct_b05"; config = "configs\exp_sty_stage15_huber_hh_hfwct.json" },
    @{ name = "stage16_huber_hh_hfwct_b03"; config = "configs\exp_sty_stage16_huber_hh_hfwct_b03.json" }
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
