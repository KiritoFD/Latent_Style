# DINO-S break round1 pipeline: sequential training on remote RTX 3060
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$experiments = @(
    @{ name = "brk_d_wll10_10ep";       config = "configs\exp_brk_d_wll10_10ep.json" },
    @{ name = "brk_e_wll30_10ep";       config = "configs\exp_brk_e_wll30_10ep.json" },
    @{ name = "brk_b_baseline_10ep";    config = "configs\exp_brk_b_baseline_10ep.json" },
    @{ name = "brk_a_ll03_10ep";        config = "configs\exp_brk_a_ll03_10ep.json" }
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
