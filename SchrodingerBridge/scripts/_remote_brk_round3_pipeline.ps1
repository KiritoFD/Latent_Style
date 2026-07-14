# DINO-S break round3: milder α, combined mechanism, seed variance
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$experiments = @(
    @{ name = "brk_i_ll02_10ep";        config = "configs\exp_brk_i_ll02_10ep.json" },
    @{ name = "brk_j_ll03_zwct_10ep";   config = "configs\exp_brk_j_ll03_zwct_10ep.json" },
    @{ name = "brk_k_ll03_seed7";       config = "configs\exp_brk_k_ll03_seed7.json" }
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
        Write-Output "WARNING: $name failed, continuing..."
    }
}

Write-Output "=== ALL ROUND3 DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
