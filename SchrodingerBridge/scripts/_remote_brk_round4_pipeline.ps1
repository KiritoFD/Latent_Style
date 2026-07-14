# Round 4 pipeline: 5 experiments to make main table more prominent
# All on top of brk_a (a=0.3, 10ep, SAT, seed=42)
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$experiments = @(
    @{ name = "brk_l_w_lh_hl_15";     config = "configs\exp_brk_l_w_lh_hl_15.json" },
    @{ name = "brk_m_adain15";         config = "configs\exp_brk_m_adain15.json" },
    @{ name = "brk_n_two_stage";       config = "configs\exp_brk_n_two_stage.json" },
    @{ name = "brk_o_endpoint_high_03"; config = "configs\exp_brk_o_endpoint_high_03.json" },
    @{ name = "brk_p_lr_0003";         config = "configs\exp_brk_p_lr_0003.json" }
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

Write-Output "=== ROUND4 ALL PIPELINE DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
