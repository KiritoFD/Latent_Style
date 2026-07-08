$ErrorActionPreference = 'Continue'

$evals = @(
    "_eval_sdturbo_256.json",
    "_eval_samst_256_unified.json",
    "_eval_samam_256_unified.json",
    "_eval_styleid_256.json",
    "_eval_identity_256_unified.json",
    "_eval_adain_256_unified.json",
    "_eval_wct_256_unified.json"
)

foreach ($e in $evals) {
    $path = "I:\Github\Latent_Style\SchrodingerBridge\exp\$e"
    Write-Host "=== $e ==="
    $ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "type `"$path`""
    Write-Host $ssh_out
    Write-Host ""
}
