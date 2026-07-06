# Check D5 MUSIQ JSON details
$ErrorActionPreference = "Continue"
$REPO = "I:\Github\Latent_Style\SchrodingerBridge"

$files = @(
    "_eval_adain_d5_musiq.json",
    "_eval_wct_d5_musiq.json",
    "_eval_sdturbo_d5_musiq.json",
    "_eval_styleid_d5_musiq.json",
    "_eval_cut_d5_musiq.json",
    "_eval_samst_d5_musiq.json",
    "_eval_samam_d5_musiq.json",
    "_eval_identity_d5_musiq.json"
)

foreach ($f in $files) {
    $p = "$REPO\exp\$f"
    if (Test-Path $p) {
        $j = Get-Content $p -Raw | ConvertFrom-Json
        Write-Host ("{0}: MUSIQ={1}  N={2}" -f $f, $j.musiq, $j.n_images)
    } else {
        Write-Host "${f}: NOT FOUND"
    }
}
