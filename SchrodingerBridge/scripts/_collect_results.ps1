# Read all eval JSON results
$ErrorActionPreference = "Continue"

$REPO = "I:\Github\Latent_Style\SchrodingerBridge"
$files = @(
    "_eval_sdturbo_w20.json",
    "_eval_styleid_w20.json",
    "_eval_samst_w20.json",
    "_eval_cut_w20.json",
    "_eval_samam_w20.json",
    "_eval_sdturbo_256.json",
    "_eval_styleid_256.json",
    "_eval_adain_256_unified.json",
    "_eval_wct_256_unified.json",
    "_eval_samst_256_unified.json",
    "_eval_samam_256_unified.json",
    "_eval_identity_256_unified.json"
)

foreach ($f in $files) {
    $p = "$REPO\exp\$f"
    Write-Host "=== $f ==="
    if (Test-Path $p) {
        Get-Content $p -Raw
    } else {
        Write-Host "NOT FOUND"
    }
    Write-Host ""
}

Write-Host "=== Phase 2 SaMam W20 v2 progress ==="
$samamDir = "$REPO\exp\baseline_wikiarts20\samam\images"
if (Test-Path $samamDir) {
    $cnt = (Get-ChildItem $samamDir -File).Count
    Write-Host "samam_w20 images: $cnt / 750"
} else {
    Write-Host "no samam_w20 images dir"
}
$v2log = "$REPO\logs\samam_w20_v2.log"
if (Test-Path $v2log) {
    Write-Host "--- v2 log tail ---"
    Get-Content $v2log -Tail 20
}
