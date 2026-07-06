# Check Random5 (5-style) data completeness for all methods
$ErrorActionPreference = "Continue"
$REPO = "I:\Github\Latent_Style\SchrodingerBridge"

Write-Host "=== Random5 image counts (target: 750 = 5x5x30) ==="
$base = "$REPO\exp\baseline_wikiarts20"
$methods = @("adain", "wct", "identity", "sdturbo", "styleid", "samst", "samam", "cut")
foreach ($m in $methods) {
    $dir = "$base\$m\images"
    if (Test-Path $dir) {
        $cnt = (Get-ChildItem $dir -File).Count
        $status = if ($cnt -ge 750) { "OK" } else { "INCOMPLETE" }
        Write-Host "  ${m}: $cnt / 750  [$status]"
    } else {
        Write-Host "  ${m}: NOT FOUND"
    }
}

Write-Host ""
Write-Host "=== Existing eval JSON results ==="
Get-ChildItem "$REPO\exp" -Filter "_eval_*.json" -ErrorAction SilentlyContinue |
    ForEach-Object {
        $r = Get-Content $_.FullName -Raw | ConvertFrom-Json
        $musiq = if ($r.musiq) { $r.musiq } else { "N/A" }
        Write-Host ("  {0}: CLIP-S={1} LPIPS={2} MUSIQ={3}" -f $_.Name, $r.clip_s, $r.lpips, $musiq)
    }

Write-Host ""
Write-Host "=== Check which methods need generation ==="
$needGen = @()
foreach ($m in @("sdturbo", "styleid", "samst", "samam", "cut")) {
    $dir = "$base\$m\images"
    $cnt = if (Test-Path $dir) { (Get-ChildItem $dir -File).Count } else { 0 }
    if ($cnt -lt 750) {
        $needGen += "$m ($cnt/750)"
    }
}
if ($needGen.Count -gt 0) {
    Write-Host "  Need generation: $($needGen -join ', ')"
} else {
    Write-Host "  All methods have 750 images"
}
