$logBase = "I:\Github\Latent_Style\SchrodingerBridge\logs\baseline_wikiarts20.log"
foreach ($ext in @("identity.gen.err", "identity.gen.out", "adain.gen.err", "adain.gen.out")) {
    $f = "$logBase.$ext"
    Write-Output "=== $ext ==="
    if (Test-Path $f) {
        $sz = (Get-Item $f).Length
        Write-Output "  size: $sz bytes"
        Get-Content $f -Tail 25
    } else {
        Write-Output "  (not found)"
    }
    Write-Output ""
}
