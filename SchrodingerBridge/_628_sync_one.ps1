$src = '/tmp/_628_start_extreme_batch.ps1'
$dst = 'I:\Github\Latent_Style\SchrodingerBridge\_628_start_extreme_batch.ps1'
Copy-Item $src $dst -Force
if (Test-Path $dst) {
    $info = Get-Item $dst
    Write-Host "OK: copied to $dst ($($info.Length) bytes)"
} else {
    Write-Host "FAILED: $dst does not exist after copy"
}
