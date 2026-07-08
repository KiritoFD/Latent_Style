$ErrorActionPreference = 'Continue'

Write-Host "=== Check P256 identity_256/images contents ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\exp_256_photo2art\identity_256\images 2>&1 | findstr /R \"^[a-zA-Z0-9]\" | head -5"
Write-Host $ssh_out

Write-Host ""
Write-Host "=== Count files in each P256 method images dir ==="
foreach ($m in @("identity_256", "adain_256", "wct_256", "samst_256", "samam_256", "sdturbo_256", "styleid_256")) {
    $ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\exp_256_photo2art\$m\images\*.png 2>nul | find /C /V """
    Write-Host "${m}: $ssh_out"
}

Write-Host ""
Write-Host "=== Check what extension the identity_256 images have ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\exp_256_photo2art\identity_256\images 2>&1"
Write-Host $ssh_out
