$ErrorActionPreference = 'Continue'

# Second pass: transfer P256 methods that use .jpg instead of .png
$local_root = "g:\GitHub\Latent_Style\SchrodingerBridge\results"
$remote_host = "administrator@100.115.18.62"
$port = "2222"

# P256 methods with .jpg files (identity, adain, wct, samst, samam)
$transfers = @()
$transfers += @{ ds="P256"; method="identity"; remote="I:/exp_256_photo2art/identity_256/images" }
$transfers += @{ ds="P256"; method="adain";    remote="I:/exp_256_photo2art/adain_256/images" }
$transfers += @{ ds="P256"; method="wct";      remote="I:/exp_256_photo2art/wct_256/images" }
$transfers += @{ ds="P256"; method="samst";    remote="I:/exp_256_photo2art/samst_256/images" }
$transfers += @{ ds="P256"; method="samam";    remote="I:/exp_256_photo2art/samam_256/images" }

$total = $transfers.Count
$idx = 0
$success = 0
$failed = @()

foreach ($t in $transfers) {
    $idx++
    $ds = $t.ds
    $method = $t.method
    $remote = $t.remote
    $local = Join-Path $local_root "$ds\$method"
    
    if (-not (Test-Path $local)) {
        New-Item -ItemType Directory -Path $local -Force | Out-Null
    }
    
    # Check if local already has any images (jpg or png)
    $existing = (Get-ChildItem $local -File -ErrorAction SilentlyContinue).Count
    if ($existing -gt 0) {
        Write-Host "[$idx/$total] SKIP $ds/$method - already has $existing files"
        $success++
        continue
    }
    
    Write-Host "[$idx/$total] Transfering $ds/$method (.jpg) ..."
    $t0 = Get-Date
    
    # Try .jpg first
    & scp -P $port -o LogLevel=ERROR "${remote_host}:${remote}/*.jpg" "$local/"
    $exit = $LASTEXITCODE
    
    $elapsed = ((Get-Date) - $t0).TotalSeconds
    $cnt = (Get-ChildItem $local -File -ErrorAction SilentlyContinue).Count
    
    if ($exit -eq 0 -and $cnt -gt 0) {
        Write-Host "  OK: $cnt images in $([math]::Round($elapsed, 1))s"
        $success++
    } else {
        Write-Host "  FAILED (exit=$exit, images=$cnt) remote=$remote"
        $failed += "$ds/$method"
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host "P256 .jpg Second-Pass Summary"
Write-Host "============================================================"
Write-Host "  Total: $total, Success: $success, Failed: $($failed.Count)"
if ($failed.Count -gt 0) {
    Write-Host "  Failed: $($failed -join ', ')"
}
