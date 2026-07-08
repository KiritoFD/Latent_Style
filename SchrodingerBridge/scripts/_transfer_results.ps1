$ErrorActionPreference = 'Continue'

$local_root = "g:\GitHub\Latent_Style\SchrodingerBridge\results"
$remote_host = "administrator@100.115.18.62"
$port = "2222"

# Complete transfer list: all methods x all datasets
$transfers = @()

# === D5-512 (9 methods, 750 png each) ===
$transfers += @{ ds="D5-512"; method="identity"; remote="I:/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/images/identity" }
$transfers += @{ ds="D5-512"; method="adain";    remote="I:/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/images/adain" }
$transfers += @{ ds="D5-512"; method="wct";      remote="I:/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/images/wct_v32k" }
$transfers += @{ ds="D5-512"; method="sdturbo";  remote="I:/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/images/sdturbo" }
$transfers += @{ ds="D5-512"; method="cut";      remote="I:/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/images/cut" }
$transfers += @{ ds="D5-512"; method="samst";    remote="I:/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/images/samst" }
$transfers += @{ ds="D5-512"; method="samam";    remote="I:/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/images/samam" }
$transfers += @{ ds="D5-512"; method="styleid";  remote="I:/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/images/styleid" }
$transfers += @{ ds="D5-512"; method="seedream"; remote="I:/Github/Latent_Style/exp_baselines/seedream45_api/distinct5_512_seedream45_windhub_20260607_repaired750/images" }
$transfers += @{ ds="D5-512"; method="weave";    remote="I:/Github/Latent_Style/SchrodingerBridge/exp/clean_base_v2/full_eval/epoch_0010/images" }

# === P256 / Photo2Art-256 (7 methods; CUT/Seedream images not retained on remote) ===
$transfers += @{ ds="P256"; method="identity"; remote="I:/exp_256_photo2art/identity_256/images" }
$transfers += @{ ds="P256"; method="adain";    remote="I:/exp_256_photo2art/adain_256/images" }
$transfers += @{ ds="P256"; method="wct";      remote="I:/exp_256_photo2art/wct_256/images" }
$transfers += @{ ds="P256"; method="sdturbo";  remote="I:/exp_256_photo2art/sdturbo_256/images" }
$transfers += @{ ds="P256"; method="samst";    remote="I:/exp_256_photo2art/samst_256/images" }
$transfers += @{ ds="P256"; method="samam";    remote="I:/exp_256_photo2art/samam_256/images" }
$transfers += @{ ds="P256"; method="styleid";  remote="I:/exp_256_photo2art/styleid_256/images" }
$transfers += @{ ds="P256"; method="weave";    remote="I:/exp_our_models_eval/latent256_e10/images" }

# === R5-WikiArt (7 methods + WEAVE 12000 png) ===
$transfers += @{ ds="R5-WikiArt"; method="identity"; remote="I:/Github/Latent_Style/SchrodingerBridge/exp/baseline_wikiarts20/identity/images" }
$transfers += @{ ds="R5-WikiArt"; method="adain";    remote="I:/Github/Latent_Style/SchrodingerBridge/exp/baseline_wikiarts20/adain/images" }
$transfers += @{ ds="R5-WikiArt"; method="wct";      remote="I:/Github/Latent_Style/SchrodingerBridge/exp/baseline_wikiarts20/wct/images" }
$transfers += @{ ds="R5-WikiArt"; method="sdturbo";  remote="I:/Github/Latent_Style/SchrodingerBridge/exp/baseline_wikiarts20/sdturbo/images" }
$transfers += @{ ds="R5-WikiArt"; method="samst";    remote="I:/Github/Latent_Style/SchrodingerBridge/exp/baseline_wikiarts20/samst/images" }
$transfers += @{ ds="R5-WikiArt"; method="samam";    remote="I:/Github/Latent_Style/SchrodingerBridge/exp/baseline_wikiarts20/samam/images" }
$transfers += @{ ds="R5-WikiArt"; method="styleid";  remote="I:/Github/Latent_Style/SchrodingerBridge/exp/baseline_wikiarts20/styleid/images" }
$transfers += @{ ds="R5-WikiArt"; method="weave";    remote="I:/Github/Latent_Style/SchrodingerBridge/exp/wikiarts20_eval/images" }

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
    
    $existing = (Get-ChildItem $local -Filter *.png -ErrorAction SilentlyContinue).Count
    if ($existing -gt 0) {
        Write-Host "[$idx/$total] SKIP $ds/$method - already has $existing images"
        $success++
        continue
    }
    
    Write-Host "[$idx/$total] Transfering $ds/$method ..."
    $t0 = Get-Date
    
    # Use scp with explicit *.png glob (avoids nested dir issue from scp -r)
    & scp -P $port -o LogLevel=ERROR "${remote_host}:${remote}/*.png" "$local/"
    $exit = $LASTEXITCODE
    
    $elapsed = ((Get-Date) - $t0).TotalSeconds
    $cnt = (Get-ChildItem $local -Filter *.png -ErrorAction SilentlyContinue).Count
    
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
Write-Host "Transfer Summary"
Write-Host "============================================================"
Write-Host "  Total: $total, Success: $success, Failed: $($failed.Count)"
if ($failed.Count -gt 0) {
    Write-Host "  Failed: $($failed -join ', ')"
}

Write-Host ""
Write-Host "=== Final image counts ==="
foreach ($ds in @("D5-512", "P256", "R5-WikiArt")) {
    Write-Host "[$ds]"
    $ds_dir = Join-Path $local_root $ds
    if (Test-Path $ds_dir) {
        Get-ChildItem $ds_dir -Directory | ForEach-Object {
            $cnt = (Get-ChildItem $_.FullName -Filter *.png -ErrorAction SilentlyContinue).Count
            Write-Host ("  " + $_.Name + ": " + $cnt + " png")
        }
    }
}
