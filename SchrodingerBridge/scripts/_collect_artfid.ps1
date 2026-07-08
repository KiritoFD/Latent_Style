$ErrorActionPreference = 'Continue'

$exp_root = "I:\Github\Latent_Style\SchrodingerBridge\exp"

Write-Host "============================================================"
Write-Host "Collect ArtFID from all baseline summary.json (D5)"
Write-Host "============================================================"

# D5 baselines in baseline_v2/images/* - check for summary.json in parent
$bv2 = "$exp_root\baseline_v2"
$methods = @("identity","adain","wct_v32k","wct_v32k_a0.60","sdturbo","styleid","samam","samst","cut","sdedit_str0.10","sdedit_str0.20","sdedit_str0.35","sdedit_str0.40")

# Check baseline_v2/eval/<method>/summary.json
foreach ($m in $methods) {
    $sj = "$bv2\eval\$m\summary.json"
    if (Test-Path $sj) {
        $j = Get-Content $sj -Raw | ConvertFrom-Json
        $artfid = $null
        $clip_s = $null
        $lpips = $null
        if ($j.all_pairs) {
            if ($j.all_pairs.mean_art_fid) { $artfid = $j.all_pairs.mean_art_fid }
            if ($j.all_pairs.mean_clip_style) { $clip_s = $j.all_pairs.mean_clip_style }
            if ($j.all_pairs.mean_content_lpips) { $lpips = $j.all_pairs.mean_content_lpips }
        }
        if ($j.transfer_only) {
            $transfer_artfid = $j.transfer_only.mean_art_fid
        }
        $img_cnt = 0
        $img_dir = "$bv2\images\$m"
        if (Test-Path $img_dir) { $img_cnt = (Get-ChildItem $img_dir -Filter *.png -ErrorAction SilentlyContinue).Count }
        Write-Host ("  ${m}: artfid_all=${artfid} artfid_transfer=${transfer_artfid} clip_s=${clip_s} lpips=${lpips} imgs=${img_cnt}")
    }
}

# Check baseline_reeval
Write-Host "`n=== baseline_reeval summaries ==="
$br = "$exp_root\baseline_reeval"
if (Test-Path $br) {
    Get-ChildItem $br -Directory | ForEach-Object {
        $sj = Join-Path $_.FullName "summary.json"
        if (Test-Path $sj) {
            $j = Get-Content $sj -Raw | ConvertFrom-Json
            $artfid = $null
            if ($j.all_pairs -and $j.all_pairs.mean_art_fid) { $artfid = $j.all_pairs.mean_art_fid }
            if ($j.transfer_only -and $j.transfer_only.mean_art_fid) { $t_artfid = $j.transfer_only.mean_art_fid }
            $img_cnt = 0
            $img_dir = Join-Path $_.FullName "images"
            if (Test-Path $img_dir) { $img_cnt = (Get-ChildItem $img_dir -Filter *.png -ErrorAction SilentlyContinue).Count }
            Write-Host ("  " + $_.Name + ": artfid_all=${artfid} artfid_transfer=${t_artfid} imgs=${img_cnt}")
        }
    }
}

# Check WEAVE D5 (clean_base_v2)
Write-Host "`n=== WEAVE D5 (clean_base_v2) ==="
$weave_d5 = "$exp_root\clean_base_v2\full_eval\epoch_0010\summary.json"
if (Test-Path $weave_d5) {
    $j = Get-Content $weave_d5 -Raw | ConvertFrom-Json
    if ($j.all_pairs) {
        Write-Host ("  artfid_all: " + $j.all_pairs.mean_art_fid)
        Write-Host ("  clip_s: " + $j.all_pairs.mean_clip_style)
        Write-Host ("  lpips: " + $j.all_pairs.mean_content_lpips)
    }
    if ($j.transfer_only) {
        Write-Host ("  artfid_transfer: " + $j.transfer_only.mean_art_fid)
    }
} else {
    Write-Host "  summary.json NOT FOUND at $weave_d5"
    # Check parent
    $parent = Split-Path $weave_d5 -Parent
    if (Test-Path $parent) {
        Get-ChildItem $parent | ForEach-Object { Write-Host ("    " + $_.Name) }
    }
}

# Check WEAVE W20 (wikiarts20_eval)
Write-Host "`n=== WEAVE W20 (wikiarts20_eval) ==="
$weave_w20 = "$exp_root\wikiarts20_eval\summary.json"
if (Test-Path $weave_w20) {
    $j = Get-Content $weave_w20 -Raw | ConvertFrom-Json
    $keys = $j.PSObject.Properties.Name
    Write-Host ("  keys: " + ($keys -join ", "))
    if ($j.all_pairs) {
        Write-Host ("  artfid_all: " + $j.all_pairs.mean_art_fid)
        Write-Host ("  clip_s: " + $j.all_pairs.mean_clip_style)
        Write-Host ("  lpips: " + $j.all_pairs.mean_content_lpips)
    }
    if ($j.transfer_only) {
        Write-Host ("  artfid_transfer: " + $j.transfer_only.mean_art_fid)
    }
}

# Check Seedream D5
Write-Host "`n=== Seedream D5 ==="
$sd_sum = "I:\Github\Latent_Style\exp_baselines\seedream45_api\distinct5_512_seedream45_windhub_20260607_repaired750\summary.json"
if (Test-Path $sd_sum) {
    $j = Get-Content $sd_sum -Raw | ConvertFrom-Json
    if ($j.all_pairs) {
        Write-Host ("  artfid_all: " + $j.all_pairs.mean_art_fid)
        Write-Host ("  clip_s: " + $j.all_pairs.mean_clip_style)
        Write-Host ("  lpips: " + $j.all_pairs.mean_content_lpips)
    }
    if ($j.transfer_only) {
        Write-Host ("  artfid_transfer: " + $j.transfer_only.mean_art_fid)
    }
}

# Check Random5 baselines
Write-Host "`n=== Random5-WikiArt baselines ==="
$r5 = "$exp_root\baseline_wikiarts20"
if (Test-Path $r5) {
    Get-ChildItem $r5 -Directory | ForEach-Object {
        $sj = Join-Path $_.FullName "summary.json"
        if (Test-Path $sj) {
            $j = Get-Content $sj -Raw | ConvertFrom-Json
            $artfid = $null
            $t_artfid = $null
            if ($j.all_pairs -and $j.all_pairs.mean_art_fid) { $artfid = $j.all_pairs.mean_art_fid }
            if ($j.transfer_only -and $j.transfer_only.mean_art_fid) { $t_artfid = $j.transfer_only.mean_art_fid }
            $img_cnt = 0
            $img_dir = Join-Path $_.FullName "images"
            if (Test-Path $img_dir) { $img_cnt = (Get-ChildItem $img_dir -Filter *.png -ErrorAction SilentlyContinue).Count }
            Write-Host ("  " + $_.Name + ": artfid_all=${artfid} artfid_transfer=${t_artfid} imgs=${img_cnt}")
        }
    }
}

# Check for any existing CFSD implementation
Write-Host "`n=== Search for CFSD implementation ==="
Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge" -Filter "*.py" -Recurse -Depth 3 -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "cfsd|csfd" } | ForEach-Object {
    Write-Host ("  " + $_.FullName)
}
Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\tools" -Filter "*.py" -ErrorAction SilentlyContinue | ForEach-Object {
    $content = Get-Content $_.FullName -Raw -ErrorAction SilentlyContinue
    if ($content -match "cfsd|csfd|content_fidelity_style_distance") {
        Write-Host ("  MATCH: " + $_.FullName)
    }
}
