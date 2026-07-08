$ErrorActionPreference = 'Continue'

$exp_root = "I:\Github\Latent_Style\SchrodingerBridge\exp"
$bv2 = "$exp_root\baseline_v2"

# Check structure of one D5 baseline summary
Write-Host "=== identity summary.json structure ==="
$id_sum = "$bv2\eval\identity\summary.json"
if (Test-Path $id_sum) {
    $j = Get-Content $id_sum -Raw | ConvertFrom-Json
    $keys = $j.PSObject.Properties.Name
    Write-Host ("  top keys: " + ($keys -join ", "))
    if ($j.matrix_breakdown) {
        Write-Host "  matrix_breakdown keys:"
        $mb = $j.matrix_breakdown
        $mb_keys = $mb.PSObject.Properties.Name
        Write-Host ("    " + ($mb_keys -join ", "))
        # Get first src_style
        $first = $mb_keys | Select-Object -First 1
        $sub = $mb.$first
        $sub_keys = $sub.PSObject.Properties.Name
        Write-Host ("    sub keys ($first): " + ($sub_keys -join ", "))
        # Get first tgt
        $first_tgt = $sub_keys | Select-Object -First 1
        $leaf = $sub.$first_tgt
        Write-Host ("    leaf keys ($first -> $first_tgt): " + ($leaf.PSObject.Properties.Name -join ", "))
        Write-Host ("    art_fid: " + $leaf.art_fid)
        Write-Host ("    clip_style: " + $leaf.clip_style)
        Write-Host ("    content_lpips: " + $leaf.content_lpips)
    }
    if ($j.analysis) {
        Write-Host "  analysis keys:"
        $ak = $j.analysis.PSObject.Properties.Name
        Write-Host ("    " + ($ak -join ", "))
        if ($j.analysis.all_pairs) {
            Write-Host ("    all_pairs keys: " + ($j.analysis.all_pairs.PSObject.Properties.Name -join ", "))
            if ($j.analysis.all_pairs.mean_art_fid) { Write-Host ("    mean_art_fid: " + $j.analysis.all_pairs.mean_art_fid) }
        }
        if ($j.analysis.transfer_only) {
            Write-Host ("    transfer_only keys: " + ($j.analysis.transfer_only.PSObject.Properties.Name -join ", "))
            if ($j.analysis.transfer_only.mean_art_fid) { Write-Host ("    transfer mean_art_fid: " + $j.analysis.transfer_only.mean_art_fid) }
        }
    }
}

# Now extract ArtFID for all D5 methods from analysis section
Write-Host "`n=== D5 ArtFID from analysis section ==="
$methods = @("identity","adain","wct_v32k","sdturbo","styleid","samam","samst","cut")
foreach ($m in $methods) {
    $sj = "$bv2\eval\${m}\summary.json"
    if (Test-Path $sj) {
        $j = Get-Content $sj -Raw | ConvertFrom-Json
        $artfid_all = "N/A"
        $artfid_transfer = "N/A"
        $clip_all = "N/A"
        $lpips_all = "N/A"
        if ($j.analysis -and $j.analysis.all_pairs) {
            if ($j.analysis.all_pairs.mean_art_fid) { $artfid_all = $j.analysis.all_pairs.mean_art_fid }
            if ($j.analysis.all_pairs.mean_clip_style) { $clip_all = $j.analysis.all_pairs.mean_clip_style }
            if ($j.analysis.all_pairs.mean_content_lpips) { $lpips_all = $j.analysis.all_pairs.mean_content_lpips }
        }
        if ($j.analysis -and $j.analysis.transfer_only) {
            if ($j.analysis.transfer_only.mean_art_fid) { $artfid_transfer = $j.analysis.transfer_only.mean_art_fid }
        }
        Write-Host ("  ${m}: artfid_all=${artfid_all} artfid_transfer=${artfid_transfer} clip=${clip_all} lpips=${lpips_all}")
    }
}

# WEAVE D5
Write-Host "`n=== WEAVE D5 ==="
$weave_d5 = "$exp_root\clean_base_v2\full_eval\epoch_0010\summary.json"
if (Test-Path $weave_d5) {
    $j = Get-Content $weave_d5 -Raw | ConvertFrom-Json
    $keys = $j.PSObject.Properties.Name
    Write-Host ("  keys: " + ($keys -join ", "))
    if ($j.analysis) {
        $ak = $j.analysis.PSObject.Properties.Name
        Write-Host ("  analysis keys: " + ($ak -join ", "))
        if ($j.analysis.all_pairs) {
            Write-Host ("  all_pairs keys: " + ($j.analysis.all_pairs.PSObject.Properties.Name -join ", "))
            Write-Host ("  mean_art_fid: " + $j.analysis.all_pairs.mean_art_fid)
            Write-Host ("  mean_clip_style: " + $j.analysis.all_pairs.mean_clip_style)
            Write-Host ("  mean_content_lpips: " + $j.analysis.all_pairs.mean_content_lpips)
        }
    }
} else {
    # Check if summary is in different location
    $weave_parent = "$exp_root\clean_base_v2\full_eval\epoch_0010"
    if (Test-Path $weave_parent) {
        Write-Host "  Files in ${weave_parent}:"
        Get-ChildItem $weave_parent | ForEach-Object { Write-Host ("    " + $_.Name) }
    }
    $weave_parent2 = "$exp_root\clean_base_v2\full_eval"
    if (Test-Path $weave_parent2) {
        Write-Host "  Files in ${weave_parent2}:"
        Get-ChildItem $weave_parent2 | ForEach-Object { Write-Host ("    " + $_.Name) }
    }
}

# WEAVE W20
Write-Host "`n=== WEAVE W20 (wikiarts20_eval) ==="
$weave_w20 = "$exp_root\wikiarts20_eval\summary.json"
if (Test-Path $weave_w20) {
    $j = Get-Content $weave_w20 -Raw | ConvertFrom-Json
    if ($j.analysis) {
        $ak = $j.analysis.PSObject.Properties.Name
        Write-Host ("  analysis keys: " + ($ak -join ", "))
        if ($j.analysis.all_pairs) {
            Write-Host ("  all_pairs keys: " + ($j.analysis.all_pairs.PSObject.Properties.Name -join ", "))
            Write-Host ("  mean_art_fid: " + $j.analysis.all_pairs.mean_art_fid)
            Write-Host ("  mean_clip_style: " + $j.analysis.all_pairs.mean_clip_style)
            Write-Host ("  mean_content_lpips: " + $j.analysis.all_pairs.mean_content_lpips)
        }
        if ($j.analysis.transfer_only) {
            Write-Host ("  transfer mean_art_fid: " + $j.analysis.transfer_only.mean_art_fid)
        }
    }
}

# Seedream D5
Write-Host "`n=== Seedream D5 ==="
$sd_sum = "I:\Github\Latent_Style\exp_baselines\seedream45_api\distinct5_512_seedream45_windhub_20260607_repaired750\summary.json"
if (Test-Path $sd_sum) {
    $j = Get-Content $sd_sum -Raw | ConvertFrom-Json
    if ($j.analysis) {
        Write-Host ("  analysis keys: " + ($j.analysis.PSObject.Properties.Name -join ", "))
        if ($j.analysis.all_pairs) {
            Write-Host ("  mean_art_fid: " + $j.analysis.all_pairs.mean_art_fid)
        }
        if ($j.analysis.transfer_only) {
            Write-Host ("  transfer mean_art_fid: " + $j.analysis.transfer_only.mean_art_fid)
        }
    }
}

# R5 baselines
Write-Host "`n=== Random5-WikiArt baselines ArtFID ==="
$r5 = "$exp_root\baseline_wikiarts20"
if (Test-Path $r5) {
    Get-ChildItem $r5 -Directory | ForEach-Object {
        $sj = Join-Path $_.FullName "summary.json"
        if (Test-Path $sj) {
            $j = Get-Content $sj -Raw | ConvertFrom-Json
            $artfid = "N/A"
            $t_artfid = "N/A"
            if ($j.analysis -and $j.analysis.all_pairs -and $j.analysis.all_pairs.mean_art_fid) { $artfid = $j.analysis.all_pairs.mean_art_fid }
            if ($j.analysis -and $j.analysis.transfer_only -and $j.analysis.transfer_only.mean_art_fid) { $t_artfid = $j.analysis.transfer_only.mean_art_fid }
            Write-Host ("  " + $_.Name + ": artfid_all=${artfid} transfer=${t_artfid}")
        }
    }
}
