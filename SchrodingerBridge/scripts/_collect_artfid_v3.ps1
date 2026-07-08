$ErrorActionPreference = 'Continue'

$exp_root = "I:\Github\Latent_Style\SchrodingerBridge\exp"
$bv2 = "$exp_root\baseline_v2"

# Check all_pairs_overview structure for identity
Write-Host "=== identity all_pairs_overview ==="
$id_sum = "$bv2\eval\identity\summary.json"
if (Test-Path $id_sum) {
    $j = Get-Content $id_sum -Raw | ConvertFrom-Json
    if ($j.analysis -and $j.analysis.all_pairs_overview) {
        $apo = $j.analysis.all_pairs_overview
        Write-Host ("  keys: " + ($apo.PSObject.Properties.Name -join ", "))
        $apo | ConvertTo-Json -Depth 3 | Write-Host
    }
}

# Extract ArtFID from all_pairs_overview for all D5 methods
Write-Host "`n=== D5 ArtFID from all_pairs_overview ==="
$methods = @("identity","adain","wct_v32k","sdturbo","styleid","samam","samst","cut")
foreach ($m in $methods) {
    $sj = "$bv2\eval\${m}\summary.json"
    if (Test-Path $sj) {
        $j = Get-Content $sj -Raw | ConvertFrom-Json
        if ($j.analysis -and $j.analysis.all_pairs_overview) {
            $apo = $j.analysis.all_pairs_overview
            $artfid = $apo.mean_art_fid
            $clip = $apo.mean_clip_style
            $lpips = $apo.mean_content_lpips
            $t_artfid = "N/A"
            if ($j.analysis.style_transfer_ability -and $j.analysis.style_transfer_ability.mean_art_fid) {
                $t_artfid = $j.analysis.style_transfer_ability.mean_art_fid
            }
            Write-Host ("  ${m}: artfid=${artfid} clip=${clip} lpips=${lpips} transfer_artfid=${t_artfid}")
        }
    }
}

# WEAVE D5
Write-Host "`n=== WEAVE D5 ==="
$weave_d5 = "$exp_root\clean_base_v2\full_eval\epoch_0010\summary.json"
if (Test-Path $weave_d5) {
    $j = Get-Content $weave_d5 -Raw | ConvertFrom-Json
    if ($j.analysis -and $j.analysis.all_pairs_overview) {
        $apo = $j.analysis.all_pairs_overview
        Write-Host ("  artfid: " + $apo.mean_art_fid)
        Write-Host ("  clip: " + $apo.mean_clip_style)
        Write-Host ("  lpips: " + $apo.mean_content_lpips)
    }
    if ($j.analysis.style_transfer_ability) {
        Write-Host ("  transfer_artfid: " + $j.analysis.style_transfer_ability.mean_art_fid)
    }
} else {
    Write-Host "  NOT FOUND"
    # Search for summary.json in clean_base_v2
    Get-ChildItem "$exp_root\clean_base_v2" -Filter "summary.json" -Recurse -Depth 3 -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host ("  Found: " + $_.FullName)
    }
}

# WEAVE W20
Write-Host "`n=== WEAVE W20 ==="
$weave_w20 = "$exp_root\wikiarts20_eval\summary.json"
if (Test-Path $weave_w20) {
    $j = Get-Content $weave_w20 -Raw | ConvertFrom-Json
    if ($j.analysis -and $j.analysis.all_pairs_overview) {
        $apo = $j.analysis.all_pairs_overview
        Write-Host ("  artfid: " + $apo.mean_art_fid)
        Write-Host ("  clip: " + $apo.mean_clip_style)
        Write-Host ("  lpips: " + $apo.mean_content_lpips)
    }
    if ($j.analysis.style_transfer_ability) {
        Write-Host ("  transfer_artfid: " + $j.analysis.style_transfer_ability.mean_art_fid)
    }
}

# Seedream D5
Write-Host "`n=== Seedream D5 ==="
$sd_sum = "I:\Github\Latent_Style\exp_baselines\seedream45_api\distinct5_512_seedream45_windhub_20260607_repaired750\summary.json"
if (Test-Path $sd_sum) {
    $j = Get-Content $sd_sum -Raw | ConvertFrom-Json
    if ($j.analysis -and $j.analysis.all_pairs_overview) {
        $apo = $j.analysis.all_pairs_overview
        Write-Host ("  artfid: " + $apo.mean_art_fid)
        Write-Host ("  clip: " + $apo.mean_clip_style)
        Write-Host ("  lpips: " + $apo.mean_content_lpips)
    }
    if ($j.analysis.style_transfer_ability) {
        Write-Host ("  transfer_artfid: " + $j.analysis.style_transfer_ability.mean_art_fid)
    }
}

# R5 baselines
Write-Host "`n=== Random5-WikiArt baselines ==="
$r5 = "$exp_root\baseline_wikiarts20"
if (Test-Path $r5) {
    Get-ChildItem $r5 -Directory | ForEach-Object {
        $sj = Join-Path $_.FullName "summary.json"
        if (Test-Path $sj) {
            $j = Get-Content $sj -Raw | ConvertFrom-Json
            if ($j.analysis -and $j.analysis.all_pairs_overview) {
                $apo = $j.analysis.all_pairs_overview
                $artfid = $apo.mean_art_fid
                $t_artfid = "N/A"
                if ($j.analysis.style_transfer_ability -and $j.analysis.style_transfer_ability.mean_art_fid) {
                    $t_artfid = $j.analysis.style_transfer_ability.mean_art_fid
                }
                Write-Host ("  " + $_.Name + ": artfid=${artfid} transfer=${t_artfid}")
            }
        }
    }
}
