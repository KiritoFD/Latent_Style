$ErrorActionPreference = 'SilentlyContinue'
$repo = "I:\Github\Latent_Style\SchrodingerBridge"
$file = Join-Path $repo "src\spectral_losses620.py"

Write-Output "=== _semantic_region_swd body (lines 170-210) ==="
$lines = Get-Content $file
for ($i = 169; $i -lt 210 -and $i -lt $lines.Count; $i++) {
    Write-Output "$($i+1): $($lines[$i])"
}

Write-Output ""
Write-Output "=== _projection_dirs method (search) ==="
for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match "_projection_dir_cache|_projection_dirs") {
        $start = [Math]::Max(0, $i - 2)
        $end = [Math]::Min($lines.Count - 1, $i + 20)
        for ($j = $start; $j -le $end; $j++) {
            Write-Output "$($j+1): $($lines[$j])"
        }
        Write-Output "---"
        break
    }
}

Write-Output ""
Write-Output "=== RESULTS ==="
$exps = @("abl_baseline","abl_k1_global","abl_blend0","abl_blend1","abl_k64","abl_soft_mask","abl_ll_w0","abl_ll_w1","abl_route_p05","abl_route_p10","abl_sinkhorn","abl_spectral","abl_no_swd_loss","abl_no_dwt_route","abl_no_wct","abl_no_eota")
foreach ($name in $exps) {
    $expDir = Join-Path $repo "exp\$name\full_eval"
    if (-not (Test-Path $expDir)) { Write-Output "$name : NO_EVAL_DIR"; continue }
    $found = $false
    foreach ($epochDir in Get-ChildItem $expDir -Directory) {
        $summary = Join-Path $epochDir.FullName "summary.json"
        if (Test-Path $summary) {
            try {
                $data = Get-Content $summary -Raw | ConvertFrom-Json
                $apo = $data.analysis.all_pairs_overview
                $xfer = $null
                if ($data.analysis.PSObject.Properties.Name -contains "xfer_pairs_overview") {
                    $xfer = $data.analysis.xfer_pairs_overview
                }
                $clipAll = $apo.clip_style
                $lpipsAll = $apo.content_lpips
                $clipXfer = if ($xfer) { $xfer.clip_style } else { "N/A" }
                $lpipsXfer = if ($xfer) { $xfer.content_lpips } else { "N/A" }
                Write-Output "$name : CLIP_all=$clipAll LPIPS_all=$lpipsAll CLIP_xfer=$clipXfer LPIPS_xfer=$lpipsXfer (epoch=$($epochDir.Name))"
                $found = $true
                break
            } catch {
                Write-Output "$name : PARSE_ERROR"
            }
        }
    }
    if (-not $found) { Write-Output "$name : NO_SUMMARY" }
}

Write-Output ""
Write-Output "=== RUNNER LOG TAIL ==="
Get-Content (Join-Path $repo "remote_ablation_log.txt") -Tail 15
