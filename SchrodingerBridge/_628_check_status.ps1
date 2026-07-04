# Check T5 ep7 and 628_destructive status on remote
$ErrorActionPreference = 'Continue'

$t5Ckpt = 'I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/t5_b2v2_d2_d4/epoch_0007.pt'
if (Test-Path $t5Ckpt) {
    $f = Get-Item $t5Ckpt
    Write-Host "T5_ep7 EXISTS size=$([math]::Round($f.Length/1MB,2))MB mtime=$($f.LastWriteTime)"
} else {
    Write-Host "T5_ep7 MISSING"
}

$t5Sum = 'I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/t5_b2v2_d2_d4/full_eval/epoch_0007/summary.json'
if (Test-Path $t5Sum) {
    $j = Get-Content $t5Sum -Raw | ConvertFrom-Json
    $ap = $j.analysis.all_pairs_overview
    $tr = $j.analysis.style_transfer_ability
    Write-Host "T5_ep7 summary: ap_clip=$($ap.clip_style) ap_lpips=$($ap.content_lpips) tr_clip=$($tr.clip_style) tr_lpips=$($tr.content_lpips)"
} else {
    Write-Host "T5_ep7 summary MISSING"
    # Try ep10 fallback
    $t5Sum10 = 'I:/Github/Latent_Style/SchrodingerBridge/exp/p4_fusion_breakout/t5_b2v2_d2_d4/full_eval/epoch_0010/summary.json'
    if (Test-Path $t5Sum10) {
        $j = Get-Content $t5Sum10 -Raw | ConvertFrom-Json
        $ap = $j.analysis.all_pairs_overview
        $tr = $j.analysis.style_transfer_ability
        Write-Host "T5_ep10 summary: ap_clip=$($ap.clip_style) ap_lpips=$($ap.content_lpips) tr_clip=$($tr.clip_style) tr_lpips=$($tr.content_lpips)"
    } else {
        Write-Host "T5_ep10 summary MISSING"
    }
}

$cfgDir = 'I:/Github/Latent_Style/SchrodingerBridge/configs/ablations/628_destructive'
if (Test-Path $cfgDir) {
    $cnt = (Get-ChildItem $cfgDir -Filter '*.json' -ErrorAction SilentlyContinue | Measure-Object).Count
    Write-Host "628_destructive configs count=$cnt"
} else {
    Write-Host "628_destructive dir MISSING"
}

$genScript = 'I:/Github/Latent_Style/SchrodingerBridge/628_gen_destructive_configs.py'
if (Test-Path $genScript) {
    Write-Host "gen_script EXISTS"
} else {
    Write-Host "gen_script MISSING"
}

# Check WSL distro
Write-Host "--- WSL distros ---"
wsl --list --verbose 2>&1 | ForEach-Object { Write-Host $_ }
