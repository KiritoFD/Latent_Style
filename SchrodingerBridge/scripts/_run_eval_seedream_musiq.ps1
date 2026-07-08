$ErrorActionPreference = 'Stop'

$env:HF_HOME = "C:\Users\Administrator\.cache\huggingface"
$env:TRANSFORMERS_OFFLINE = "1"
$env:TORCH_HOME = "C:\Users\Administrator\.cache\torch"
$env:CUDA_VISIBLE_DEVICES = "0"

$script = "C:\Users\Administrator\_eval_unified.py"
$exp_root = "I:\Github\Latent_Style\SchrodingerBridge\exp"
$sd_dir = "I:\Github\Latent_Style\exp_baselines\seedream45_api\distinct5_512_seedream45_windhub_20260607_repaired750\images"

Write-Host "============================================================"
Write-Host "SeeDream D5-512 MUSIQ-only evaluation (750 images)"
Write-Host "============================================================"
python $script `
    --image-dir $sd_dir `
    --dataset wiki20distinct5 `
    --output "$exp_root\_eval_seedream_d5_musiq.json" `
    --max-images 750 `
    --skip-clip --skip-lpips
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] SeeDream D5 MUSIQ failed (exit=$LASTEXITCODE)"
} else {
    Write-Host "[OK] SeeDream D5 MUSIQ done"
}

Write-Host "`n=== Result ==="
$out = "$exp_root\_eval_seedream_d5_musiq.json"
if (Test-Path $out) {
    Get-Content $out
    Write-Host ""
}

Write-Host "`n=== Also check final_works\CUT summary for P256 MUSIQ ==="
$fw_cut = "I:\Github\Latent_Style\final_works\CUT"
if (Test-Path "$fw_cut\summary.json") {
    $j = Get-Content "$fw_cut\summary.json" -Raw | ConvertFrom-Json
    $keys = $j.PSObject.Properties.Name
    Write-Host ("  keys: " + ($keys -join ", "))
    # Check if there's any musiq or 256 reference
    if ($j.metrics_note) { Write-Host ("  metrics_note keys: " + ($j.metrics_note.PSObject.Properties.Name -join ", ")) }
    if ($j.settings) {
        Write-Host ("  settings.image_size: " + $j.settings.image_size)
        Write-Host ("  settings.test_root: " + $j.settings.test_root)
        Write-Host ("  settings.dataset: " + $j.settings.dataset)
    }
    # Check for musiq in pool results
    if ($j.pool) {
        $pk = $j.pool.PSObject.Properties.Name
        Write-Host ("  pool keys: " + ($pk -join ", "))
        $musiq_keys = $pk | Where-Object { $_ -match "musiq" }
        foreach ($k in $musiq_keys) { Write-Host ("    pool." + $k + ": " + $j.pool.$k) }
    }
}

Write-Host "`n=== Check final_works\CUT meta.json ==="
if (Test-Path "$fw_cut\meta.json") {
    Get-Content "$fw_cut\meta.json"
    Write-Host ""
}
