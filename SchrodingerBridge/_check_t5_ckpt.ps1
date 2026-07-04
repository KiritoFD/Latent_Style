$dir = "I:\Github\Latent_Style\SchrodingerBridge\exp\p4_fusion_breakout\infer_ablation"
Write-Host "=== T5_D4_u01_v3 details (best Pareto) ==="
$t5File = "$dir\T5_D4_u01_v3.json"
if (Test-Path $t5File) {
    $data = Get-Content $t5File -Raw | ConvertFrom-Json
    Write-Host "checkpoint: $($data.checkpoint)"
    Write-Host "config_path: $($data.config_path)"
    Write-Host "params:"
    $data.params | Format-List
    Write-Host "metrics:"
    $data.metrics | Format-List
}

Write-Host ""
Write-Host "=== T4_D1_dwt details (best clip) ==="
$t4File = "$dir\T4_D1_dwt.json"
if (Test-Path $t4File) {
    $data = Get-Content $t4File -Raw | ConvertFrom-Json
    Write-Host "checkpoint: $($data.checkpoint)"
    Write-Host "config_path: $($data.config_path)"
    Write-Host "params:"
    $data.params | Format-List
    Write-Host "metrics:"
    $data.metrics | Format-List
}

Write-Host ""
Write-Host "=== T5_D3_u01 details (U=0.1 only, no V) ==="
$t5d3File = "$dir\T5_D3_u01.json"
if (Test-Path $t5d3File) {
    $data = Get-Content $t5d3File -Raw | ConvertFrom-Json
    Write-Host "checkpoint: $($data.checkpoint)"
    Write-Host "params:"
    $data.params | Format-List
    Write-Host "metrics:"
    $data.metrics | Format-List
}

Write-Host ""
Write-Host "=== Check T5 checkpoint exists ==="
$t5Ckpt = "I:\Github\Latent_Style\SchrodingerBridge\exp\p4_fusion_breakout\t5_full_fusion\checkpoint"
if (Test-Path $t5Ckpt) {
    Write-Host "T5 checkpoint dir: $t5Ckpt"
    Get-ChildItem $t5Ckpt -Filter "*.pt" | ForEach-Object { Write-Host $_.Name }
} else {
    Write-Host "T5 checkpoint dir not found, searching..."
    Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\exp\p4_fusion_breakout" -Directory | ForEach-Object { Write-Host $_.Name }
}
