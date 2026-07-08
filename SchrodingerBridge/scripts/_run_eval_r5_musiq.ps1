$ErrorActionPreference = 'Stop'

$env:HF_HOME = "C:\Users\Administrator\.cache\huggingface"
$env:TRANSFORMERS_OFFLINE = "1"
$env:TORCH_HOME = "C:\Users\Administrator\.cache\torch"
$env:CUDA_VISIBLE_DEVICES = "0"

$script = "C:\Users\Administrator\_eval_unified.py"
$exp_root = "I:\Github\Latent_Style\SchrodingerBridge\exp"

# Identity R5 MUSIQ (12000 images, use max-images=750 for consistency with D5 baselines)
Write-Host "============================================================"
Write-Host "Identity R5 MUSIQ"
Write-Host "============================================================"
python $script --image-dir "$exp_root\baseline_wikiarts20\identity\images" --dataset wiki20distinct5 --output "$exp_root\_eval_identity_r5_musiq.json" --max-images 750 --skip-clip --skip-lpips
if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] Identity R5 failed" } else { Write-Host "[OK] Identity R5 done" }

# AdaIN R5 MUSIQ
Write-Host "`n============================================================"
Write-Host "AdaIN R5 MUSIQ"
Write-Host "============================================================"
python $script --image-dir "$exp_root\baseline_wikiarts20\adain\images" --dataset wiki20distinct5 --output "$exp_root\_eval_adain_r5_musiq.json" --max-images 750 --skip-clip --skip-lpips
if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] AdaIN R5 failed" } else { Write-Host "[OK] AdaIN R5 done" }

# WCT R5 MUSIQ
Write-Host "`n============================================================"
Write-Host "WCT R5 MUSIQ"
Write-Host "============================================================"
python $script --image-dir "$exp_root\baseline_wikiarts20\wct\images" --dataset wiki20distinct5 --output "$exp_root\_eval_wct_r5_musiq.json" --max-images 750 --skip-clip --skip-lpips
if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] WCT R5 failed" } else { Write-Host "[OK] WCT R5 done" }

Write-Host "`n============================================================"
Write-Host "RESULTS"
Write-Host "============================================================"
@("_eval_identity_r5_musiq.json","_eval_adain_r5_musiq.json","_eval_wct_r5_musiq.json") | ForEach-Object {
    $p = Join-Path $exp_root $_
    if (Test-Path $p) {
        Write-Host "--- $_ ---"
        Get-Content $p
        Write-Host ""
    }
}
