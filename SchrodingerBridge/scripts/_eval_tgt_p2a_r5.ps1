# TGT baseline evaluation: P2A-256 and R5-WikiArt
# Runs on remote server (I: drive)
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "C:\Users\Administrator\.cache\huggingface\hub"

# ===== P2A-256 TGT CLIP/LPIPS =====
$p2aTestDir = "I:\datasets\legacy256_overfit50\test"
$p2aStyles = "cezanne,Hayao,monet,photo,vangogh"
$p2aTgtDir = "exp\target_style_baseline_p2a"

Write-Output "=== P2A-256 TGT CLIP/LPIPS START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
if (Test-Path "$p2aTgtDir\summary.json") { Remove-Item "$p2aTgtDir\summary.json" -Force }
$p2aArgs = @(
    "-u", "src\utils\run_evaluation.py",
    $p2aTgtDir,
    "--style_subdirs", $p2aStyles,
    "--test_dir", $p2aTestDir,
    "--cache_dir", $cacheDir,
    "--clip_hf_cache_dir", $hfCache,
    "--reuse_generated",
    "--eval_only_lpips_clip_style",
    "--eval_lpips_chunk_size", "4",
    "--batch_size", "2",
    "--generation_batch_size", "2",
    "--metric_batch_size", "2",
    "--target_chunk_size", "1",
    "--vae_decode_batch_size", "2",
    "--ref_feature_batch_size", "2"
)
& python @p2aArgs 2>&1
Write-Output "=== P2A-256 TGT CLIP/LPIPS DONE exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

# ===== R5-WikiArt TGT CLIP/LPIPS =====
$r5TestDir = "I:\datasets\wikiarts20_512_test"
$r5Styles = "Cubism,Expressionism,Pop_Art,Romanticism,Symbolism"
$r5TgtDir = "exp\target_style_baseline_r5"

Write-Output ""
Write-Output "=== R5 TGT CLIP/LPIPS START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
if (Test-Path "$r5TgtDir\summary.json") { Remove-Item "$r5TgtDir\summary.json" -Force }
$r5Args = @(
    "-u", "src\utils\run_evaluation.py",
    $r5TgtDir,
    "--style_subdirs", $r5Styles,
    "--test_dir", $r5TestDir,
    "--cache_dir", $cacheDir,
    "--clip_hf_cache_dir", $hfCache,
    "--reuse_generated",
    "--eval_only_lpips_clip_style",
    "--eval_lpips_chunk_size", "4",
    "--batch_size", "2",
    "--generation_batch_size", "2",
    "--metric_batch_size", "2",
    "--target_chunk_size", "1",
    "--vae_decode_batch_size", "2",
    "--ref_feature_batch_size", "2"
)
& python @r5Args 2>&1
Write-Output "=== R5 TGT CLIP/LPIPS DONE exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

Write-Output ""
Write-Output "=== ALL TGT CLIP/LPIPS COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="