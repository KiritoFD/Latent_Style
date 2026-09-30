# Infra diagnosis: profile_timing b16 to get accurate stage breakdown
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$ckpt = "exp\t1_asg_5ep\epoch_0005.pt"
$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "C:\Users\Administrator\.cache\huggingface\hub"
$logOut = "C:\Users\Administrator\logs\infra_diag_profile.out"
$evalDir = "exp\infra_diag\b16_profile\full_eval\epoch_0005"

Write-Output "=== DIAG: b16 + profile_timing START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
if (Test-Path "$evalDir\summary.json") { Remove-Item "$evalDir\summary.json" -Force }
$allArgs = @(
    "-u", "src\utils\run_evaluation.py",
    "--checkpoint", $ckpt,
    "--output", $evalDir,
    "--test_dir", $testDir,
    "--cache_dir", $cacheDir,
    "--clip_hf_cache_dir", $hfCache,
    "--eval_only_lpips_clip_style", "--eval_lpips_chunk_size", "4",
    "--batch_size", "16", "--generation_batch_size", "16", "--metric_batch_size", "16",
    "--target_chunk_size", "1", "--vae_decode_batch_size", "16",
    "--profile_timing"
)
& python @allArgs 2>&1 | Tee-Object -FilePath $logOut -Append
Write-Output "=== DIAG DONE exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
