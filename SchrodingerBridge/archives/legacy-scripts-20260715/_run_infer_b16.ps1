# Infra optimization: inference benchmark — b16 configurations
# Push batch_size to 16 to test if gen time can be further reduced
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$ckpt = "exp\t1_asg_5ep\epoch_0005.pt"
$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "C:\Users\Administrator\.cache\huggingface\hub"
$logOut = "C:\Users\Administrator\logs\infra_infer_b16.out"

function Run-Eval {
    param([string]$name, [string[]]$extraArgs)
    $evalDir = "exp\infra_infer_bench\$name\full_eval\epoch_0005"
    Write-Output ""
    Write-Output "=== BENCH: $name START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
    if (Test-Path "$evalDir\summary.json") { Remove-Item "$evalDir\summary.json" -Force }
    $allArgs = @(
        "-u", "src\utils\run_evaluation.py",
        "--checkpoint", $ckpt,
        "--output", $evalDir,
        "--test_dir", $testDir,
        "--cache_dir", $cacheDir,
        "--clip_hf_cache_dir", $hfCache,
        "--eval_only_lpips_clip_style", "--eval_lpips_chunk_size", "4"
    ) + $extraArgs
    & python @allArgs 2>&1 | Tee-Object -FilePath $logOut -Append
    Write-Output "=== BENCH: $name DONE exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
}

# Config 1: batch=16, save images
Run-Eval "b16_save" @(
    "--batch_size", "16", "--generation_batch_size", "16", "--metric_batch_size", "16",
    "--target_chunk_size", "1", "--vae_decode_batch_size", "16"
)

# Config 2: batch=16, NO save images (push gen + minimize other)
Run-Eval "b16_nosave" @(
    "--batch_size", "16", "--generation_batch_size", "16", "--metric_batch_size", "16",
    "--target_chunk_size", "1", "--vae_decode_batch_size", "16",
    "--no-save_generated_images", "--no-save_summary_grid"
)

Write-Output ""
Write-Output "=== ALL B16 BENCHMARKS COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
