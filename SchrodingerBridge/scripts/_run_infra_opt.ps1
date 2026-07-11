# Infra optimization: test VAE compile + bigger vae_decode_batch_size
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$ckpt = "exp\t1_asg_5ep\epoch_0005.pt"
$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "C:\Users\Administrator\.cache\huggingface\hub"

function Run-Eval {
    param([string]$name, [string[]]$extraArgs)
    $evalDir = "exp\infra_opt\$name\full_eval\epoch_0005"
    Write-Output ""
    Write-Output "=== OPT: $name START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
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
        "--target_chunk_size", "1",
        "--profile_timing"
    ) + $extraArgs
    & python @allArgs 2>&1 | Tee-Object -FilePath "C:\Users\Administrator\logs\infra_opt_$name.out" -Append
    Write-Output "=== OPT: $name DONE exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
}

# Config 1: b16 + vae_compile_decoder (reduce-overhead)
Run-Eval "b16_vaecomp" @(
    "--vae_decode_batch_size", "16",
    "--vae_compile_decoder", "--vae_compile_mode", "reduce-overhead"
)

# Config 2: b32 vae decode batch (bigger VAE batch, no compile)
Run-Eval "b16_vaebs32" @(
    "--vae_decode_batch_size", "32"
)

# Config 3: b32 vae decode + compile
Run-Eval "b16_vaebs32_comp" @(
    "--vae_decode_batch_size", "32",
    "--vae_compile_decoder", "--vae_compile_mode", "reduce-overhead"
)

# Config 4: b48 vae decode + compile (push batch)
Run-Eval "b16_vaebs48_comp" @(
    "--vae_decode_batch_size", "48",
    "--vae_compile_decoder", "--vae_compile_mode", "reduce-overhead"
)

Write-Output ""
Write-Output "=== ALL OPT BENCHMARKS COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
