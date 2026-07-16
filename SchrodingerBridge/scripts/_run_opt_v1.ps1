# Run optimized evaluation with VAE compile on the correct main-table checkpoint
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\WEAVE"

$ckpt = "runs\submission\repro_brk_a_15ep\epoch_0006.pt"
$outDir = "_tmp_opt_v1"
$cfg = "I:\Github\Latent_Style\WEAVE\inference_optimized.json"
$python = "C:\Program Files\Python312\python.exe"
$script = "I:\Github\Latent_Style\WEAVE\utils\run_evaluation.py"
$logFile = "I:\Github\Latent_Style\WEAVE\_run_opt_v2.log"

# Clean output dir if it exists (to force fresh run)
if (Test-Path $outDir) {
    Remove-Item -Recurse -Force $outDir
}

# Keep compile cache persistent on disk — first run compiles & caches, later runs load from disk
$cacheDir = "experiments\inference_speed\.compile_cache"
if (Test-Path $cacheDir) {
    Write-Host "Compile cache exists at $cacheDir (will reuse)"
} else {
    Write-Host "No compile cache yet (first run will compile & cache)"
}

Write-Host "Starting optimized evaluation..."
Write-Host "  checkpoint: $ckpt"
Write-Host "  output:     $outDir"
Write-Host "  config:     $cfg"

# Build argument list
$pyArgs = @(
    $script,
    "--checkpoint", $ckpt,
    "--output", $outDir,
    "--test_dir", "data\test",
    "--cache_dir", "runs\cache",
    "--clip_hf_cache_dir", "runs\cache\hf",
    "--batch_size", "16",
    "--save_generated_images",
    "--config_override", $cfg,
    "--vae_compile_decoder",
    "--vae_compile_mode", "reduce-overhead",
    "--vae_compile_fullgraph",
    "--vae_compile_cache_dir", "experiments/inference_speed/.compile_cache_ro"
)

Write-Host "Python: $python"
Write-Host "Args: $($pyArgs -join ' ')"
Write-Host "Log:   $logFile"

# Run synchronously with output redirected to log
& $python @pyArgs *> $logFile
$exitCode = $LASTEXITCODE

Write-Host "Exit code: $exitCode"
if (Test-Path "$outDir\summary.json") {
    Write-Host "SUCCESS: summary.json generated"
} else {
    Write-Host "FAILED: summary.json not generated"
    Write-Host "--- last 30 lines of log ---"
    if (Test-Path $logFile) {
        Get-Content $logFile -Tail 30
    }
}
