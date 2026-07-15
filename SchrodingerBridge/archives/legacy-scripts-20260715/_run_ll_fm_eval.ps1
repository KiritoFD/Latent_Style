# Direct python launch with native PS redirection - survives ssh disconnect
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONPATH = "."

$ckpt = "exp\abl_no_ll_fm\epoch_0015.pt"
$evalDir = "exp\abl_no_ll_fm\full_eval\epoch_0015"
$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "C:\Users\Administrator\.cache\huggingface\hub"
$dinoOut = "exp\_dino_results\abl_no_ll_fm.json"
$logOut = "C:\Users\Administrator\logs\abl_ll_fm_full.out"
$logErr = "C:\Users\Administrator\logs\abl_ll_fm_full.err"

# Phase 1: full_eval
Add-Content -Path $logOut -Value "=== START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
$p1 = Start-Process -FilePath "python.exe" `
    -ArgumentList @(
        "-u", "src\utils\run_evaluation.py",
        "--checkpoint", $ckpt,
        "--output", $evalDir,
        "--test_dir", $testDir,
        "--cache_dir", $cacheDir,
        "--clip_hf_cache_dir", $hfCache,
        "--batch_size", "2",
        "--generation_batch_size", "2",
        "--metric_batch_size", "2",
        "--target_chunk_size", "1",
        "--vae_decode_batch_size", "16",
        "--eval_only_lpips_clip_style",
        "--eval_lpips_chunk_size", "4"
    ) `
    -RedirectStandardOutput $logOut `
    -RedirectStandardError $logErr `
    -NoNewWindow -PassThru
Add-Content -Path $logOut -Value "=== full_eval PID=$($p1.Id) launched ==="
$p1.WaitForExit()
$ec1 = $p1.ExitCode
Add-Content -Path $logOut -Value "=== full_eval DONE exit=$ec1 $(Get-Date -Format 'HH:mm:ss') ==="

# Phase 2: DINO
if ($ec1 -eq 0) {
    $imgDir = Join-Path $evalDir "images"
    $p2 = Start-Process -FilePath "python.exe" `
        -ArgumentList @(
            "_compute_dino.py",
            "--images_dir", $imgDir,
            "--test_dir", $testDir,
            "--dataset", "wikiart",
            "--output", $dinoOut,
            "--hf_cache", $hfCache,
            "--max_refs", "30"
        ) `
        -RedirectStandardOutput $logOut `
        -RedirectStandardError $logErr `
        -NoNewWindow -PassThru
    $p2.WaitForExit()
    $ec2 = $p2.ExitCode
    Add-Content -Path $logOut -Value "=== DINO DONE exit=$ec2 $(Get-Date -Format 'HH:mm:ss') ==="
} else {
    Add-Content -Path $logOut -Value "=== SKIP DINO (eval failed) ==="
}
Add-Content -Path $logOut -Value "=== ALL COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
