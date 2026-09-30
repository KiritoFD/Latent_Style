# Main table evaluation: P2A-256 and R5 with T1 ASG checkpoint
# Runs inference (b16_save) + CLIP/LPIPS eval, then DINO eval
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$ckpt = "exp\t1_asg_5ep\epoch_0005.pt"
$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "C:\Users\Administrator\.cache\huggingface\hub"
$logOut = "C:\Users\Administrator\logs\main_table_eval.out"

# ===== P2A-256 =====
$p2aTestDir = "I:\datasets\legacy256_overfit50\test"
$p2aStyles = "cezanne,Hayao,monet,photo,vangogh"
$p2aEvalDir = "exp\main_table\p2a_256\full_eval\epoch_0005"

Write-Output ""
Write-Output "=== P2A-256 EVAL START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
if (Test-Path "$p2aEvalDir\summary.json") { Remove-Item "$p2aEvalDir\summary.json" -Force }
$p2aArgs = @(
    "-u", "src\utils\run_evaluation.py",
    "--checkpoint", $ckpt,
    "--output", $p2aEvalDir,
    "--test_dir", $p2aTestDir,
    "--style_subdirs", $p2aStyles,
    "--cache_dir", $cacheDir,
    "--clip_hf_cache_dir", $hfCache,
    "--batch_size", "16", "--generation_batch_size", "16", "--metric_batch_size", "16",
    "--target_chunk_size", "1", "--vae_decode_batch_size", "16",
    "--eval_only_lpips_clip_style", "--eval_lpips_chunk_size", "4"
)
& python @p2aArgs 2>&1 | Tee-Object -FilePath $logOut -Append
Write-Output "=== P2A-256 EVAL DONE exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

# P2A-256 DINO
Write-Output ""
Write-Output "=== P2A-256 DINO START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
$dinoP2AOut = "exp\_dino_results\t1_asg_5ep_p2a.json"
if (Test-Path $dinoP2AOut) { Remove-Item $dinoP2AOut -Force }
$dinoP2AArgs = @(
    "_compute_dino.py",
    "--images_dir", "$p2aEvalDir\images",
    "--test_dir", $p2aTestDir,
    "--dataset", "p2a",
    "--output", $dinoP2AOut,
    "--hf_cache", $hfCache,
    "--max_refs", "30"
)
& python @dinoP2AArgs 2>&1 | Tee-Object -FilePath $logOut -Append
Write-Output "=== P2A-256 DINO DONE exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

# ===== R5 =====
$r5TestDir = "I:\datasets\wikiarts20_512_test"
$r5Styles = "Cubism,Expressionism,Pop_Art,Romanticism,Symbolism"
$r5EvalDir = "exp\main_table\r5\full_eval\epoch_0005"

Write-Output ""
Write-Output "=== R5 EVAL START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
if (Test-Path "$r5EvalDir\summary.json") { Remove-Item "$r5EvalDir\summary.json" -Force }
$r5Args = @(
    "-u", "src\utils\run_evaluation.py",
    "--checkpoint", $ckpt,
    "--output", $r5EvalDir,
    "--test_dir", $r5TestDir,
    "--style_subdirs", $r5Styles,
    "--cache_dir", $cacheDir,
    "--clip_hf_cache_dir", $hfCache,
    "--batch_size", "16", "--generation_batch_size", "16", "--metric_batch_size", "16",
    "--target_chunk_size", "1", "--vae_decode_batch_size", "16",
    "--eval_only_lpips_clip_style", "--eval_lpips_chunk_size", "4"
)
& python @r5Args 2>&1 | Tee-Object -FilePath $logOut -Append
Write-Output "=== R5 EVAL DONE exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

# R5 DINO
Write-Output ""
Write-Output "=== R5 DINO START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
$dinoR5Out = "exp\_dino_results\t1_asg_5ep_r5.json"
if (Test-Path $dinoR5Out) { Remove-Item $dinoR5Out -Force }
$dinoR5Args = @(
    "_compute_dino.py",
    "--images_dir", "$r5EvalDir\images",
    "--test_dir", $r5TestDir,
    "--dataset", "wikiart",
    "--output", $dinoR5Out,
    "--hf_cache", $hfCache,
    "--max_refs", "30"
)
& python @dinoR5Args 2>&1 | Tee-Object -FilePath $logOut -Append
Write-Output "=== R5 DINO DONE exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

Write-Output ""
Write-Output "=== ALL MAIN TABLE EVAL COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
