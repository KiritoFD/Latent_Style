$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
Remove-Item -Path "eval_stage1_log.txt" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "eval_stage1_err.txt" -Force -ErrorAction SilentlyContinue

# Delete old task if exists
schtasks /Delete /TN "sb_eval_stage1" /F 2>$null | Out-Null

$pythonExe = (Get-Command python.exe).Source
$workDir = "I:\Github\Latent_Style\SchrodingerBridge"
$ckpt = "$workDir\exp\clean_base_v2\epoch_0010.pt"
$outDir = "$workDir\exp\clean_base_v2\full_eval\epoch_0010"
$testDir = "I:\wikiart_distinct5_samam_512_classview\test"
$cacheDir = "I:/Github/Latent_Style/eval_cache"
$hfCacheDir = "I:/Github/Latent_Style/eval_cache/hf"

$script = "`"$pythonExe`" -u `"$workDir\src\utils\run_evaluation.py`" --checkpoint `"$ckpt`" --output `"$outDir`" --test_dir `"$testDir`" --cache_dir `"$cacheDir`" --clip_hf_cache_dir `"$hfCacheDir`" --batch_size 16 --num_steps 8 --max_src_samples 30 --max_ref_compare 30 --max_ref_cache 30 --ref_feature_batch_size 16 --target_chunk_size 1 1> `"$workDir\eval_stage1_log.txt`" 2> `"$workDir\eval_stage1_err.txt`""

$batPath = "$workDir\_run_eval.bat"
Set-Content -Path $batPath -Value $script -Encoding ASCII

schtasks /Create /TN "sb_eval_stage1" /TR $batPath /SC ONCE /ST 00:00 /RL HIGHEST /F
schtasks /Run /TN "sb_eval_stage1"

Write-Output "Eval task started at $(Get-Date)"
Start-Sleep -Seconds 10
Write-Output "=== Process status ==="
tasklist | findstr python
Write-Output ""
Write-Output "=== ERR (last 10 lines) ==="
Get-Content "eval_stage1_err.txt" -ErrorAction SilentlyContinue -Tail 10
