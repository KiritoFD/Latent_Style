$ErrorActionPreference = "Continue"
Set-Location "G:\GitHub\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$testDir = "G:\GitHub\Latent_Style\Dataset\wikiart_distinct5_samam_512_classview\test"
$cacheDir = "G:\GitHub\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "G:\GitHub\Latent_Style\SchrodingerBridge\exp\eval_cache\hf"

$exp = "t11e_l_samam_repro"
$ckpt = "exp\t11evo\$exp\epoch_0005.pt"
$evalDir = "exp\t11evo\${exp}_eval_classview\epoch_0005"

Write-Output "=== $exp EVAL-CLASSVIEW START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
$evalStart = Get-Date
python -u src\utils\run_evaluation.py `
    --checkpoint $ckpt `
    --output $evalDir `
    --test_dir $testDir `
    --cache_dir $cacheDir `
    --clip_hf_cache_dir $hfCache `
    --batch_size 2 --generation_batch_size 2 --metric_batch_size 2 `
    --target_chunk_size 2 --vae_decode_batch_size 16 `
    --eval_only_lpips_clip_style --eval_lpips_chunk_size 4 2>&1 | Tee-Object -FilePath "exp\t11evo\${exp}_eval_classview_log.txt"
$evalMin = [math]::Round(((Get-Date) - $evalStart).TotalMinutes, 1)
Write-Output "EVAL: exit=$LASTEXITCODE dur=${evalMin}min"

$summary = Join-Path $evalDir "summary.json"
if (Test-Path $summary) {
    python -c "import json; d=json.load(open(r'$summary','r',encoding='utf-8')); apo=d.get('analysis',{}).get('all_pairs_overview',{}); print(f'clip={apo.get(chr(34)+chr(99)+chr(108)+chr(105)+chr(112)+chr(95)+chr(115)+chr(116)+chr(121)+chr(108)+chr(101)+chr(34),0):.4f} lpips={apo.get(chr(34)+chr(99)+chr(111)+chr(110)+chr(116)+chr(101)+chr(110)+chr(116)+chr(95)+chr(108)+chr(112)+chr(105)+chr(112)+chr(115)+chr(34),0):.4f}')"
}
Write-Output "=== DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
