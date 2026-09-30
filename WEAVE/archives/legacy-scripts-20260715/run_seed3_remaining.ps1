# 3-seed experiment: run remaining evaluations (seed42_r5 + seed123/seed2024 all)
# Robust: each eval is a separate python process, skip already-done, log exit codes
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$logDir = "C:\Users\Administrator\logs\seed3"
$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "C:\Users\Administrator\.cache\huggingface\hub"

# Dataset configs
$datasets = @(
    @{name="d5";  testDir="I:\datasets\wikiart_distinct5_samam_512_classview\test"; styles="Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e"; dataset="wikiart"},
    @{name="p2a"; testDir="I:\datasets\legacy256_overfit50\test";                    styles="cezanne,Hayao,monet,photo,vangogh";                       dataset="p2a"},
    @{name="r5";  testDir="I:\datasets\wikiarts20_512_test";                         styles="Cubism,Expressionism,Pop_Art,Romanticism,Symbolism";      dataset="wikiart"; styleSubdirs=$true}
)

$seeds = @(
    @{name="seed42";   ckpt="exp\seed3\seed42_b96\epoch_0005.pt"},
    @{name="seed123";  ckpt="exp\seed3\seed123_b96\epoch_0005.pt"},
    @{name="seed2024"; ckpt="exp\seed3\seed2024_b96\epoch_0005.pt"}
)

# Already done: seed42_d5 (CLIP+DINO), seed42_p2a (CLIP+DINO)
$skipEval = @("seed42_d5", "seed42_p2a")

$summaryLog = "$logDir\remaining_summary.log"
"=== Remaining evals started: $(Get-Date) ===" | Out-File -FilePath $summaryLog -Encoding utf8

foreach ($seed in $seeds) {
    foreach ($ds in $datasets) {
        $tag = "$($seed.name)_$($ds.name)"
        $evalDir = "exp\seed3\$($seed.name)_$($ds.name)_eval\full_eval\epoch_0005"
        $summaryPath = "$evalDir\summary.json"
        $dinoPath = "exp\seed3\_dino\$tag.json"

        # ===== STEP 1: CLIP/LPIPS evaluation =====
        if ($tag -in $skipEval) {
            "[SKIP] $tag CLIP/LPIPS already done" | Tee-Object -FilePath $summaryLog -Append
        } elseif (Test-Path $summaryPath) {
            "[SKIP] $tag CLIP/LPIPS summary.json exists" | Tee-Object -FilePath $summaryLog -Append
        } else {
            # Clean up incomplete eval dir
            $evalParent = "exp\seed3\$($seed.name)_$($ds.name)_eval"
            if (Test-Path $evalParent) {
                Remove-Item -Recurse -Force $evalParent -ErrorAction SilentlyContinue
            }

            $evalLog = "$logDir\$tag`_eval.out"
            "[RUN ] $tag CLIP/LPIPS -> $evalLog" | Tee-Object -FilePath $summaryLog -Append

            $evalArgs = @(
                "-u", "src\utils\run_evaluation.py",
                "--checkpoint", $seed.ckpt,
                "--output", "$evalDir",
                "--test_dir", $ds.testDir,
                "--style_subdirs", $ds.styles,
                "--cache_dir", $cacheDir,
                "--clip_hf_cache_dir", $hfCache,
                "--batch_size", "8",
                "--generation_batch_size", "8",
                "--metric_batch_size", "4",
                "--target_chunk_size", "2",
                "--vae_decode_batch_size", "16",
                "--max_src_samples", "30",
                "--max_ref_compare", "24",
                "--max_ref_cache", "80",
                "--ref_feature_batch_size", "8",
                "--eval_only_lpips_clip_style",
                "--eval_lpips_chunk_size", "4"
            )
            $proc = Start-Process -FilePath "python" -ArgumentList $evalArgs -NoNewWindow -PassThru -RedirectStandardOutput $evalLog -RedirectStandardError "$evalLog.err"
            $proc | Wait-Process
            $code = $proc.ExitCode
            "[DONE] $tag CLIP/LPIPS exit=$code" | Tee-Object -FilePath $summaryLog -Append
            if (Test-Path "$evalLog.err") {
                Get-Content "$evalLog.err" -Tail 5 | Out-File -FilePath $summaryLog -Append -Encoding utf8
            }
        }

        # ===== STEP 2: DINO evaluation =====
        if (Test-Path $dinoPath) {
            "[SKIP] $tag DINO already done" | Tee-Object -FilePath $summaryLog -Append
        } else {
            $dinoLog = "$logDir\$tag`_dino.out"
            "[RUN ] $tag DINO -> $dinoLog" | Tee-Object -FilePath $summaryLog -Append

            $imagesDir = "$evalDir\images"
            $dinoArgs = @(
                "_compute_dino.py",
                "--images_dir", $imagesDir,
                "--test_dir", $ds.testDir,
                "--dataset", $ds.dataset,
                "--output", $dinoPath,
                "--hf_cache", $hfCache,
                "--max_refs", "30"
            )
            if ($ds.styleSubdirs) {
                $dinoArgs += @("--style_subdirs", $ds.styles)
            }
            $proc = Start-Process -FilePath "python" -ArgumentList $dinoArgs -NoNewWindow -PassThru -RedirectStandardOutput $dinoLog -RedirectStandardError "$dinoLog.err"
            $proc | Wait-Process
            $code = $proc.ExitCode
            "[DONE] $tag DINO exit=$code" | Tee-Object -FilePath $summaryLog -Append
            if (Test-Path "$dinoLog.err") {
                Get-Content "$dinoLog.err" -Tail 5 | Out-File -FilePath $summaryLog -Append -Encoding utf8
            }
        }
    }
}

"=== All remaining evals completed: $(Get-Date) ===" | Tee-Object -FilePath $summaryLog -Append
