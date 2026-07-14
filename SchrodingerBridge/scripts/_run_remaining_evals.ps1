# Run all remaining CLIP/LPIPS + DINO evals for StyTR-2 and AesPA-Net
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$hfCache = "C:\Users\Administrator\.cache\huggingface\hub"
$logOut = "C:\Users\Administrator\logs\remaining_evals.out"
$python = "C:\Program Files\Python312\python.exe"

# Dataset configs
$dsConfigs = @{
    "d5_512" = @{
        test_dir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
        style_names = "Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e"
        dataset_type = "wikiart"
        style_subdirs = ""
    }
    "p2a_256" = @{
        test_dir = "I:\datasets\legacy256_overfit50\test"
        style_names = "cezanne,Hayao,monet,photo,vangogh"
        dataset_type = "p2a"
        style_subdirs = ""
    }
    "r5_wikiart" = @{
        test_dir = "I:\datasets\wikiarts20_512_test"
        style_names = "Cubism,Expressionism,Pop_Art,Romanticism,Symbolism"
        dataset_type = "wikiart"
        style_subdirs = "Cubism,Expressionism,Pop_Art,Romanticism,Symbolism"
    }
}

# DINO output naming uses hyphen format
$dinoNameMap = @{
    "d5_512" = "d5-512"
    "p2a_256" = "p2a-256"
    "r5_wikiart" = "r5-wikiart"
}

$methods = @("stytr2", "aespa")
$dinoOutDir = "I:\Github\Latent_Style\SchrodingerBridge\state\dino"
$clipOutBase = "I:\exp_baselines"

Write-Output "=== REMAINING EVALS START: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Tee-Object -FilePath $logOut

foreach ($method in $methods) {
    foreach ($dsName in $dsConfigs.Keys) {
        $imgDir = "$clipOutBase\$method\$dsName\images"
        $dsCfg = $dsConfigs[$dsName]
        $dsNameH = $dinoNameMap[$dsName]

        Write-Output "" | Tee-Object -FilePath $logOut -Append
        Write-Output "=== $method / $dsName START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Tee-Object -FilePath $logOut -Append

        if (-not (Test-Path $imgDir)) {
            Write-Output "  SKIP: image dir not found: $imgDir" | Tee-Object -FilePath $logOut -Append
            continue
        }
        $imgCount = (Get-ChildItem $imgDir -File).Count
        Write-Output "  Found $imgCount images in $imgDir" | Tee-Object -FilePath $logOut -Append
        if ($imgCount -eq 0) {
            Write-Output "  SKIP: no images" | Tee-Object -FilePath $logOut -Append
            continue
        }

        # CLIP-S/LPIPS eval (skip if summary exists)
        $clipOutDir = "$clipOutBase\$method\$dsName\eval"
        $clipSummary = "$clipOutDir\clip_lpips_summary.json"
        if (-not (Test-Path $clipSummary)) {
            if (-not (Test-Path $clipOutDir)) { New-Item -ItemType Directory -Force -Path $clipOutDir | Out-Null }
            Write-Output "  Running CLIP-S/LPIPS eval..." | Tee-Object -FilePath $logOut -Append
            & $python tools\eval_clip_lpips_other5.py `
                --gen-dir $imgDir `
                --test-dir $dsCfg.test_dir `
                --output-dir $clipOutDir `
                --style-names $dsCfg.style_names `
                --num-src 30 `
                --clip-local-dir "nonexistent" `
                --clip-cache-dir $hfCache `
                --batch-size 8 2>&1 | Tee-Object -FilePath $logOut -Append
            Write-Output "  CLIP-S/LPIPS done exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Tee-Object -FilePath $logOut -Append
        } else {
            Write-Output "  CLIP/LPIPS summary exists, skip" | Tee-Object -FilePath $logOut -Append
        }

        # DINO eval (always re-run with fixed parse_p2a)
        $dinoOut = "$dinoOutDir\${method}_$dsNameH.json"
        if (Test-Path $dinoOut) { Remove-Item $dinoOut -Force }
        Write-Output "  Running DINO eval..." | Tee-Object -FilePath $logOut -Append
        $dinoArgs = @(
            "_compute_dino.py",
            "--images_dir", $imgDir,
            "--test_dir", $dsCfg.test_dir,
            "--dataset", $dsCfg.dataset_type,
            "--output", $dinoOut,
            "--hf_cache", $hfCache,
            "--max_refs", "30"
        )
        if ($dsCfg.style_subdirs) {
            $dinoArgs += @("--style_subdirs", $dsCfg.style_subdirs)
        }
        & $python @dinoArgs 2>&1 | Tee-Object -FilePath $logOut -Append
        Write-Output "  DINO done exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Tee-Object -FilePath $logOut -Append

        Write-Output "=== $method / $dsName DONE ===" | Tee-Object -FilePath $logOut -Append
    }
}

Write-Output "" | Tee-Object -FilePath $logOut -Append
Write-Output "=== ALL REMAINING EVALS COMPLETE: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Tee-Object -FilePath $logOut -Append
