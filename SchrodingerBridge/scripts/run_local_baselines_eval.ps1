# Local evaluation of StyTR-2 and AesPA-Net on local GPU
$ErrorActionPreference = "Continue"
Set-Location "G:\GitHub\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$hfCache = "$env:USERPROFILE\.cache\huggingface\hub"
$logOut = "G:\GitHub\Latent_Style\SchrodingerBridge\logs\local_baselines_eval.out"
$logDir = Split-Path $logOut -Parent
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Force -Path $logDir | Out-Null }

# Dataset configs (LOCAL paths)
$dsConfigs = @{
    "d5_512" = @{
        test_dir = "G:\GitHub\Latent_Style\SchrodingerBridge\datasets\wikiart5_test\test"
        style_names = "Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e"
        dataset_type = "wikiart"
        style_subdirs = ""
    }
    "p2a_256" = @{
        test_dir = "G:\GitHub\Latent_Style\SchrodingerBridge\datasets\p2a_test\test"
        style_names = "cezanne,Hayao,monet,photo,vangogh"
        dataset_type = "p2a"
        style_subdirs = ""
    }
    "r5_wikiart" = @{
        test_dir = "G:\GitHub\Latent_Style\SchrodingerBridge\datasets\wikiart20_test"
        style_names = "Cubism,Expressionism,Pop_Art,Romanticism,Symbolism"
        dataset_type = "wikiart"
        style_subdirs = "Cubism,Expressionism,Pop_Art,Romanticism,Symbolism"
    }
}

$dinoNameMap = @{
    "d5_512" = "d5-512"
    "p2a_256" = "p2a-256"
    "r5_wikiart" = "r5-wikiart"
}

$methods = @("stytr2", "aespa")
$dinoOutDir = "G:\GitHub\Latent_Style\SchrodingerBridge\state\dino"
if (-not (Test-Path $dinoOutDir)) { New-Item -ItemType Directory -Force -Path $dinoOutDir | Out-Null }
$clipOutBase = "G:\GitHub\Latent_Style\SchrodingerBridge\exp_baselines"

Write-Output "=== LOCAL BASELINES EVAL START: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Tee-Object -FilePath $logOut

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

        # CLIP-S/LPIPS eval
        $clipOutDir = "$clipOutBase\$method\$dsName\eval"
        $clipSummary = "$clipOutDir\clip_lpips_summary.json"
        if (-not (Test-Path $clipSummary)) {
            if (-not (Test-Path $clipOutDir)) { New-Item -ItemType Directory -Force -Path $clipOutDir | Out-Null }
            Write-Output "  Running CLIP-S/LPIPS eval..." | Tee-Object -FilePath $logOut -Append
            & python tools\eval_clip_lpips_other5.py `
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

        # DINO eval
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
        & python @dinoArgs 2>&1 | Tee-Object -FilePath $logOut -Append
        Write-Output "  DINO done exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Tee-Object -FilePath $logOut -Append

        Write-Output "=== $method / $dsName DONE ===" | Tee-Object -FilePath $logOut -Append
    }
}

Write-Output "" | Tee-Object -FilePath $logOut -Append
Write-Output "=== ALL LOCAL BASELINES EVAL COMPLETE: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Tee-Object -FilePath $logOut -Append
