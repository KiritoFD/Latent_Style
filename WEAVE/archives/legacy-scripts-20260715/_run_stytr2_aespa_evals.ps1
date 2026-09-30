# Eval script for StyTR-2 and AesPA-Net only (StyleID already evaluated)
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$hfCache = "C:\Users\Administrator\.cache\huggingface\hub"
$logOut = "C:\Users\Administrator\logs\stytr2_aespa_evals.out"
$python = "C:\Program Files\Python312\python.exe"

# Dataset configs
$dsConfigs = @{
    "D5-512" = @{
        test_dir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
        style_names = "Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e"
        dataset_type = "wikiart"
        style_subdirs = ""
    }
    "P2A-256" = @{
        test_dir = "I:\datasets\legacy256_overfit50\test"
        style_names = "cezanne,Hayao,monet,photo,vangogh"
        dataset_type = "p2a"
        style_subdirs = ""
    }
    "R5-WikiArt" = @{
        test_dir = "I:\datasets\wikiarts20_512_test"
        style_names = "Cubism,Expressionism,Pop_Art,Romanticism,Symbolism"
        dataset_type = "wikiart"
        style_subdirs = "Cubism,Expressionism,Pop_Art,Romanticism,Symbolism"
    }
}

$methods = @(
    @{ name = "stytr2"; dirs = @{
        "D5-512" = "I:\exp_baselines\stytr2\d5_512\images"
        "P2A-256" = "I:\exp_baselines\stytr2\p2a_256\images"
        "R5-WikiArt" = "I:\exp_baselines\stytr2\r5_wikiart\images"
    }}
    @{ name = "aespa"; dirs = @{
        "D5-512" = "I:\exp_baselines\aespa\d5_512\images"
        "P2A-256" = "I:\exp_baselines\aespa\p2a_256\images"
        "R5-WikiArt" = "I:\exp_baselines\aespa\r5_wikiart\images"
    }}
)

$dinoOutDir = "I:\Github\Latent_Style\SchrodingerBridge\state\dino"
$clipOutBase = "I:\exp_baselines"

Write-Output "=== STYTR2+AESPA EVAL START: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Tee-Object -FilePath $logOut

foreach ($method in $methods) {
    $mname = $method.name
    foreach ($dsName in $method.dirs.Keys) {
        $imgDir = $method.dirs[$dsName]
        $dsCfg = $dsConfigs[$dsName]

        Write-Output "" | Tee-Object -FilePath $logOut -Append
        Write-Output "=== $mname / $dsName EVAL START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Tee-Object -FilePath $logOut -Append

        if (-not (Test-Path $imgDir)) {
            Write-Output "  SKIP: image dir not found: $imgDir" | Tee-Object -FilePath $logOut -Append
            continue
        }
        $imgCount = (Get-ChildItem $imgDir -File).Count
        Write-Output "  Found $imgCount images in $imgDir" | Tee-Object -FilePath $logOut -Append
        if ($imgCount -eq 0) {
            Write-Output "  SKIP: no images found" | Tee-Object -FilePath $logOut -Append
            continue
        }

        # CLIP-S/LPIPS eval
        $clipOutDir = "$clipOutBase\$mname\$($dsName.ToLower())\eval"
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

        # DINO eval
        $dinoOut = "$dinoOutDir\${mname}_$($dsName.ToLower()).json"
        if (-not (Test-Path $dinoOutDir)) { New-Item -ItemType Directory -Force -Path $dinoOutDir | Out-Null }

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

        Write-Output "=== $mname / $dsName EVAL COMPLETE ===" | Tee-Object -FilePath $logOut -Append
    }
}

Write-Output "" | Tee-Object -FilePath $logOut -Append
Write-Output "=== ALL STYTR2+AESPA EVALS COMPLETE: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Tee-Object -FilePath $logOut -Append
