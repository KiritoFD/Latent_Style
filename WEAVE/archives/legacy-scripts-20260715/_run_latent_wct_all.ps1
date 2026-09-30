# Latent-WCT baseline on 3 datasets: D5-512, P2A-256, R5-WikiArt
# Usage: powershell -File scripts\_run_latent_wct_all.ps1
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

# Dataset configs: name -> @{test_dir, image_size, styles, output_dir, eval_dir}
$datasets = @(
    @{
        name = "d5_512"
        test_dir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
        image_size = 512
        styles = "Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e"
        output_dir = "I:\Github\Latent_Style\SchrodingerBridge\exp\latent_wct_baseline\d5_512\images"
        eval_dir = "I:\Github\Latent_Style\SchrodingerBridge\exp\latent_wct_baseline\d5_512"
    },
    @{
        name = "p2a_256"
        test_dir = "I:\datasets\wikiarts15_256_test"
        image_size = 256
        styles = "Abstract_Expressionism,Art_Nouveau_Modern,Baroque,Color_Field_Painting,Cubism,Expressionism,Fauvism,High_Renaissance,Mannerism_Late_Renaissance,Naive_Art_Primitivism,Northern_Renaissance,Pop_Art,Post_Impressionism,Romanticism,Symbolism"
        output_dir = "I:\Github\Latent_Style\SchrodingerBridge\exp\latent_wct_baseline\p2a_256\images"
        eval_dir = "I:\Github\Latent_Style\SchrodingerBridge\exp\latent_wct_baseline\p2a_256"
    },
    @{
        name = "r5_wikiart"
        test_dir = "I:\datasets\wikiarts20_512_test"
        image_size = 512
        styles = "Abstract_Expressionism,Art_Nouveau_Modern,Baroque,Color_Field_Painting,Cubism,Early_Renaissance,Expressionism,Fauvism,High_Renaissance,Impressionism,Mannerism_Late_Renaissance,Minimalism,Naive_Art_Primitivism,Northern_Renaissance,Pop_Art,Post_Impressionism,Rococo,Romanticism,Symbolism,Ukiyo_e"
        output_dir = "I:\Github\Latent_Style\SchrodingerBridge\exp\latent_wct_baseline\r5_wikiart\images"
        eval_dir = "I:\Github\Latent_Style\SchrodingerBridge\exp\latent_wct_baseline\r5_wikiart"
    }
)

$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "C:\Users\Administrator\.cache\huggingface\hub"

foreach ($ds in $datasets) {
    $name = $ds.name
    Write-Output "=== [$name] START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

    # Phase 1: Generate images via infer_latent_wct.py
    $genLog = "C:\Users\Administrator\logs\latent_wct_${name}_gen.log"
    Write-Output "  Generating images -> $genLog"
    python -u tools\infer_latent_wct.py `
        --output_dir $ds.output_dir `
        --test_dir $ds.test_dir `
        --styles $ds.styles `
        --image_size $ds.image_size `
        2>&1 | Tee-Object -FilePath $genLog -Append

    # Phase 2: Evaluate CLIP-S + LPIPS
    $evalLog = "C:\Users\Administrator\logs\latent_wct_${name}_eval.log"
    Write-Output "  Evaluating CLIP-S + LPIPS -> $evalLog"
    if (Test-Path "$($ds.eval_dir)\summary.json") { Remove-Item "$($ds.eval_dir)\summary.json" -Force }
    python -u src\utils\run_evaluation.py `
        --output $ds.eval_dir `
        --test_dir $ds.test_dir `
        --cache_dir $cacheDir `
        --clip_hf_cache_dir $hfCache `
        --eval_only_lpips_clip_style `
        --eval_lpips_chunk_size 4 `
        --reuse_generated `
        --style_subdirs $ds.styles `
        --batch_size 16 --metric_batch_size 16 `
        2>&1 | Tee-Object -FilePath $evalLog -Append

    Write-Output "=== [$name] DONE exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
}

Write-Output "ALL DONE."
