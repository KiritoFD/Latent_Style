# Re-run only the evaluation phase (images already generated)
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$datasets = @(
    @{ name="d5_512"; test_dir="I:\datasets\wikiart_distinct5_samam_512_classview\test"; styles="Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e" },
    @{ name="p2a_256"; test_dir="I:\datasets\wikiarts15_256_test"; styles="Abstract_Expressionism,Art_Nouveau_Modern,Baroque,Color_Field_Painting,Cubism,Expressionism,Fauvism,High_Renaissance,Mannerism_Late_Renaissance,Naive_Art_Primitivism,Northern_Renaissance,Pop_Art,Post_Impressionism,Romanticism,Symbolism" },
    @{ name="r5_wikiart"; test_dir="I:\datasets\wikiarts20_512_test"; styles="Abstract_Expressionism,Art_Nouveau_Modern,Baroque,Color_Field_Painting,Cubism,Early_Renaissance,Expressionism,Fauvism,High_Renaissance,Impressionism,Mannerism_Late_Renaissance,Minimalism,Naive_Art_Primitivism,Northern_Renaissance,Pop_Art,Post_Impressionism,Rococo,Romanticism,Symbolism,Ukiyo_e" }
)

$cacheDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache"
$hfCache = "C:\Users\Administrator\.cache\huggingface\hub"

foreach ($ds in $datasets) {
    $name = $ds.name
    $evalDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\latent_wct_baseline\$name"
    Write-Output "=== [$name] EVAL START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
    $evalLog = "C:\Users\Administrator\logs\latent_wct_${name}_eval2.log"

    # Clean stale summary
    if (Test-Path "$evalDir\summary.json") { Remove-Item "$evalDir\summary.json" -Force }
    if (Test-Path "$evalDir\images\summary.json") { Remove-Item "$evalDir\images\summary.json" -Force }

    # List count of reusable images
    $imgCount = (Get-ChildItem "$evalDir\images\*_to_*.png" -ErrorAction SilentlyContinue).Count
    Write-Output "  Found $imgCount reusable images in $evalDir\images\"

    python -u src\utils\run_evaluation.py `
        --output $evalDir `
        --test_dir $ds.test_dir `
        --cache_dir $cacheDir `
        --clip_hf_cache_dir $hfCache `
        --eval_only_lpips_clip_style `
        --eval_lpips_chunk_size 4 `
        --reuse_generated `
        --style_subdirs $ds.styles `
        --batch_size 16 --metric_batch_size 16 `
        2>&1 | Tee-Object -FilePath $evalLog -Append

    Write-Output "=== [$name] EVAL DONE exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
}
Write-Output "ALL EVAL DONE."
