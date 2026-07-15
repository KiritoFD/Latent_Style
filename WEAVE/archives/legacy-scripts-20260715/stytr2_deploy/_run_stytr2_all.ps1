# Remote: Run StyTR-2 inference on all 3 datasets
$ErrorActionPreference = "Continue"

# Setup experiments directory with weights
$expDir = "I:\StyTR2\experiments"
if (-not (Test-Path $expDir)) { New-Item -ItemType Directory -Force -Path $expDir | Out-Null }

# Copy/rename weights to expected locations
if (-not (Test-Path "$expDir\vgg_normalised.pth") -and (Test-Path "I:\stytr2_vgg.pth")) {
    Copy-Item "I:\stytr2_vgg.pth" "$expDir\vgg_normalised.pth"
}
if (-not (Test-Path "$expDir\embedding_iter_160000.pth") -and (Test-Path "I:\stytr2_embedding.pth")) {
    Copy-Item "I:\stytr2_embedding.pth" "$expDir\embedding_iter_160000.pth"
}
if (-not (Test-Path "$expDir\decoder_iter_160000.pth") -and (Test-Path "I:\stytr2_decoder.pth")) {
    Copy-Item "I:\stytr2_decoder.pth" "$expDir\decoder_iter_160000.pth"
}
if (-not (Test-Path "$expDir\transformer_iter_160000.pth") -and (Test-Path "I:\stytr2_transformer.pth")) {
    Copy-Item "I:\stytr2_transformer.pth" "$expDir\transformer_iter_160000.pth"
}
Write-Output "Weights prepared in $expDir"

# Dataset configurations
$datasets = @(
    @{
        name = "D5-512"
        test_dir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
        style_names = "Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e"
        output_dir = "I:\exp_baselines\stytr2\d5_512\images"
        content_size = 512
        style_size = 512
    },
    @{
        name = "P2A-256"
        test_dir = "I:\datasets\legacy256_overfit50\test"
        style_names = "cezanne,Hayao,monet,photo,vangogh"
        output_dir = "I:\exp_baselines\stytr2\p2a_256\images"
        content_size = 256
        style_size = 256
    },
    @{
        name = "R5-WikiArt"
        test_dir = "I:\datasets\wikiarts20_512_test"
        style_names = "Cubism,Expressionism,Pop_Art,Romanticism,Symbolism"
        output_dir = "I:\exp_baselines\stytr2\r5_wikiart\images"
        content_size = 512
        style_size = 512
    }
)

Set-Location "I:\StyTR2"

foreach ($ds in $datasets) {
    Write-Output ""
    Write-Output "=== Running StyTR-2 on $($ds.name) ==="
    Write-Output "Started: $(Get-Date)"

    $outDir = $ds.output_dir
    $outParent = Split-Path $outDir -Parent
    if (-not (Test-Path $outParent)) { New-Item -ItemType Directory -Force -Path $outParent | Out-Null }

    python run_stytr2_inference.py `
        --test_dir $ds.test_dir `
        --output_dir $outDir `
        --style_names $ds.style_names `
        --vgg "$expDir\vgg_normalised.pth" `
        --decoder_path "$expDir\decoder_iter_160000.pth" `
        --trans_path "$expDir\transformer_iter_160000.pth" `
        --embedding_path "$expDir\embedding_iter_160000.pth" `
        --stytr2_root "I:\StyTR2" `
        --num_src 30 `
        --content_size $ds.content_size `
        --style_size $ds.style_size

    Write-Output "Finished $($ds.name): $(Get-Date), exit=$LASTEXITCODE"
}

Write-Output ""
Write-Output "=== ALL DATASETS DONE ==="
