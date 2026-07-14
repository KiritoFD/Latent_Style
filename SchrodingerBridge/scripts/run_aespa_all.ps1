# Remote: Run AesPA-Net inference on all 3 datasets
$ErrorActionPreference = "Continue"

$aespaRoot = "I:\AesPA-Net"
$vggPath = "$aespaRoot\baseline_checkpoints\vgg_normalised_conv5_1.pth"
$decPath = "$aespaRoot\train_results\aespa\log\dec_model_.pth"
$transPath = "$aespaRoot\train_results\aespa\log\transformer_model_.pth"

# Dataset configurations
$datasets = @(
    @{
        name = "D5-512"
        test_dir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
        style_names = "Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e"
        output_dir = "I:\exp_baselines\aespa\d5_512\images"
        imsize = 512
    },
    @{
        name = "P2A-256"
        test_dir = "I:\datasets\legacy256_overfit50\test"
        style_names = "cezanne,Hayao,monet,photo,vangogh"
        output_dir = "I:\exp_baselines\aespa\p2a_256\images"
        imsize = 256
    },
    @{
        name = "R5-WikiArt"
        test_dir = "I:\datasets\wikiarts20_512_test"
        style_names = "Cubism,Expressionism,Pop_Art,Romanticism,Symbolism"
        output_dir = "I:\exp_baselines\aespa\r5_wikiart\images"
        imsize = 512
    }
)

Set-Location $aespaRoot

foreach ($ds in $datasets) {
    Write-Output ""
    Write-Output "=== Running AesPA-Net on $($ds.name) ==="
    Write-Output "Started: $(Get-Date)"

    $outDir = $ds.output_dir
    $outParent = Split-Path $outDir -Parent
    if (-not (Test-Path $outParent)) { New-Item -ItemType Directory -Force -Path $outParent | Out-Null }

    python run_aespa_inference.py `
        --test_dir $ds.test_dir `
        --output_dir $outDir `
        --style_names $ds.style_names `
        --aespa_root $aespaRoot `
        --vgg_path $vggPath `
        --dec_path $decPath `
        --trans_path $transPath `
        --num_src 30 `
        --imsize $ds.imsize

    Write-Output "Finished $($ds.name): $(Get-Date), exit=$LASTEXITCODE"
}

Write-Output ""
Write-Output "=== ALL DATASETS DONE ==="
