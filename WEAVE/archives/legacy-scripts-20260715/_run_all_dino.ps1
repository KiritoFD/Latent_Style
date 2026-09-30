$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"

# Clear __pycache__
Get-ChildItem -Path "src" -Filter "__pycache__" -Directory -Recurse | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

$testDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$hfCache = "I:\Github\Latent_Style\SchrodingerBridge\exp\eval_cache\hf"

$experiments = @(
    @{name="refactor_minimal_baseline"; dir="exp\refactor_minimal_baseline\full_eval\epoch_0005\images"},
    @{name="wo_asg"; dir="exp\ablation_destructive\wo_asg\full_eval\epoch_0005\images"},
    @{name="wo_endpoint_adain"; dir="exp\ablation_destructive\wo_endpoint_adain\full_eval\epoch_0005\images"},
    @{name="wo_flow"; dir="exp\ablation_destructive\wo_flow\full_eval\epoch_0005\images"},
    @{name="wo_spectral_ode"; dir="exp\ablation_destructive\wo_spectral_ode\full_eval\epoch_0005\images"},
    @{name="wo_wavelet"; dir="exp\ablation_destructive\wo_wavelet\full_eval\epoch_0005\images"}
)

foreach ($exp in $experiments) {
    $name = $exp.name
    $imgDir = $exp.dir
    $outFile = "I:\Github\Latent_Style\SchrodingerBridge\state\dino\D5-512__$name.json"
    if (Test-Path $outFile) {
        Write-Output "SKIP $name (already exists)"
        continue
    }
    if (-not (Test-Path $imgDir)) {
        Write-Output "SKIP $name (images dir not found: $imgDir)"
        continue
    }
    Write-Output "`n=== DINO eval: $name ==="
    python _compute_dino.py --images_dir $imgDir --test_dir $testDir --dataset wikiart --output $outFile --hf_cache $hfCache --max_refs 30 *>&1 | Tee-Object -FilePath "C:\Users\Administrator\logs\dino_$name.log"
    Write-Output "DINO_EXIT_$name=$LASTEXITCODE"
}

Write-Output "`n=== ALL DINO EVAL DONE ==="
