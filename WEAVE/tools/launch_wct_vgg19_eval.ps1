$python = "C:\Program Files\Python312\python.exe"
$evalScript = "I:\GitHub\Latent_Style\SchrodingerBridge\src\utils\run_evaluation.py"
$evalDir = "I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\eval\wct_vgg19"
$testDir = "I:\wikiart_distinct5_samam_512_classview\test"
$logFile = "I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\eval\wct_vgg19_eval.log"
$errFile = "I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\eval\wct_vgg19_eval_err.log"

$args = @(
    $evalScript,
    $evalDir,
    "--reuse_generated",
    "--save_generated_images",
    "--style_subdirs", "Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e",
    "--test_dir", $testDir,
    "--eval_only_lpips_clip_style",
    "--clip_style_idt_baseline", "0.6399"
)

Start-Process -FilePath $python -ArgumentList $args -RedirectStandardOutput $logFile -RedirectStandardError $errFile -NoNewWindow
Write-Host "LAUNCHED"
