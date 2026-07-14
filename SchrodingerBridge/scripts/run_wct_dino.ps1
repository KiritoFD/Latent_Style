# Run DINO for latent_wct p2a_256 and r5_wikiart (750 images each)
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$logDir = "C:\Users\Administrator\logs\wct_dino"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$hfCache = "C:\Users\Administrator\.cache\huggingface\hub"

# P2A: 5 wikiart styles from wikiarts15_256_test
$p2aStyles = "Abstract_Expressionism,Art_Nouveau_Modern,Baroque,Color_Field_Painting,Cubism"
$p2aArgs = @(
    "_compute_dino.py",
    "--images_dir", "exp\latent_wct_baseline\p2a_256\images_750",
    "--test_dir", "I:\datasets\wikiarts15_256_test",
    "--dataset", "wikiart",
    "--output", "exp\_dino_results\latent_wct_p2a.json",
    "--hf_cache", $hfCache,
    "--max_refs", "30",
    "--style_subdirs", $p2aStyles
)
$p2aLog = "$logDir\p2a_dino.out"
Write-Host "[RUN] P2A DINO -> $p2aLog"
$proc = Start-Process -FilePath "python" -ArgumentList $p2aArgs -NoNewWindow -PassThru -RedirectStandardOutput $p2aLog -RedirectStandardError "$p2aLog.err"
$proc | Wait-Process
Write-Host "[DONE] P2A DINO exit=$($proc.ExitCode)"

# R5: 5 wikiart styles from wikiarts20_512_test
$r5Styles = "Cubism,Expressionism,Pop_Art,Romanticism,Symbolism"
$r5Args = @(
    "_compute_dino.py",
    "--images_dir", "exp\latent_wct_baseline\r5_wikiart\images_750",
    "--test_dir", "I:\datasets\wikiarts20_512_test",
    "--dataset", "wikiart",
    "--output", "exp\_dino_results\latent_wct_r5.json",
    "--hf_cache", $hfCache,
    "--max_refs", "30",
    "--style_subdirs", $r5Styles
)
$r5Log = "$logDir\r5_dino.out"
Write-Host "[RUN] R5 DINO -> $r5Log"
$proc = Start-Process -FilePath "python" -ArgumentList $r5Args -NoNewWindow -PassThru -RedirectStandardOutput $r5Log -RedirectStandardError "$r5Log.err"
$proc | Wait-Process
Write-Host "[DONE] R5 DINO exit=$($proc.ExitCode)"

Write-Host "=== All DINO evals completed ==="
