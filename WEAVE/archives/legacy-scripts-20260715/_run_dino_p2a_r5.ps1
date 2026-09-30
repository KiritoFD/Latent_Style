# Re-run DINO eval for P2A-256 and R5 with fixed _compute_dino.py
$ErrorActionPreference = "Continue"
Set-Location "I:\Github\Latent_Style\SchrodingerBridge"
$env:PYTHONIOENCODING = "utf-8"

$hfCache = "C:\Users\Administrator\.cache\huggingface\hub"
$logOut = "C:\Users\Administrator\logs\dino_rerun.out"

# P2A-256 DINO (with fixed parse_p2a + find_content_p2a)
Write-Output "=== P2A-256 DINO RERUN $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
$dinoP2AOut = "exp\_dino_results\t1_asg_5ep_p2a.json"
if (Test-Path $dinoP2AOut) { Remove-Item $dinoP2AOut -Force }
& python _compute_dino.py `
    --images_dir "exp\main_table\p2a_256\full_eval\epoch_0005\images" `
    --test_dir "I:\datasets\legacy256_overfit50\test" `
    --dataset "p2a" `
    --output $dinoP2AOut `
    --hf_cache $hfCache `
    --max_refs 30 2>&1 | Tee-Object -FilePath $logOut -Append
Write-Output "=== P2A-256 DINO DONE exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

# R5 DINO (with --style_subdirs for Cubism,Expressionism,Pop_Art,Romanticism,Symbolism)
Write-Output ""
Write-Output "=== R5 DINO RERUN $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
$dinoR5Out = "exp\_dino_results\t1_asg_5ep_r5.json"
if (Test-Path $dinoR5Out) { Remove-Item $dinoR5Out -Force }
& python _compute_dino.py `
    --images_dir "exp\main_table\r5\full_eval\epoch_0005\images" `
    --test_dir "I:\datasets\wikiarts20_512_test" `
    --dataset "wikiart" `
    --style_subdirs "Cubism,Expressionism,Pop_Art,Romanticism,Symbolism" `
    --output $dinoR5Out `
    --hf_cache $hfCache `
    --max_refs 30 2>&1 | Tee-Object -FilePath $logOut -Append
Write-Output "=== R5 DINO DONE exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

Write-Output ""
Write-Output "=== ALL DINO RERUN COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
