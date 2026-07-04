# FC-SB Phase 2 - Final Comprehensive Check
$sshHost = "administrator@100.115.18.62"
$sshPort = "2222"
$remoteBase = "/home/xy/Latent_Style/SchrodingerBridge"
$outputFile = "g:\GitHub\Latent_Style\SchrodingerBridge\phase2_status_report.txt"

Write-Host "=========================================="
Write-Host "FC-SB Phase 2 - Final Comprehensive Check"
Write-Host "$(Get-Date)"
Write-Host "=========================================="

"=" * 70 | Out-File $outputFile
"FC-SB PHASE 2 REMOTE EXPERIMENT STATUS REPORT" | Out-File -FilePath $outputFile -Append
"Check Time: $(Get-Date)" | Out-File -FilePath $outputFile -Append
"=" * 70 | Out-File -FilePath $outputFile -Append

# Section 1: Tmux Status
Write-Host "`n[Section 1] Tmux Session Status"
$r1 = & ssh -p $sshPort $sshHost "bash -c 'tmux list-sessions'" 2>&1
"`n`n### TMUX SESSION STATUS ###" | Out-File -FilePath $outputFile -Append
$r1 | Out-File -FilePath $outputFile -Append
Write-Host $r1

# Section 2: Current Experiment Info
Write-Host "`n[Section 2] Experiment Configuration"`n`n### EXPERIMENT CONFIGURATION ###" | Out-File -FilePath $outputFile -Append
"Experiment Name: p3_remote_10h" | Out-File -FilePath $outputFile -Append
"Config: fc_sb_kernel7 (F3: FC-SB kernel=7, larger fiber projection)" | Out-File -FilePath $outputFile -Append
"Total Epochs: 3" | Out-File -FilePath $outputFile -Append
"Batch Size: 12" | Out-File -FilePath $outputFile -Append
"Learning Rate: 0.0002" | Out-File -FilePath $outputFile -Append
"Two-Stage Training: Enabled" | Out-File -FilePath $outputFile -Append
"Styles: Early_Renaissance, Impressionism, Minimalism, Rococo, Ukiyo_e" | Out-File -FilePath $outputFile -Append
Write-Host "Experiment: p3_remote_10h"
Write-Host "Config: fc_sb_kernel7"

# Section 3: Training Progress
Write-Host "`n[Section 3] Training Progress"
$r3 = & ssh -p $sshPort $sshHost "bash -c 'ls -lht $remoteBase/fc_sb_kernel7/checkpoints/*.pt 2>/dev/null'" 2>&1
"`n`n### CHECKPOINT PROGRESS ###" | Out-File -FilePath $outputFile -Append
$r3 | Out-File -FilePath $outputFile -Append
Write-Host $r3

# Section 4: Evaluation Results  
Write-Host "`n[Section 4] Evaluation Results"
$r4 = & ssh -p $sshPort $sshHost "bash -c 'ls -la $remoteBase/fc_sb_kernel7/checkpoints/full_eval/epoch_0001/'" 2>&1
"`n`n### EVALUATION STATUS ###" | Out-File -FilePath $outputFile -Append
$r4 | Out-File -FilePath $outputFile -Append
Write-Host $r4

# Check for generated images
Write-Host "`nGenerated samples:"
$r4b = & ssh -p $sshPort $sshHost "bash -c 'ls $remoteBase/fc_sb_kernel7/checkpoints/full_eval/epoch_0001/images/ | wc -l'" 2>&1
$imageCount = ($r4b -replace '\s','')
"Generated Images Count: $imageCount" | Out-File -FilePath $outputFile -Append
Write-Host "Images: $imageCount files"

# Section 5: Look for metrics/convergence data
Write-Host "`n[Section 5] Metrics Search"
$r5 = & ssh -p $sshPort $sshHost "bash -c 'find $remoteBase/fc_sb_kernel7/checkpoints -maxdepth 3 -name *summary* -o -name *metrics* -o -name *convergence* 2>/dev/null'" 2>&1
"`n`n### METRICS FILES FOUND ###" | Out-File -FilePath $outputFile -Append
$r5 | Out-File -FilePath $outputFile -Append
Write-Host $r5

# Section 6: Process Status
Write-Host "`n[Section 6] Running Processes"
$r6 = & ssh -p $sshPort $sshHost "bash -c 'ps aux | grep -E python|torch | wc -l'" 2>&1
"`n`n### ACTIVE PROCESSES ###" | Out-File -FilePath $outputFile -Append
"Python/Torch processes running: $r6" | Out-File -FilePath $outputFile -Append
Write-Host "Active processes: $r6"

# Section 7: Disk Usage
Write-Host "`n[Section 7] Resource Usage"
$r7 = & ssh -p $sshPort $sshHost "bash -c 'du -sh $remoteBase/fc_sb_kernel7/'" 2>&1
"`n`n### DISK USAGE ###" | Out-File -FilePath $outputFile -Append
"Experiment directory size: $r7" | Out-File -FilePath $outputFile -Append
Write-Host "Size: $r7"

Write-Host "`n=========================================="
Write-Host "`n=== COMPLETE STATUS REPORT ==="
Write-Host "=========================================="
Get-Content $outputFile
Write-Host "=========================================="
