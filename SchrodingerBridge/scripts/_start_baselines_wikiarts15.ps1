# Start the wikiarts-15 baselines evaluation on remote as a detached job
$script = @'
$repo = "I:\Github\Latent_Style\SchrodingerBridge"
$ps1 = "$repo\scripts\_eval_baselines_wikiarts15.ps1"
$py = "$repo\scripts\gen_trainfree_wikiarts15.py"

# Ensure scripts are in place (they were scp'd to C:\Users\Administrator, copy into repo)
$adminBase = "C:\Users\Administrator"
if (-not (Test-Path "$repo\scripts")) { New-Item -ItemType Directory -Force -Path "$repo\scripts" | Out-Null }
Copy-Item "$adminBase\_eval_baselines_wikiarts15.ps1" $ps1 -Force -ErrorAction SilentlyContinue
Copy-Item "$adminBase\gen_trainfree_wikiarts15.py" $py -Force -ErrorAction SilentlyContinue

# Verify VGG/decoder weights exist for AdaIN/WCT
$modelsDir = "I:\Github\Latent_Style\Related_Works\repos\pytorch-AdaIN\models"
Write-Output "=== models dir ==="
Get-ChildItem $modelsDir -ErrorAction SilentlyContinue | ForEach-Object { Write-Output ("{0}`t{1}" -f $_.Name, $_.Length) }

# Verify test_dir per-style subdirs exist
$testDir = "I:\datasets\wikiarts15_512_test"
Write-Output ""
Write-Output "=== test_dir per-style subdirs ==="
Get-ChildItem $testDir -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    $cnt = (Get-ChildItem $_.FullName -File -ErrorAction SilentlyContinue | Measure-Object).Count
    Write-Output ("{0}`t{1} files" -f $_.Name, $cnt)
}

# Launch the baselines evaluation as a background job
Write-Output ""
Write-Output "=== launching _eval_baselines_wikiarts15.ps1 as background job ==="
$logFile = "$repo\logs\baseline_wikiarts15_launcher.log"
$job = Start-Job -ScriptBlock {
    param($ps1)
    & powershell -ExecutionPolicy Bypass -File $ps1
} -ArgumentList $ps1

$job | Select-Object Id,Name,State,HasMoreData | Format-Table -AutoSize
Write-Output ("Job ID: {0}" -f $job.Id)
Write-Output ("Job State: {0}" -f $job.State)

# Save job ID for later tracking
$job.Id | Out-File "$repo\logs\baseline_wikiarts15_jobid.txt" -Encoding ascii
Write-Output "Job ID saved to $repo\logs\baseline_wikiarts15_jobid.txt"
'@

$bytes = [System.Text.Encoding]::Unicode.GetBytes($script)
$b64 = [Convert]::ToBase64String($bytes)

ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "powershell -EncodedCommand $b64"
