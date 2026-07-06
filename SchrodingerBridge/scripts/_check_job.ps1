# Check PowerShell job status on remote
$script = @'
# Read job ID
$jobIdFile = "I:\Github\Latent_Style\SchrodingerBridge\logs\baseline_wikiarts15_jobid.txt"
if (Test-Path $jobIdFile) {
    $jobId = (Get-Content $jobIdFile -Raw).Trim()
    Write-Output "Job ID from file: $jobId"

    $job = Get-Job -Id ([int]$jobId) -ErrorAction SilentlyContinue
    if ($job) {
        Write-Output "Job State: $($job.State)"
        Write-Output "Job Name: $($job.Name)"
        Write-Output ""
        Write-Output "=== Job output (last 50 lines) ==="
        Receive-Job -Id $job.Id -Keep 2>&1 | Select-Object -Last 50
    } else {
        Write-Output "Job not found with ID $jobId"
        Write-Output ""
        Write-Output "=== All jobs ==="
        Get-Job | Select-Object Id,Name,State | Format-Table -AutoSize
    }
} else {
    Write-Output "Job ID file not found"
}

Write-Output ""
Write-Output "=== check if _eval_baselines_wikiarts15.ps1 is actually running ==="
Get-WmiObject Win32_Process -Filter "Name='powershell.exe' OR Name='python.exe'" -ErrorAction SilentlyContinue | Select-Object ProcessId,Name,CommandLine | Format-List
'@

$bytes = [System.Text.Encoding]::Unicode.GetBytes($script)
$b64 = [Convert]::ToBase64String($bytes)

ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "powershell -EncodedCommand $b64"
