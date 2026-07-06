# Launch master_pipeline in background via Start-Process
$ErrorActionPreference = "Continue"

$REPO = "I:\Github\Latent_Style\SchrodingerBridge"
$script = "$REPO\scripts\_master_pipeline.ps1"
$masterLog = "$REPO\logs\master_pipeline.log"

# Kill any old instances (be careful: only those running our script)
Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
    Where-Object { $_.CommandLine -like "*_master_pipeline.ps1*" } |
    ForEach-Object {
        Write-Host "Stopping old PID $($_.ProcessId)"
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
Start-Sleep -Seconds 2

# Launch as detached background process
"=== LAUNCH $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss') ===" | Out-File $masterLog -Encoding utf8

$proc = Start-Process -FilePath "powershell.exe" `
    -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File",$script `
    -WindowStyle Hidden `
    -PassThru

Write-Host "Launched PID=$($proc.Id)"
Start-Sleep -Seconds 8

# Confirm running
Get-CimInstance Win32_Process -Filter "Name='powershell.exe' OR Name='python.exe'" |
    Select-Object ProcessId, Name, @{N='Start';E={$_.CreationDate}}, CommandLine |
    Format-List
