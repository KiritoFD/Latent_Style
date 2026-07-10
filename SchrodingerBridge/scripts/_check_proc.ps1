$pid_to_check = $args[0]
$p = Get-CimInstance Win32_Process -Filter "ProcessId=$pid_to_check" -ErrorAction SilentlyContinue
if ($p) {
    Write-Host "PID: $($p.ProcessId)"
    Write-Host "Name: $($p.Name)"
    Write-Host "CommandLine: $($p.CommandLine)"
} else {
    Write-Host "Process $pid_to_check not found"
}
