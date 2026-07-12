$logFile = $args[0]
if (Test-Path $logFile) {
    Get-Content $logFile -Tail 20
} else {
    Write-Output "LOG_NOT_FOUND: $logFile"
}
