$procs = Get-Process python -ErrorAction SilentlyContinue
if ($procs) {
    Write-Output "PYTHON_PROCS_FOUND: $($procs.Count)"
    $procs | ForEach-Object { Write-Output "PID=$($_.Id) START=$($_.StartTime) CPU=$($_.CPU)" }
} else {
    Write-Output "NO_PYTHON_PROCS"
}
