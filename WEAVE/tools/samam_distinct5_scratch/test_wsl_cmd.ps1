Write-Host "=== TEST 1: Simple echo ==="
$out = &wsl -d Ubuntu-22.04 -e echo "HELLO" 2>&1
Write-Host "Output: $out"
Write-Host "ExitCode: $LASTEXITCODE"

Write-Host ""
Write-Host "=== TEST 2: bash sleep 5 ==="
$startTime = Get-Date
$out = &wsl -d Ubuntu-22.04 -e bash -c "sleep 5; echo DONE" 2>&1
$endTime = Get-Date
Write-Host "Output: $out"
Write-Host "ExitCode: $LASTEXITCODE"
Write-Host "Duration: $($endTime - $startTime)"

Write-Host ""
Write-Host "=== TEST 3: bash while loop ==="
$startTime = Get-Date
$out = &wsl -d Ubuntu-22.04 -e bash -c 'while true; do sleep 3600; done' 2>&1
$endTime = Get-Date
Write-Host "Output: $out"
Write-Host "ExitCode: $LASTEXITCODE"
Write-Host "Duration: $($endTime - $startTime)"
