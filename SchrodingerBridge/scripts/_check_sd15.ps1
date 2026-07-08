# Check SD1.5 local path
$sdPath = "C:\Users\Administrator\.cache\huggingface\hub\models--runwayml--stable-diffusion-v1-5\snapshots\451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
if (Test-Path $sdPath) {
    Write-Output "SD1.5 snapshot found"
    Get-ChildItem $sdPath | Select-Object Name
} else {
    Write-Output "SD1.5 snapshot NOT found at expected path"
    # Search for it
    Write-Output "Searching for SD1.5 cache..."
    $hub = "C:\Users\Administrator\.cache\huggingface\hub"
    if (Test-Path $hub) {
        Get-ChildItem $hub -Directory -Filter "*stable*" | ForEach-Object {
            Write-Output "Found: $($_.Name)"
            $snapDir = Join-Path $_.FullName "snapshots"
            if (Test-Path $snapDir) {
                Get-ChildItem $snapDir -Directory | ForEach-Object { Write-Output "  snapshot: $($_.Name)" }
            }
        }
    }
}
