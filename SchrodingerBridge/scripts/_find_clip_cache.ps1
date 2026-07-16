# Find CLIP model cache on remote
Write-Host "=== Search for CLIP model cache ==="
$cacheDir = 'I:\Github\Latent_Style\WEAVE\eval_cache'
Get-ChildItem $cacheDir -Directory -Filter "models--*" -ErrorAction SilentlyContinue | Select-Object Name | Format-Table -AutoSize

Write-Host "=== Search other common cache locations ==="
$locations = @(
    'C:\Users\Administrator\.cache\huggingface\hub',
    'C:\Users\Administrator\AppData\Local\huggingface\hub',
    'C:\Users\Administrator\.cache\torch\hub'
)
foreach ($loc in $locations) {
    if (Test-Path $loc) {
        Write-Host "FOUND: $loc"
        Get-ChildItem $loc -Directory -Filter "models--*" -ErrorAction SilentlyContinue | Select-Object Name | Format-Table -AutoSize
    }
}

Write-Host "=== Search for clip in eval_cache ==="
Get-ChildItem $cacheDir -Recurse -Directory -Filter "*clip*" -ErrorAction SilentlyContinue -Depth 2 | Select-Object FullName | Format-Table -AutoSize
