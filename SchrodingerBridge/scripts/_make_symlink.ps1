# Create symlink so dataset is accessible via relative path from project root
$target = 'I:\datasets\wikiart_distinct5_samam_512_classview'
$link = 'I:\Github\Latent_Style\Dataset\wikiart_distinct5_samam_512_classview'

if (Test-Path $link) {
    Write-Host "Link already exists: $link"
    $item = Get-Item $link
    Write-Host "Mode: $($item.Mode) Target: $($item.Target) LinkType: $($item.LinkType)"
} else {
    try {
        New-Item -ItemType SymbolicLink -Path $link -Target $target -ErrorAction Stop | Out-Null
        Write-Host "Created symlink: $link -> $target"
    } catch {
        Write-Host "Symlink failed: $_"
        Write-Host "Trying junction..."
        cmd /c mklink /J "$link" "$target"
    }
}

# Verify
if (Test-Path $link) {
    Write-Host "Verify: exists"
    Get-ChildItem $link | Select-Object Name | Format-Table -AutoSize
}
