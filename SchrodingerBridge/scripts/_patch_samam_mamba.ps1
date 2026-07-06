# Patch SS2D_Encoder.py to make mamba_ssm import optional
$file = "I:\Github\Latent_Style\Related_Works\repos\SaMam\ARCHI\SS2D_Encoder.py"

# Backup
Copy-Item $file "$file.bak" -Force

# Read content
$content = Get-Content $file -Raw -Encoding UTF8

# Replace the import block
$old = @'
        if mamba_from_trion:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        else:
            from ARCHI.selective_scan_torch import selective_scan_fn
'@

$new = @'
        if mamba_from_trion:
            try:
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
            except ImportError:
                from ARCHI.selective_scan_torch import selective_scan_fn
        else:
            from ARCHI.selective_scan_torch import selective_scan_fn
'@

if ($content.Contains($old)) {
    $content = $content.Replace($old, $new)
    Set-Content $file -Value $content -Encoding UTF8 -NoNewline
    Write-Host "Patched $file successfully."
} else {
    Write-Host "Pattern not found in $file. Checking current content..."
    Get-Content $file | Select-Object -First 40
}
