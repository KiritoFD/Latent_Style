# Patch SS2D_Encoder.py - simpler approach replacing just the import line
$file = "I:\Github\Latent_Style\Related_Works\repos\SaMam\ARCHI\SS2D_Encoder.py"

# Backup
Copy-Item $file "$file.bak" -Force

# Read all lines
$lines = Get-Content $file -Encoding UTF8
$patched = $false

for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match '^\s+from mamba_ssm\.ops\.selective_scan_interface import selective_scan_fn\s*$' -and -not $patched) {
        # Check if previous line has "if mamba_from_trion:"
        if ($lines[$i-1] -match 'if mamba_from_trion:') {
            $indent = ($lines[$i] -replace '^(\s+).*', '$1')
            $lines[$i] = "${indent}try:"
            $lines[$i] = $lines[$i] + "`n" + "${indent}    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn"
            $lines[$i] = $lines[$i] + "`n" + "${indent}except ImportError:"
            $lines[$i] = $lines[$i] + "`n" + "${indent}    from ARCHI.selective_scan_torch import selective_scan_fn"
            $patched = $true
            Write-Host "Patched line $i"
        }
    }
}

if ($patched) {
    Set-Content $file -Value $lines -Encoding UTF8
    Write-Host "File saved successfully."
} else {
    Write-Host "No matching pattern found."
}

# Verify
Write-Host "`n=== First 45 lines after patch ==="
Get-Content $file -TotalCount 45
