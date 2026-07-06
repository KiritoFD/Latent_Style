# Check current state of SS2D files
$files = @(
    "I:\Github\Latent_Style\Related_Works\repos\SaMam\ARCHI\SS2D_Encoder.py",
    "I:\Github\Latent_Style\Related_Works\repos\SaMam\ARCHI\SAVSSM\SS2D_Decoder.py"
)
foreach ($f in $files) {
    Write-Host "=== $f ==="
    if (Test-Path $f) {
        $content = Get-Content $f -Raw
        $matches = [regex]::Matches($content, "from mamba_ssm")
        Write-Host "  mamba_ssm import count: $($matches.Count)"
        $exceptCount = [regex]::Matches($content, "except ImportError").Count
        Write-Host "  except ImportError count: $exceptCount"
        # Show context
        $lines = Get-Content $f
        for ($i = 0; $i -lt $lines.Count; $i++) {
            if ($lines[$i] -match "mamba_ssm|except ImportError|try:") {
                Write-Host "  $($i+1): $($lines[$i])"
            }
        }
    } else {
        Write-Host "  NOT FOUND"
    }
    Write-Host ""
}
