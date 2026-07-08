$ErrorActionPreference = 'Continue'

Write-Host "=== clean_base_v2 contents ==="
$cb = 'I:\Github\Latent_Style\SchrodingerBridge\exp\clean_base_v2'
if (Test-Path $cb) {
    Get-ChildItem -Path $cb -Recurse -Depth 2 -ErrorAction SilentlyContinue | ForEach-Object {
        $rel = $_.FullName.Substring($cb.Length)
        if ($_.PSIsContainer) {
            Write-Host ("  [DIR] " + $rel)
        } else {
            Write-Host ("  [FILE] " + $rel + " (" + $_.Length + " bytes)")
        }
    }
}

Write-Host ""
Write-Host "=== clean_base_v2_final contents ==="
$cbf = 'I:\Github\Latent_Style\SchrodingerBridge\exp\629_subtractive\clean_base_v2_final'
if (Test-Path $cbf) {
    Get-ChildItem -Path $cbf -Recurse -Depth 2 -ErrorAction SilentlyContinue | ForEach-Object {
        $rel = $_.FullName.Substring($cbf.Length)
        if ($_.PSIsContainer) {
            Write-Host ("  [DIR] " + $rel)
        } else {
            Write-Host ("  [FILE] " + $rel + " (" + $_.Length + " bytes)")
        }
    }
} else {
    Write-Host "  NOT FOUND"
}

Write-Host ""
Write-Host "=== Search for full_eval dirs ==="
Get-ChildItem -Path 'I:\Github\Latent_Style\SchrodingerBridge\exp' -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    $fe = Join-Path $_.FullName 'full_eval'
    if (Test-Path $fe) {
        Write-Host ("  " + $fe)
        $sub = Get-ChildItem -Path $fe -ErrorAction SilentlyContinue
        $sub | Select-Object -First 10 | ForEach-Object {
            if ($_.PSIsContainer) { Write-Host ("    [DIR] " + $_.Name) }
            else { Write-Host ("    [FILE] " + $_.Name + " (" + $_.Length + " bytes)") }
        }
    }
}

Write-Host ""
Write-Host "=== Sample W20 filename parsing test ==="
$sample = 'Abstract_Expressionism_Abstract_Expressionism__aaron-siskind_acolman-1-1955_to_Abstract_Expressionism.png'
Write-Host ("  sample: " + $sample)
$stem = $sample.Substring(0, $sample.Length - 4)
Write-Host ("  stem: " + $stem)
if ($stem -match '__to__') {
    Write-Host "  contains __to__"
    $parts = $stem -split '__to__'
    Write-Host ("  left: " + $parts[0])
    Write-Host ("  tgt: " + $parts[1])
} else {
    Write-Host "  NO __to__ - trying regex"
    if ($stem -match '^(.+?)_(.+?)_to_(.+)$') {
        Write-Host ("  src_style: " + $matches[1])
        Write-Host ("  src_stem: " + $matches[2])
        Write-Host ("  tgt_style: " + $matches[3])
    }
}
