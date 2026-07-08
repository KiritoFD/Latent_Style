$ErrorActionPreference = 'Continue'

Write-Host "============================================================"
Write-Host "PART A: Search for CUT checkpoints (any .pt/.pth files)"
Write-Host "============================================================"

# Check baseline_v2/checkpoints/cut structure
$cut_ckpt = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\checkpoints\cut"
if (Test-Path $cut_ckpt) {
    Write-Host "=== $cut_ckpt full tree ==="
    Get-ChildItem $cut_ckpt -Recurse -ErrorAction SilentlyContinue | ForEach-Object {
        $rel = $_.FullName.Substring($cut_ckpt.Length)
        if ($_.PSIsContainer) { Write-Host ("  [DIR] " + $rel) }
        else { Write-Host ("  [FILE] " + $rel + " (" + $_.Length + " bytes)") }
    }
}

# Search for any cut .pth/.pt files in baseline_v2
Write-Host "`n=== Search CUT .pth/.pt files in baseline_v2 ==="
Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2" -Recurse -Include "*.pth","*.pt" -ErrorAction SilentlyContinue | Where-Object { $_.FullName -match "cut" } | ForEach-Object {
    Write-Host ("  " + $_.FullName + " (" + $_.Length + " bytes)")
}

# Search for CUT code
Write-Host "`n=== Search for CUT code repositories ==="
@(
    "I:\Github\Latent_Style\exp_baselines\cut",
    "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\cut_code",
    "I:\Github\Latent_Style\CUT",
    "I:\Github\Latent_Style\SchrodingerBridge\CUT",
    "C:\Users\Administrator\CUT",
    "C:\Users\Administrator\baseline_v2\cut_code"
) | ForEach-Object {
    if (Test-Path $_) {
        Write-Host "  FOUND: $_"
        Get-ChildItem $_ -ErrorAction SilentlyContinue | Select-Object -First 10 | ForEach-Object { Write-Host ("    " + $_.Name) }
    }
}

# Search I:\Github for any CUT repo
Write-Host "`n=== Search I:\Github for CUT dirs ==="
Get-ChildItem "I:\Github" -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "CUT|cut" } | ForEach-Object {
    Write-Host ("  " + $_.FullName)
}

Write-Host "`n============================================================"
Write-Host "PART B: Check Photo2Art-256 dataset availability"
Write-Host "============================================================"
$p256_dirs = @(
    "I:\datasets\legacy256_overfit50",
    "I:\datasets\photo2art256",
    "I:\datasets\photo2art_256",
    "I:\datasets\legacy256"
)
foreach ($d in $p256_dirs) {
    if (Test-Path $d) {
        Write-Host "  FOUND: $d"
        Get-ChildItem $d -Directory | Select-Object -First 10 | ForEach-Object {
            $cnt = (Get-ChildItem $_.FullName -Filter *.jpg -ErrorAction SilentlyContinue).Count
            $cnt2 = (Get-ChildItem $_.FullName -Filter *.png -ErrorAction SilentlyContinue).Count
            Write-Host ("    " + $_.Name + " jpg=" + $cnt + " png=" + $cnt2)
        }
    }
}

Write-Host "`n============================================================"
Write-Host "PART C: Search for Seedream API scripts"
Write-Host "============================================================"
Get-ChildItem "I:\Github\Latent_Style" -Filter "*.py" -Recurse -Depth 3 -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "seedream|see_dream|api_call" } | ForEach-Object {
    Write-Host ("  " + $_.FullName)
}

Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\scripts" -Filter "*.py" -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "seedream|api" } | ForEach-Object {
    Write-Host ("  " + $_.FullName)
}

# Check seedream dirs for any scripts
@(
    "I:\Github\Latent_Style\seedream45_api",
    "I:\Github\Latent_Style\exp_baselines\seedream45_api"
) | ForEach-Object {
    if (Test-Path $_) {
        Write-Host "`n=== $_ ==="
        Get-ChildItem $_ -Recurse -ErrorAction SilentlyContinue | ForEach-Object {
            $rel = $_.FullName.Substring($_path.Length)
            if ($_.PSIsContainer) { Write-Host ("  [DIR] " + $rel) }
            else { Write-Host ("  [FILE] " + $rel + " (" + $_.Length + " bytes)") }
        }
    }
}

Write-Host "`n============================================================"
Write-Host "PART D: Inspect baseline_v2 config/code for CUT"
Write-Host "============================================================"
$cut_cfg = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\cut_code"
if (Test-Path $cut_cfg) {
    Get-ChildItem $cut_cfg -ErrorAction SilentlyContinue | ForEach-Object { Write-Host ("  " + $_.Name) }
}

# Check if there's a run script for CUT
Write-Host "`n=== Search for run_cut scripts ==="
Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge" -Filter "*cut*" -Recurse -Depth 3 -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host ("  " + $_.FullName + " (" + $_.Length + " bytes)")
}

Write-Host "`n============================================================"
Write-Host "PART E: Check _baseline_fill_results.json for clues"
Write-Host "============================================================"
$bfr = "I:\Github\Latent_Style\SchrodingerBridge\exp\_baseline_fill_results.json"
if (Test-Path $bfr) {
    $j = Get-Content $bfr -Raw | ConvertFrom-Json
    Write-Host ($j | ConvertTo-Json -Depth 5)
}

# Also check the pipeline fill results
Write-Host "`n=== _pipeline_fill_results.json ==="
$pfr = "I:\Github\Latent_Style\SchrodingerBridge\exp\_pipeline_fill_results.json"
if (Test-Path $pfr) {
    Get-Content $pfr -Raw
    Write-Host ""
}
