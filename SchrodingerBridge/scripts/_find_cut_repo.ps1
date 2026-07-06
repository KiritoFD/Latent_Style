# Find CUT repo and existing 256 baseline_v2/image generation script
Write-Host "=== Searching for CUT repo (broader) ==="
$found = Get-ChildItem "I:\Github" -Directory -Recurse -Filter "*CUT*" -ErrorAction SilentlyContinue -Depth 4
foreach ($f in $found) { Write-Host "  $($f.FullName)" }

Write-Host "`n=== Searching for cut_256 / cut photo2art generation scripts ==="
$genScripts = Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\scripts" -Filter "*cut*" -ErrorAction SilentlyContinue
foreach ($s in $genScripts) { Write-Host "  $($s.FullName) ($($s.Length)B)" }

Write-Host "`n=== Existing 512 distinct5 CUT images structure ==="
$cutImg = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\images\cut"
if (Test-Path $cutImg) {
    Write-Host "Image count: $((Get-ChildItem $cutImg -Filter *.png).Count + (Get-ChildItem $cutImg -Filter *.jpg).Count)"
    Write-Host "First 5 files:"
    Get-ChildItem $cutImg | Select-Object -First 5 | ForEach-Object { Write-Host "  $($_.Name)" }
}

Write-Host "`n=== CUT checkpoint structure ==="
$cutCkpt = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\checkpoints\cut"
if (Test-Path $cutCkpt) {
    Get-ChildItem $cutCkpt -Directory | ForEach-Object { Write-Host "  $($_.Name)" }
}

Write-Host "`n=== Check if CycleGAN CUT repo exists in WSL ==="
Write-Host "Looking for /mnt/i/Github or similar CUT repos..."
$wslPaths = @(
    "I:\Github\Latent_Style\Related_Works\repos\CUT",
    "I:\Github\Latent_Style\Related_Works\CUT",
    "I:\Github\CUT"
)
foreach ($p in $wslPaths) {
    if (Test-Path $p) { Write-Host "  EXISTS: $p" }
    else { Write-Host "  NOT EXIST: $p" }
}
