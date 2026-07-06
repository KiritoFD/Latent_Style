# Find CUT repo and checkpoints on remote
Write-Host "=== Searching for CUT repos ==="
$cutRepos = @()
$searchRoots = @(
    "I:\Github\Latent_Style\Related_Works\repos",
    "I:\Github\Latent_Style\Related_Works",
    "I:\Github\Latent_Style"
)
foreach ($root in $searchRoots) {
    if (Test-Path $root) {
        $found = Get-ChildItem $root -Directory -Recurse -Filter "*CUT*" -ErrorAction SilentlyContinue -Depth 3
        foreach ($f in $found) {
            $cutRepos += $f.FullName
            Write-Host "  FOUND: $($f.FullName)"
        }
    }
}

Write-Host "`n=== Searching for CUT checkpoints ==="
$ckptDirs = @(
    "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\checkpoints\cut",
    "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\checkpoints",
    "I:\exp_256_photo2art",
    "I:\Github\Latent_Style\SchrodingerBridge\exp"
)
foreach ($d in $ckptDirs) {
    if (Test-Path $d) {
        Write-Host "--- $d ---"
        Get-ChildItem $d -Recurse -Include "*.pt","*.pth","*.ckpt" -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "cut|CUT" -or $_.DirectoryName -match "cut|CUT" } | ForEach-Object { Write-Host "  $($_.FullName) ($([math]::Round($_.Length/1MB,1))MB)" }
    }
}

Write-Host "`n=== CUT 256 image outputs ==="
$cutImgDirs = @(
    "I:\exp_256_photo2art\cut_256",
    "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\images\cut",
    "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\cut"
)
foreach ($d in $cutImgDirs) {
    if (Test-Path $d) {
        $cnt = (Get-ChildItem $d -Filter *.png -ErrorAction SilentlyContinue).Count + (Get-ChildItem $d -Filter *.jpg -ErrorAction SilentlyContinue).Count
        Write-Host "  $d : $cnt images"
    } else {
        Write-Host "  $d : NOT EXIST"
    }
}

Write-Host "`n=== All dirs in Related_Works/repos ==="
if (Test-Path "I:\Github\Latent_Style\Related_Works\repos") {
    Get-ChildItem "I:\Github\Latent_Style\Related_Works\repos" -Directory | ForEach-Object { Write-Host "  $($_.Name)" }
}
