# Find existing 256 CUT training data
Write-Host "=== Related_Works/runs/cut_5x5 structure ==="
$cutRuns = "I:\Github\Latent_Style\Related_Works\runs\cut_5x5"
if (Test-Path $cutRuns) {
    Get-ChildItem $cutRuns | ForEach-Object { Write-Host "  $($_.Name)" }
    Write-Host "`n--- infer_5x5/images ---"
    $imgs = "$cutRuns\infer_5x5\images"
    if (Test-Path $imgs) {
        $cnt = (Get-ChildItem $imgs -Filter *.png).Count + (Get-ChildItem $imgs -Filter *.jpg).Count
        Write-Host "  Total images: $cnt"
        Get-ChildItem $imgs | Select-Object -First 5 | ForEach-Object { Write-Host "  $($_.Name)" }
    }
    Write-Host "`n--- train_5x5 dir? ---"
    $trainDir = "$cutRuns\train_5x5"
    if (Test-Path $trainDir) {
        Get-ChildItem $trainDir | ForEach-Object { Write-Host "  $($_.Name)" }
    }
    Write-Host "`n--- All subdirs ---"
    Get-ChildItem $cutRuns -Directory -Recurse -Depth 2 | ForEach-Object { Write-Host "  $($_.FullName)" }
}

Write-Host "`n=== Check legacy256_overfit50 train dir ==="
$trainRoot = "I:\datasets\legacy256_overfit50\train"
if (Test-Path $trainRoot) {
    Get-ChildItem $trainRoot -Directory | ForEach-Object {
        $cnt = (Get-ChildItem $_.FullName -File -ErrorAction SilentlyContinue).Count
        Write-Host "  $($_.Name): $cnt files"
    }
} else {
    Write-Host "  NOT EXIST"
}

Write-Host "`n=== Check exp_baselines/_auxiliary_runs/cut_5x5/cut_repo ==="
$cutAuxRepo = "I:\Github\Latent_Style\exp_baselines\_auxiliary_runs\cut_5x5\cut_repo"
if (Test-Path $cutAuxRepo) {
    Get-ChildItem $cutAuxRepo | ForEach-Object { Write-Host "  $($_.Name)" }
}

Write-Host "`n=== final_works/CUT ==="
$fwCut = "I:\Github\Latent_Style\final_works\CUT"
if (Test-Path $fwCut) {
    Get-ChildItem $fwCut -Recurse -Depth 2 | ForEach-Object { Write-Host "  $($_.FullName)" }
}
