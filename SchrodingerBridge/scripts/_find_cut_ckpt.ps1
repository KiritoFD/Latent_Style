# Find CUT 256 checkpoints
Write-Host "=== exp_baselines/_auxiliary_runs/cut_5x5/cut_repo/infer_5x5 ==="
$auxCut = "I:\Github\Latent_Style\exp_baselines\_auxiliary_runs\cut_5x5\cut_repo"
if (Test-Path $auxCut) {
    Get-ChildItem $auxCut -Recurse -Depth 3 | ForEach-Object { Write-Host "  $($_.FullName)" }
}

Write-Host "`n=== Search for cut_to_cezanne, cut_to_monet etc checkpoints ==="
$searchDirs = @(
    "I:\Github\Latent_Style\final_works\CUT",
    "I:\Github\Latent_Style\exp_baselines",
    "I:\Github\Latent_Style\Related_Works\runs"
)
foreach ($d in $searchDirs) {
    if (Test-Path $d) {
        $found = Get-ChildItem $d -Recurse -Directory -Filter "cut_to_*" -ErrorAction SilentlyContinue -Depth 4
        foreach ($f in $found) {
            Write-Host "  $($f.FullName)"
            $ckpts = Get-ChildItem $f.FullName -Filter "*.pth" -ErrorAction SilentlyContinue
            foreach ($c in $ckpts) { Write-Host "    $($c.Name) ($([math]::Round($c.Length/1MB,1))MB)" }
        }
    }
}

Write-Host "`n=== Search for latest_net_G.pth everywhere under Latent_Style ==="
$gFiles = Get-ChildItem "I:\Github\Latent_Style" -Recurse -Filter "latest_net_G.pth" -ErrorAction SilentlyContinue -Depth 6
foreach ($g in $gFiles) {
    Write-Host "  $($g.FullName) ($([math]::Round($g.Length/1MB,1))MB)"
}

Write-Host "`n=== final_works/CUT/meta.json ==="
$meta = "I:\Github\Latent_Style\final_works\CUT\meta.json"
if (Test-Path $meta) { Get-Content $meta -Raw }

Write-Host "`n=== final_works/CUT/summary.json ==="
$summ = "I:\Github\Latent_Style\final_works\CUT\summary.json"
if (Test-Path $summ) { Get-Content $summ -Raw }
