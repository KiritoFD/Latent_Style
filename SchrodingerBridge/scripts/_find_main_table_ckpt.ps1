# Find correct main-table checkpoint (early stop, epoch 4)
Write-Host "=== repro_brk_a_15ep checkpoints ==="
$ckptDir = "I:\Github\Latent_Style\WEAVE\runs\submission\repro_brk_a_15ep"
if (Test-Path $ckptDir) {
    Get-ChildItem $ckptDir -Filter "epoch_*.pt" | ForEach-Object {
        Write-Host $_.Name
    }
}

Write-Host ""
Write-Host "=== All epoch dirs with dino_summary.json ==="
Get-ChildItem -Path "I:\Github\Latent_Style\WEAVE\runs\submission\repro_brk_a_15ep" -Recurse -Filter "dino_summary.json" -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host ""
    Write-Host $_.FullName
    $j = Get-Content $_.FullName -Raw | ConvertFrom-Json
    Write-Host ("  all_dino_s = " + $j.all_dino_s)
    Write-Host ("  all_dino_c = " + $j.all_dino_c)
    Write-Host ("  all_clip_s = " + $j.all_clip_s)
    Write-Host ("  all_lpips  = " + $j.all_lpips)
}

Write-Host ""
Write-Host "=== All epoch dirs with summary.json (CLIP/LPIPS) ==="
Get-ChildItem -Path "I:\Github\Latent_Style\WEAVE\runs\submission\repro_brk_a_15ep" -Recurse -Filter "summary.json" -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host ""
    Write-Host $_.FullName
    try {
        $j = Get-Content $_.FullName -Raw | ConvertFrom-Json
        Write-Host ("  checkpoint = " + $j.checkpoint)
        if ($j.analysis.all_pairs_overview) {
            $a = $j.analysis.all_pairs_overview
            Write-Host ("  clip_style = " + $a.clip_style)
            Write-Host ("  lpips      = " + $a.content_lpips)
        }
        if ($j.timings_sec) {
            Write-Host ("  wall_total = " + $j.timings_sec.wall_total)
        }
    } catch {
        Write-Host "  (parse error)"
    }
}

Write-Host ""
Write-Host "=== Search for main_table reference in docs ==="
$docPaths = @(
    "I:\Github\Latent_Style\WEAVE\docs\exp\brk_a_15ep_handoff.md",
    "I:\Github\Latent_Style\WEAVE\docs\README.md"
)
foreach ($d in $docPaths) {
    if (Test-Path $d) {
        Write-Host ""
        Write-Host "--- $d ---"
        Select-String -Path $d -Pattern "epoch|main.table|early.stop|DINO|0.491|0.712" | Select-Object -First 20 | ForEach-Object {
            Write-Host $_.Line
        }
    }
}
