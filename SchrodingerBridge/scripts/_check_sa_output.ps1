# Check StyleAligned output paths
Write-Output "=== Script output dir config ==="
Select-String -Path "C:\Users\Administrator\_run_stylealigned_remote.py" -Pattern "out_dir|output|save|images|mkdir" | ForEach-Object { $_.Line.Trim() }

Write-Output "`n=== Check exp_baselines for stylealigned ==="
if (Test-Path "I:\Github\Latent_Style\exp_baselines\stylealigned") {
    Get-ChildItem "I:\Github\Latent_Style\exp_baselines\stylealigned" -Recurse -File | Measure-Object | Select-Object -ExpandProperty Count
    Get-ChildItem "I:\Github\Latent_Style\exp_baselines\stylealigned" -Directory | Select-Object Name
} else {
    Write-Output "stylealigned dir not found in exp_baselines"
}

Write-Output "`n=== Search recent dirs on I: ==="
Get-ChildItem "I:\Github\Latent_Style" -Directory | Where-Object { $_.Name -like "*style*" -or $_.Name -like "*aligned*" -or $_.Name -like "*zstar*" } | Select-Object Name

Write-Output "`n=== Check any new images under exp_baselines ==="
Get-ChildItem "I:\Github\Latent_Style\exp_baselines" -Directory | ForEach-Object { 
    $count = (Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
    if ($count -gt 0) { Write-Output "$($_.Name): $count files" }
}

Write-Output "`n=== Check for stylealigned output anywhere ==="
Get-ChildItem "C:\Users\Administrator" -Directory -Filter "*style*" -ErrorAction SilentlyContinue | Select-Object Name
Get-ChildItem "I:\" -Directory -Filter "*style*" -ErrorAction SilentlyContinue | Select-Object Name
