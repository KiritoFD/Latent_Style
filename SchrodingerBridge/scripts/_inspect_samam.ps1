# Inspect baseline_wikiarts20 and samam process
$ErrorActionPreference = "Continue"

Write-Host "=== baseline_wikiarts20 ==="
$bw = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20"
if (Test-Path $bw) {
    Get-ChildItem $bw -Directory | ForEach-Object {
        $imgs = Join-Path $_.FullName "images"
        $cnt = 0
        if (Test-Path $imgs) {
            $cnt = (Get-ChildItem $imgs -File -ErrorAction SilentlyContinue | Measure-Object).Count
        } elseif ($_.Name -eq "samam") {
            $cnt = (Get-ChildItem $_.FullName -File -ErrorAction SilentlyContinue | Measure-Object).Count
        }
        Write-Host ("{0}: {1} files" -f $_.Name, $cnt)
    }
} else {
    Write-Host "no baseline_wikiarts20 dir"
}

Write-Host ""
Write-Host "=== SaMam process detail ==="
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Select-Object ProcessId, CreationDate, CommandLine |
    Format-List

Write-Host ""
Write-Host "=== samam_w20.gen.log size ==="
$log = "I:\Github\Latent_Style\SchrodingerBridge\logs\samam_w20.gen.log"
if (Test-Path $log) {
    $len = (Get-Item $log).Length
    Write-Host "len=$len"
    if ($len -gt 0) {
        Get-Content $log -Tail 30
    }
}

Write-Host ""
Write-Host "=== post_pipeline.log tail ==="
$pp = "I:\Github\Latent_Style\SchrodingerBridge\logs\post_pipeline.log"
if (Test-Path $pp) {
    Get-Content $pp -Tail 50
}
