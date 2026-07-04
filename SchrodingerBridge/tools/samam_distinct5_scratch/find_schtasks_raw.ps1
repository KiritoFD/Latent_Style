$ErrorActionPreference = 'SilentlyContinue'

Write-Host "===== ALL scheduled tasks (raw, then filtered) ====="
$raw = schtasks /Query /FO LIST 2>$null
Write-Host "Total lines: $($raw.Count)"

# Parse list format: TaskName: / Status: / Task To Run:
$blocks = $raw -split "`r`n`r`n"
Write-Host "Total blocks: $($blocks.Count)"

$hits = @()
foreach ($b in $blocks) {
    if ($b -match '628|watchdog|p7_runner|p8d|p8e|destructive|clean_base|samam|SchrodingerBridge') {
        $hits += $b
    }
}
Write-Host "===== Suspicious task blocks (containing 628/watchdog/SchrodingerBridge/etc) ====="
$hits | ForEach-Object { Write-Host "----"; Write-Host $_ }

Write-Host ""
Write-Host "===== Validate.log (clean_base) tail ====="
$vlog = "I:\Github\Latent_Style\SchrodingerBridge\exp\clean_base\validate.log"
if (Test-Path $vlog) { Get-Content $vlog -Tail 30 } else { Write-Host "(no validate.log)" }

Write-Host ""
Write-Host "===== p8e_launcher / p8f_launcher bats existence ====="
Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\_628_p8*.bat" -ErrorAction SilentlyContinue |
    Select-Object Name, LastWriteTime | Format-Table -AutoSize

Write-Host ""
Write-Host "===== wsl_holder.log tail ====="
$whl = "C:\Users\Administrator\wsl_holder.log"
if (Test-Path $whl) { Get-Content $whl -Tail 15 } else { Write-Host "(no wsl_holder.log)" }
