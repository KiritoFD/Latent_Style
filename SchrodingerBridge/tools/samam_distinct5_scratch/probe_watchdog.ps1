$ErrorActionPreference = 'SilentlyContinue'

Write-Host "===== PID 2340 (suspected scheduler) ====="
$p = Get-CimInstance Win32_Process -Filter "ProcessId = 2340"
if ($p) {
    Write-Host "Name: $($p.Name)"
    Write-Host "Path: $($p.ExecutablePath)"
    Write-Host "CmdLine: $($p.CommandLine)"
    Write-Host "ParentPID: $($p.ParentProcessId)"
    Write-Host "CreationDate: $($p.CreationDate)"
    $pp = Get-CimInstance Win32_Process -Filter "ProcessId = $($p.ParentProcessId)"
    if ($pp) { Write-Host "Parent: $($pp.Name) -> $($pp.CommandLine)" }
}

Write-Host ""
Write-Host "===== PID 16784 (suspected watchdog powershell) ====="
$p2 = Get-CimInstance Win32_Process -Filter "ProcessId = 16784"
if ($p2) {
    Write-Host "Name: $($p2.Name)"
    Write-Host "Path: $($p2.ExecutablePath)"
    Write-Host "CmdLine: $($p2.CommandLine)"
    Write-Host "ParentPID: $($p2.ParentProcessId)"
    Write-Host "CreationDate: $($p2.CreationDate)"
    $pp2 = Get-CimInstance Win32_Process -Filter "ProcessId = $($p2.ParentProcessId)"
    if ($pp2) { Write-Host "Parent: $($pp2.Name) -> $($pp2.CommandLine)" }
}

Write-Host ""
Write-Host "===== All children of PID 2340 ====="
Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq 2340 } |
    Select-Object ProcessId, Name, CreationDate, CommandLine | Format-List

Write-Host ""
Write-Host "===== Recursive descendants of 2340 ====="
function Get-Descendants($rootPid, $depth=0) {
    $children = Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $rootPid }
    foreach ($c in $children) {
        Write-Host ("  " * $depth) "+-$($c.ProcessId) $($c.Name) : $($c.CommandLine)"
        if ($depth -lt 4) { Get-Descendants $c.ProcessId ($depth+1) }
    }
}
Get-Descendants 2340

Write-Host ""
Write-Host "===== Children of 16784 ====="
Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq 16784 } |
    Select-Object ProcessId, Name, CreationDate, CommandLine | Format-List

Write-Host ""
Write-Host "===== wsl_persistent_holder.bat content ====="
$holder = "C:\Users\Administrator\wsl_persistent_holder.bat"
if (Test-Path $holder) { Get-Content $holder } else { Write-Host "(not found)" }

Write-Host ""
Write-Host "===== test_wsl_cmd.ps1 content ====="
$tw = "C:\Users\Administrator\test_wsl_cmd.ps1"
if (Test-Path $tw) { Get-Content $tw } else { Write-Host "(not found)" }

Write-Host ""
Write-Host "===== _628_watchdog.ps1 content ====="
$wd = "I:\Github\Latent_Style\SchrodingerBridge\_628_watchdog.ps1"
if (Test-Path $wd) { Get-Content $wd } else { Write-Host "(not found)" }

Write-Host ""
Write-Host "===== Check Startup folder / Run registry ====="
Get-ChildItem "C:\Users\Administrator\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup\" -ErrorAction SilentlyContinue | Format-Table Name, LastWriteTime
Get-ItemProperty "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run" -ErrorAction SilentlyContinue
