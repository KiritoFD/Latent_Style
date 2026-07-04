$ErrorActionPreference = 'SilentlyContinue'
$pids = 11504, 27860, 29896, 33832
Write-Host "===== Targeted python processes ====="
foreach ($pid_ in $pids) {
    $p = Get-CimInstance Win32_Process -Filter "ProcessId = $pid_"
    if ($p) {
        Write-Host "--- PID $pid_ ---"
        Write-Host "  Name:      $($p.Name)"
        Write-Host "  Path:      $($p.ExecutablePath)"
        Write-Host "  CmdLine:   $($p.CommandLine)"
        Write-Host "  ParentPID: $($p.ParentProcessId)"
        Write-Host "  CreationDate: $($p.CreationDate)"
        $parent = Get-CimInstance Win32_Process -Filter "ProcessId = $($p.ParentProcessId)"
        if ($parent) {
            Write-Host "  Parent Name:    $($parent.Name)"
            Write-Host "  Parent Path:    $($parent.ExecutablePath)"
            Write-Host "  Parent CmdLine: $($parent.CommandLine)"
        }
        Write-Host ""
    } else {
        Write-Host "PID $pid_ : not found (already exited)"
    }
}

Write-Host "===== ALL python.exe processes ====="
Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'pythonw.exe' OR Name = 'python3.exe'" |
    Select-Object ProcessId, ParentProcessId, CreationDate, CommandLine |
    Format-List

Write-Host "===== Schtasks with python ====="
schtasks /Query /FO CSV /V 2>$null | Select-String -Pattern 'python|samam|train' -SimpleMatch

Write-Host ""
Write-Host "===== Process tree of python parents ====="
$allPy = Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'pythonw.exe'"
$parentIds = $allPy.ParentProcessId | Sort-Object -Unique
foreach ($ppid in $parentIds) {
    if ($ppid -and $ppid -ne 0) {
        $pp = Get-CimInstance Win32_Process -Filter "ProcessId = $ppid"
        if ($pp) {
            Write-Host "Parent PID $ppid : $($pp.Name) -> $($pp.CommandLine)"
        }
    }
}
