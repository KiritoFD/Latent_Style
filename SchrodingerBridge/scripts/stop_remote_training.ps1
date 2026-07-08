param(
    [string]$Root = "I:\Github\Latent_Style\SchrodingerBridge",
    [switch]$KillAllPython
)

$ErrorActionPreference = "Continue"
$patterns = @(
    [regex]::Escape($Root),
    "SchrodingerBridge",
    "src\\run\.py",
    "launch_remote_swd_ablation\.ps1",
    "python.*watchdog",
    "python.*monitor",
    "_628_"
)

$rows = Get-CimInstance Win32_Process |
    Where-Object {
        $cmd = [string]$_.CommandLine
        if ($_.ProcessId -eq $PID -or $_.Name -match "msedgewebview2|conhost") { return $false }
        if ($KillAllPython -and $_.Name -match "python") { return $true }
        foreach ($pat in $patterns) {
            if ($cmd -match $pat) { return $true }
        }
        return $false
    } |
    Sort-Object CreationDate

if (-not $rows) {
    "NO_MATCHING_PROCESSES"
    exit 0
}

"MATCHING_PROCESSES"
$rows | Select-Object ProcessId,Name,CreationDate,CommandLine | Format-List

foreach ($row in $rows) {
    try {
        Stop-Process -Id $row.ProcessId -Force -ErrorAction Stop
        "KILLED pid=$($row.ProcessId) name=$($row.Name)"
    } catch {
        "FAILED pid=$($row.ProcessId) name=$($row.Name) error=$($_.Exception.Message)"
    }
}
