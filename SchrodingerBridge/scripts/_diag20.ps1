$LOG = "I:\Github\Latent_Style\SchrodingerBridge\logs\baseline_wikiarts20.log"
$now = Get-Date
"=== DIAG at $now ==="

# 1. schtasks status
"--- schtasks wikiarts20_all ---"
$st = schtasks /Query /TN wikiarts20_all /V /FO CSV 2>$null | ConvertFrom-Csv
if ($st) {
    $keys = $st | Get-Member -MemberType NoteProperty | Select-Object -ExpandProperty Name
    "  Available fields: $keys"
    # Print first few fields
    foreach ($k in $keys) {
        $v = $st.$k
        if ($v -and $v.ToString().Length -gt 0) {
            "  $k = $v"
        }
    }
}

# 2. Running processes
"--- powershell processes ---"
Get-Process powershell -ErrorAction SilentlyContinue | Select-Object Id, StartTime, CPU | Format-Table -AutoSize

"--- python processes ---"
$py = Get-Process python -ErrorAction SilentlyContinue
if ($py) {
    $py | Select-Object Id, StartTime, CPU, @{N='WS_MB';E={[int]($_.WorkingSet64/1MB)}} | Format-Table -AutoSize
} else {
    "  (none)"
}

# 3. Log file sizes
"--- eval log files ---"
foreach ($m in @('identity','adain','wct')) {
    foreach ($t in @('eval.out','eval.err')) {
        $f = "$LOG.$m.$t"
        if (Test-Path $f) {
            $fi = Get-Item $f
            "  $m.$t size=$($fi.Length) mtime=$($fi.LastWriteTime)"
        } else {
            "  $m.$t NOT FOUND"
        }
    }
}

# 4. CSV status
"--- CSV status ---"
foreach ($m in @('identity','adain','wct')) {
    $csv = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20\$m\metrics.csv"
    if (Test-Path $csv) {
        $fi = Get-Item $csv
        "  $m csv size=$($fi.Length) mtime=$($fi.LastWriteTime)"
    } else {
        "  $m NO CSV"
    }
}

# 5. Last 8 lines of main log
"--- last 8 lines of main log ---"
Get-Content $LOG -Tail 8

# 6. Last 3 lines of identity eval out
"--- identity.eval.out last 3 ---"
$ieo = "$LOG.identity.eval.out"
if (Test-Path $ieo) { Get-Content $ieo -Tail 3 }

# 7. identity eval err
"--- identity.eval.err (full) ---"
$iee = "$LOG.identity.eval.err"
if (Test-Path $iee) {
    "  size=$((Get-Item $iee).Length)"
    Get-Content $iee -Tail 30
} else {
    "  NOT FOUND"
}

# 8. python 4828 cmdline
"--- python 4828 cmdline ---"
$cmd = (Get-CimInstance Win32_Process -Filter "ProcessId=4828" -ErrorAction SilentlyContinue).CommandLine
if ($cmd) { "  $cmd" } else { "  (proc not found or cannot read cmdline)" }

"=== DIAG END ==="
