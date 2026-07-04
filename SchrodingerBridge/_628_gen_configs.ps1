# Run 628_gen_destructive_configs.py on remote and verify
$ErrorActionPreference = 'Stop'
$root = 'I:/Github/Latent_Style/SchrodingerBridge'
Set-Location $root

$py = 'C:\Program Files\Python312\python.exe'
if (-not (Test-Path $py)) { $py = 'python' }

Write-Host "=== Running 628_gen_destructive_configs.py ==="
& $py "$root\628_gen_destructive_configs.py" 2>&1 | Select-Object -Last 5

Write-Host "`n=== Verifying config count ==="
$cfgDir = "$root\configs\ablations\628_destructive"
$files = Get-ChildItem $cfgDir -Filter '*.json' -ErrorAction SilentlyContinue
Write-Host "Total configs: $($files.Count)"

Write-Host "`n=== Verifying by prefix ==="
$prefixes = @('D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13','D14','D15','D16','D17','D18','D19','D20','D21','D22','D23','D24','D25','D26','D27','D28','D29','D30','L1','L2','L3','L4','L5','L6','L7','L8','L9','L10','L11','L12','L13','L14','L15','L16','E1','E2','E3','E4','E5','E6','E7','E8','E9','E10','E11','E12','E13','E14','E15','E16','E17','E18','E19','E20','E21','E22','E23','E24','P1','P2','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','P17','P18')
$counts = @{}
foreach ($f in $files) {
    $name = $f.BaseName
    foreach ($p in $prefixes) {
        if ($name.StartsWith($p + '_') -or $name -eq $p) {
            if (-not $counts.ContainsKey($p)) { $counts[$p] = 0 }
            $counts[$p]++
            break
        }
    }
}
Write-Host "D count: $(($counts.Keys | Where-Object { $_ -like 'D*' }).Count) prefixes"
Write-Host "L count: $(($counts.Keys | Where-Object { $_ -like 'L*' }).Count) prefixes"
Write-Host "E count: $(($counts.Keys | Where-Object { $_ -like 'E*' }).Count) prefixes"
Write-Host "P count: $(($counts.Keys | Where-Object { $_ -like 'P*' }).Count) prefixes"

Write-Host "`n=== Spot check: D19, L13, E1, P13_wflow_01 ==="
foreach ($n in @('D19_attn_gated_raw','L13_no_flow','E1_w_contrast_preserve','P13_wflow_01','P18_gate_init_10')) {
    $f = "$cfgDir\$n.json"
    if (Test-Path $f) {
        $j = Get-Content $f -Raw | ConvertFrom-Json
        $resume = $j.training.resume_checkpoint
        $ne = $j.training.num_epochs
        $feee = $j.training.full_eval_each_epoch
        Write-Host ("  $n : OK resume_ep7=$($resume -like '*epoch_0007*') num_epochs=$ne full_eval_each=$feee")
    } else {
        Write-Host "  $n : MISSING"
    }
}
