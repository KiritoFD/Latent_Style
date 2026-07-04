# Collect 628-ALL-DEBUG output from X experiments
$remote = 'Administrator@100.115.18.62'
$port = '2222'
$remoteRoot = 'I:\Github\Latent_Style\SchrodingerBridge'

$xExps = @(
    'X1_velmag_w10','X2_velmag_w50','X3_velmag_w100',
    'X4_dir_cos_w10','X5_dir_cos_w50','X6_dir_cos_w100',
    'X7_outvar_w10','X8_outvar_w50','X9_outvar_w100',
    'X10_contrast_w10','X11_contrast_w50','X12_contrast_w100',
    'X13_chvar_w10','X14_chvar_w50','X15_chvar_w100',
    'X16_hfenergy_w10','X17_hfenergy_w50','X18_hfenergy_w100',
    'X19_colormatch_w10','X20_colormatch_w50','X21_colormatch_w100',
    'X22_hsvsat_w1','X23_hsvsat_w10','X24_hsvsat_w50',
    'X25_attnent_w1','X26_attnent_w10','X27_attnent_w50',
    'X28_combo_content_w50','X29_combo_direction_w50',
    'X30_combo_all_w10','X31_combo_all_w50'
)

foreach ($x in $xExps) {
    $logPath = "$remoteRoot\exp\628_ablation\destructive_logs\$x.log"
    $cmd = "powershell -NoProfile -Command `"if (Test-Path '$logPath') { (Get-Content '$logPath' | Select-String '628-ALL-DEBUG' | Select-Object -Last 1).Line } else { Write-Host 'MISSING: $x' }`""
    $result = ssh -o ConnectTimeout=10 $remote -p $port $cmd 2>$null
    if ($result) {
        Write-Host "[$x] $result"
    } else {
        Write-Host "[$x] (no 628-ALL-DEBUG output yet)"
    }
}
