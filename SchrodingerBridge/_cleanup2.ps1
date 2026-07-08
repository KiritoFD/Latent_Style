$ErrorActionPreference = 'SilentlyContinue'
$root = "g:\GitHub\Latent_Style\SchrodingerBridge"
$deleted = 0

# Second-pass: non-underscore-prefixed one-off scripts
$patterns = @(
    "check_*.sh", "check_*.py", "check_*.bat", "check_*.ps1",
    "test_*.py", "test_*.sh", "test_*.ps1",
    "sweep_*.sh",
    "step1_*.bat", "step1_*.ps1",
    "deploy_*.sh", "deploy_*.ps1",
    "upload_*.py",
    "explore_*.py", "explore_*.sh",
    "diagnose_*.py",
    "find_*.py",
    "extract_*.py", "extract_metrics_remote.py", "extract_samam_metrics.py",
    "read_*.py",
    "analyze_*.py", "analyze_*.sh",
    "fix_*.sh",
    "run_bench_eval.cmd", "run_clip_style_decision_tree.bat", "run_clip_style_decision_tree.py",
    "run_extract_metrics.bat", "run_extraction.py", "run_extraction_final.py", "run_extraction_v2.py",
    "run_fiber_bundle.sh", "run_import_test.sh", "run_phase1_diagnostic_probes.py",
    "run_sdedit_sdturbo.py", "run_wsl_extraction.py",
    "run_adain_inference.py",
    "eval_fc_sb.sh", "eval_fc_sb_v2.sh", "eval_fc_sb_v3.sh", "eval_fg.sh", "direct_eval.sh",
    "copy_to_wsl.sh", "restart_experiment.sh", "launch_remote_620.sh",
    "deep_dive.py", "investigate_status.py",
    "start_remote_dino_pairing_cache.bat", "start_remote_dino_pairing_plan.bat", "start_wikiart512_ema_calib.bat",
    "take_dashboard_screenshot.js", "take_screenshot.js",
    "v3_fwd_sig.sh", "update_utils.sh", "with_src_path.sh", "test_with_src_path.sh",
    "remote_data_check.ps1", "run_remote_check.ps1",
    "phase2_status_report.txt", "FC_SB_PHASE2_STATUS_REPORT.md", "task4_analysis_report.md",
    "final_status_check.sh", "status.sh",
    "run_ablation_batch.py",
    "gen_task5_configs.py",
    "scan_analyzed.json", "scan_report.md",
    "run_eval_fc_sb.sh",
    "run_eval.py"
)

foreach ($pat in $patterns) {
    $files = Get-ChildItem -Path $root -Filter $pat -File
    foreach ($f in $files) {
        Remove-Item $f.FullName -Force
        $deleted++
    }
}
Write-Output "Deleted $deleted files in pass 2"

# Also clean scripts/ remaining non-underscore one-offs
$scriptsDir = Join-Path $root "scripts"
if (Test-Path $scriptsDir) {
    $sBefore = $deleted
    # Remove everything in scripts/ except README and truly essential files
    Get-ChildItem $scriptsDir -File | Where-Object {
        $_.Name -notmatch "^(README|check_deps|check_gpu|install_deps)" -and
        $_.Extension -in ".sh",".ps1",".py",".bat",".cmd"
    } | ForEach-Object {
        Remove-Item $_.FullName -Force
        $deleted++
    }
    Write-Output "Deleted $($deleted - $sBefore) more from scripts/"
}

Write-Output ""
Write-Output "=== TOTAL DELETED pass2: $deleted ==="
Write-Output ""
Write-Output "=== Remaining root files ==="
Get-ChildItem $root -File | Select-Object Name | Sort-Object Name
Write-Output ""
Write-Output "=== Remaining scripts/ ==="
Get-ChildItem $scriptsDir -File -ErrorAction SilentlyContinue | Select-Object Name
