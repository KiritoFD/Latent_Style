# Regenerate every figure used by paper.tex from the bundled scripts + fig_data/.
# The paper folder is self-contained: no repo-external paths are required.
# Usage:  .\gen_figures.ps1
$ErrorActionPreference = "Stop"
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $dir
try {
    python gen_framework_figure.py      # framework_sfm_main.png  (architecture diagram)
    python gen_teaser_figure.py         # fig_teaser_comparison.png
    python plot_page1_summary.py        # fig_distinct5_page1_summary.pdf
    python make_radar.py                # radar_baselines_14.png
    Write-Host "All figures regenerated."
} finally {
    Pop-Location
}
