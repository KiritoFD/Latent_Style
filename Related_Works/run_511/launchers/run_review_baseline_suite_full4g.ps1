$ErrorActionPreference = "Stop"
Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue

$python = $env:UV_PYTHON
if (-not $python) {
  $python = "python"
}

$root = "G:\GitHub\Latent_Style"
$script = Join-Path $root "Related_Works\run_511\launchers\run_review_baseline_suite.py"
$outRoot = Join-Path $root "Related_Works\run_511\outputs\review_baseline_suite_full4g"

Set-Location $root
& $python $script --mode all --profile 4g --output_root $outRoot
