# Build the ICME paper and supplement (Windows).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python tools/make_main_table.py
latexmk -pdf -interaction=nonstopmode weave.tex weave_supp.tex
