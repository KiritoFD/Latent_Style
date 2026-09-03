@echo off
setlocal enabledelayedexpansion

set "SRC=G:\GitHub\Latent_Style\SchrodingerBridge\paper_refine_v2\aaai_submission"
set "OUT=%SRC%\paper_aaai2026.pdf"
set "TEX=xelatex"
set "ENGINE=C:\texlive\2026\bin\windows\xelatex.exe"
set "INKSCAPE=C:\Program Files\Inkscape\bin\inkscape.exe"

echo ===== Building paper_aaai2026.pdf with SVG framework =====

if not exist "%ENGINE%" (
    echo ERROR: xelatex not found at "%ENGINE%"
    echo Please check your TeX Live installation.
    exit /b 1
)

cd /d "%SRC%"

rem Convert SVG to PDF if needed
if exist "framework_figure_final.svg" (
    echo [0/3] Converting SVG to PDF...
    if exist "%INKSCAPE%" (
        "%INKSCAPE%" --export-type=pdf --export-filename=framework_figure.pdf framework_figure_final.svg >nul 2>&1
        if !errorlevel! equ 0 (
            echo SVG converted to PDF successfully
        ) else (
            echo WARNING: Inkscape conversion failed, trying fallback...
        )
    ) else (
        echo WARNING: Inkscape not found, trying direct SVG usage...
    )
)

rem Check if required figure exists
if not exist "fig_quality_tradeoff.png" (
    echo WARNING: fig_quality_tradeoff.png not found
    echo Trying to copy from figures folder...
    if exist "..\figures\fig_pareto_lpips_vs_style.png" (
        copy "..\figures\fig_pareto_lpips_vs_style.png" "fig_quality_tradeoff.png" >nul
        echo Copied fig_pareto_lpips_vs_style.png as fig_quality_tradeoff.png
    ) else (
        echo ERROR: No suitable figure found
        exit /b 1
    )
)

echo [1/3] Running xelatex (pass 1)...
"%ENGINE%" -interaction=nonstopmode -halt-on-error paper_aaai2026.tex > xelatex_pass1.log 2>&1

echo [2/3] Running xelatex (pass 2 - cross-refs)...
"%ENGINE%" -interaction=nonstopmode -halt-on-error paper_aaai2026.tex > xelatex_pass2.log 2>&1

echo [3/3] Running xelatex (pass 3 - final)...
"%ENGINE%" -interaction=nonstopmode -halt-on-error paper_aaai2026.tex > xelatex_pass3.log 2>&1

if exist "%OUT%" (
    for %%A in ("%OUT%") do (
        set /A size=%%~zA/1024
        echo SUCCESS: paper_aaai2026.pdf generated - !size! KB
    )
) else (
    echo ERROR: PDF not generated. Check xelatex_pass*.log
    echo.
    echo Last 30 lines of pass3 log:
    type xelatex_pass3.log | findstr /E "^" | sort /R | for /f "delims=" %%x in ('more') do echo %%x
    exit /b 1
)

echo ===== Done =====
