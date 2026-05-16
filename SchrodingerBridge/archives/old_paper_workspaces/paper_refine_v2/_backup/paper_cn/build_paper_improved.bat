@echo off
setlocal enabledelayedexpansion

set "SRC=G:\GitHub\Latent_Style\SchrodingerBridge\paper_refine_v2\paper_cn"
set "OUT=%SRC%\paper.pdf"
set "TEX=xelatex"
set "ENGINE=C:\texlive\2026\bin\windows\xelatex.exe"
set "BIBTEX=C:\texlive\2026\bin\windows\bibtex.exe"

echo ===== Building paper.pdf =====

if not exist "%ENGINE%" (
    echo ERROR: xelatex not found at "%ENGINE%"
    echo Please check your TeX Live installation.
    exit /b 1
)

if not exist "%BIBTEX%" (
    echo ERROR: bibtex not found at "%BIBTEX%"
    echo Please check your TeX Live installation.
    exit /b 1
)

cd /d "%SRC%"

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

echo [1/4] Running xelatex (pass 1)...
"%ENGINE%" -interaction=nonstopmode -halt-on-error paper.tex > xelatex_pass1.log 2>&1

echo [2/4] Running bibtex to process references...
"%BIBTEX%" paper > bibtex.log 2>&1

echo [3/4] Running xelatex (pass 2 - cross-refs)...
"%ENGINE%" -interaction=nonstopmode -halt-on-error paper.tex > xelatex_pass2.log 2>&1

echo [4/4] Running xelatex (pass 3 - final)...
"%ENGINE%" -interaction=nonstopmode -halt-on-error paper.tex > xelatex_pass3.log 2>&1

if exist "%OUT%" (
    for %%A in ("%OUT%") do (
        set /A size=%%~zA/1024
        echo SUCCESS: paper.pdf generated - !size! KB
    )
) else (
    echo ERROR: PDF not generated. Check xelatex_pass*.log and bibtex.log
    echo.
    echo Last 20 lines of pass3 log:
    for /f "tokens=*" %%i in ('type xelatex_pass3.log ^| findstr /C:" " ^| find /V /C "" ^| tail -n 20') do echo %%i
    exit /b 1
)

echo ===== Done =====
