@echo off
setlocal enabledelayedexpansion

set "SRC=G:\GitHub\Latent_Style\SchrodingerBridge\paper_refine_v2\paper_cn"
set "OUT=%SRC%\paper_cn.pdf"
set "ENGINE=C:\texlive\2026\bin\windows\xelatex.exe"

echo ===== Building paper_cn.pdf =====

if not exist "%ENGINE%" (
    echo ERROR: xelatex not found at "%ENGINE%"
    exit /b 1
)

cd /d "%SRC%"

echo [1/3] Running xelatex (pass 1)...
"%ENGINE%" -interaction=nonstopmode -halt-on-error paper.tex > xelatex_pass1.log 2>&1

echo [2/3] Running xelatex (pass 2)...
"%ENGINE%" -interaction=nonstopmode -halt-on-error paper.tex > xelatex_pass2.log 2>&1

echo [3/3] Running xelatex (pass 3)...
"%ENGINE%" -interaction=nonstopmode -halt-on-error paper.tex > xelatex_pass3.log 2>&1

if exist "%OUT%" (
    for %%A in ("%OUT%") do (
        set /A size=%%~zA/1024
        echo SUCCESS: paper_cn.pdf generated - !size! KB
    )
) else (
    echo ERROR: PDF not generated. Check xelatex_pass*.log
    exit /b 1
)

echo ===== Done =====