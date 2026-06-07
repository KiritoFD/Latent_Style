@echo off
setlocal

set "SRC=%~dp0"
if "%SRC:~-1%"=="\" set "SRC=%SRC:~0,-1%"
set "OUT=%SRC%\paper_aaai2027.pdf"
set "TEX=pdflatex"
set "ENGINE=C:\texlive\2026\bin\windows\pdflatex.exe"

echo ===== Building paper_aaai2027.pdf =====

if not exist "%ENGINE%" (
    echo ERROR: pdflatex not found at "%ENGINE%"
    echo Please check your TeX Live installation.
    exit /b 1
)

cd /d "%SRC%"

echo [1/4] Running pdflatex (pass 1)...
"%ENGINE%" -interaction=nonstopmode -halt-on-error paper_aaai2027.tex > pdflatex_pass1.log 2>&1

echo [2/4] Running bibtex...
bibtex paper_aaai2027 > bibtex.log 2>&1

echo [3/4] Running pdflatex (pass 2 - cross-refs)...
"%ENGINE%" -interaction=nonstopmode -halt-on-error paper_aaai2027.tex > pdflatex_pass2.log 2>&1

echo [4/4] Running pdflatex (pass 3 - final)...
"%ENGINE%" -interaction=nonstopmode -halt-on-error paper_aaai2027.tex > pdflatex_pass3.log 2>&1

if exist "%OUT%" (
    for %%A in ("%OUT%") do echo SUCCESS: paper_aaai2027.pdf - %%~nA KB = %%~zA / 1024 MB
) else (
    echo ERROR: PDF not generated. Check pdflatex_pass*.log
    exit /b 1
)

echo ===== Done =====
