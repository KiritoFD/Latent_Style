@echo off
setlocal

set "SRC=%~dp0"
if "%SRC:~-1%"=="\" set "SRC=%SRC:~0,-1%"
set "OUT=%SRC%\paper_aaai2026.pdf"
set "TEX=xelatex"
set "ENGINE=C:\texlive\2026\bin\windows\xelatex.exe"

echo ===== Building paper_aaai2026.pdf =====

if not exist "%ENGINE%" (
    echo ERROR: xelatex not found at "%ENGINE%"
    echo Please check your TeX Live installation.
    exit /b 1
)

cd /d "%SRC%"

echo [1/4] Running xelatex (pass 1)...
"%ENGINE%" paper_aaai2026.tex > xelatex_pass1.log 2>&1

echo [2/4] Running bibtex...
bibtex paper_aaai2026 > bibtex.log 2>&1

echo [3/4] Running xelatex (pass 2 - cross-refs)...
"%ENGINE%" paper_aaai2026.tex > xelatex_pass2.log 2>&1

echo [4/4] Running xelatex (pass 3 - final)...
"%ENGINE%" paper_aaai2026.tex > xelatex_pass3.log 2>&1

if exist "%OUT%" (
    for %%A in ("%OUT%") do echo SUCCESS: paper_aaai2026.pdf - %%~nA KB = %%~zA / 1024 MB
) else (
    echo ERROR: PDF not generated. Check xelatex_pass*.log
    exit /b 1
)

echo ===== Done =====
