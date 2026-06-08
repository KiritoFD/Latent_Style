@echo off
setlocal

set "SRC=%~dp0"
if "%SRC:~-1%"=="\" set "SRC=%SRC:~0,-1%"
set "OUT=%SRC%\supplement_aaai2027.pdf"
set "ENGINE=C:\texlive\2026\bin\windows\pdflatex.exe"

echo ===== Building supplement_aaai2027.pdf =====

if not exist "%ENGINE%" (
    echo ERROR: pdflatex not found at "%ENGINE%"
    exit /b 1
)

cd /d "%SRC%"

echo [1/2] Running pdflatex (pass 1)...
"%ENGINE%" -interaction=nonstopmode -halt-on-error supplement_aaai2027.tex > supplement_pdflatex_pass1.log 2>&1

echo [2/2] Running pdflatex (pass 2)...
"%ENGINE%" -interaction=nonstopmode -halt-on-error supplement_aaai2027.tex > supplement_pdflatex_pass2.log 2>&1

if exist "%OUT%" (
    for %%A in ("%OUT%") do echo SUCCESS: supplement_aaai2027.pdf - %%~zA bytes
) else (
    echo ERROR: Supplement PDF not generated. Check supplement_pdflatex_pass*.log
    exit /b 1
)

echo ===== Done =====
