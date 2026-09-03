@echo off
setlocal
cd /d "%~dp0"

python ..\tools\prepare_horse2zebra.py --download --dataset_root "..\datasets\horse2zebra" --batch_size 8

exit /b %ERRORLEVEL%
