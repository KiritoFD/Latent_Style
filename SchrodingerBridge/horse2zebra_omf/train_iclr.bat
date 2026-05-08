@echo off
setlocal
cd /d "%~dp0"

python ..\run.py --config config_iclr.json

exit /b %ERRORLEVEL%
