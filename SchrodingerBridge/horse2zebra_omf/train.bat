@echo off
setlocal
cd /d "%~dp0"

python ..\run.py --config config.json

exit /b %ERRORLEVEL%
