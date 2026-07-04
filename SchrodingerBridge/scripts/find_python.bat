@echo off
echo === Check Python paths ===
if exist "C:\Program Files\Python312\python.exe" (echo PF312 EXISTS) else (echo PF312 MISSING)
if exist "C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe" (echo APPDATA312 EXISTS) else (echo APPDATA312 MISSING)
where python 2>nul
where py 2>nul
echo === Try python -c ===
python -c "import sys; print(sys.executable); print(sys.version)"
echo === DONE ===
