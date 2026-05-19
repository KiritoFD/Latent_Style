@echo off
setlocal

cd /d "%~dp0"

echo Running clip_style decision-tree experiments...
echo Root: %CD%
echo.

python run_clip_style_decision_tree.py --train --eval-main --eval-topk --summarize %*

set EXIT_CODE=%ERRORLEVEL%
echo.
if not "%EXIT_CODE%"=="0" (
  echo Decision-tree run failed with exit code %EXIT_CODE%.
) else (
  echo Decision-tree run finished.
  echo Summary: exp\decision_tree_clip_style\decision_tree_results.csv
)

exit /b %EXIT_CODE%
