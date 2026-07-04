@echo off
echo === SaMST ckpt search via Windows ===
if exist "I:\GitHub\Latent_Style\Related_Works\repos\external\SaMST\checkpoint\repro_5style_train2" (
  dir /b "I:\GitHub\Latent_Style\Related_Works\repos\external\SaMST\checkpoint\repro_5style_train2"
) else (
  echo NOT_EXISTS_1
)
if exist "I:\GitHub\Latent_Style\Related_Works\repos\external\SaMST" (
  dir /b "I:\GitHub\Latent_Style\Related_Works\repos\external\SaMST"
) else (
  echo NOT_EXISTS_2
)
echo === Try alternate paths ===
if exist "I:\Github\Latent_Style\Related_Works\repos\external\SaMST\checkpoint" (
  dir /b "I:\Github\Latent_Style\Related_Works\repos\external\SaMST\checkpoint"
) else (
  echo NOT_EXISTS_3
)
echo === Try I:\GitHub case-insensitive ===
dir /b /s "I:\GitHub\Latent_Style\Related_Works\repos\external" 2>nul | findstr /i "samst" | findstr /i ".pth"
echo === Try I:\Github ===
dir /b /s "I:\Github\Latent_Style\Related_Works\repos\external" 2>nul | findstr /i "samst" | findstr /i ".pth"
echo === List exp_samam parent ===
dir /b "I:\Github\Latent_Style\exp_samam" 2>nul
echo === DONE ===
