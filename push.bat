@echo off
echo Running Clean Git Push to GitHub...
echo ===================================
cd /d C:\ClarityX
git push -f origin master:main
echo ===================================
echo Done! You can close this window now.
pause
