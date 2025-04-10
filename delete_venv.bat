@echo off
echo This will delete the virtual environment folder (.venv)
echo All installed packages will be removed.
echo.
set /p confirm=Are you sure you want to proceed? (y/n): 
if /i "%confirm%" neq "y" goto :end

echo.
echo Deleting virtual environment...
rmdir /s /q .venv
echo.
echo Virtual environment deleted.

:end
pause
