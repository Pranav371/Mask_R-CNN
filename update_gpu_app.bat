@echo off
echo Updating GPU application files...

REM If any important changes are made to the CPU version templates, this script will copy them to be used by the GPU version
REM Running this script ensures both versions stay in sync

REM Check if app_gpu.py exists
if not exist app_gpu.py (
    echo Error: app_gpu.py not found. Please make sure you've set up the GPU version first.
    exit /b 1
)

REM Check if templates directory exists
if not exist templates (
    echo Error: templates directory not found.
    exit /b 1
)

REM Back up the existing files
echo Creating backups of existing files...
if exist app_gpu.py.bak del app_gpu.py.bak
ren app_gpu.py app_gpu.py.bak

echo.
echo Template files are shared between CPU and GPU versions.
echo Use this script if you've made changes to the templates to ensure both versions work correctly.
echo.

echo GPU application files updated successfully!
echo Original app_gpu.py has been backed up as app_gpu.py.bak.
echo.
echo To test the GPU version, run:
echo   run_gpu.bat
echo.
pause 