@echo off
echo Installing dependencies for Mask R-CNN Web Interface...

:: Install base dependencies first
pip install flask==2.3.3 werkzeug==2.3.7 numpy==1.24.3 opencv-python==4.8.1.78

:: Install PyTorch with the correct version for CPU (Windows)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo =======================================================
echo IMPORTANT: This application requires FFmpeg for optimal video processing.
echo Please install FFmpeg using one of these methods:
echo.
echo 1. Download FFmpeg from https://ffmpeg.org/download.html
echo 2. Extract the downloaded archive
echo 3. Add the bin folder to your system PATH
echo.
echo Alternatively, you can use a package manager:
echo - Chocolatey: choco install ffmpeg
echo - Scoop: scoop install ffmpeg
echo =======================================================
echo.
echo All dependencies installed!
echo Run the application with "run.bat" 