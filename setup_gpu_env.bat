@echo off
echo Setting up GPU-accelerated environment for Mask R-CNN...

REM Check if conda is available
call conda --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Conda is not installed or not in PATH
    echo Please install Anaconda or Miniconda first
    pause
    exit /b 1
)

REM Create new conda environment for GPU processing
echo Creating new conda environment: maskrcnn_gpu
call conda create -n maskrcnn_gpu python=3.10 -y
if %ERRORLEVEL% NEQ 0 (
    echo Error creating conda environment
    pause
    exit /b 1
)

REM Activate the environment and install PyTorch with CUDA support
echo Installing PyTorch with CUDA support...
call conda activate maskrcnn_gpu
call conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
if %ERRORLEVEL% NEQ 0 (
    echo Warning: PyTorch installation with CUDA 12.1 failed, trying CUDA 11.8...
    call conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    if %ERRORLEVEL% NEQ 0 (
        echo Error installing PyTorch with GPU support
        pause
        exit /b 1
    )
)

REM Install other dependencies
echo Installing other dependencies...
call pip install flask==2.3.3 werkzeug==2.3.7 numpy==1.24.3 opencv-python==4.8.1.78 flask-socketio

REM Verify CUDA is available with PyTorch
echo Verifying CUDA availability with PyTorch...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count()); print('CUDA device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

REM Create run_gpu.bat file
echo Creating GPU run script...
echo @echo off > run_gpu.bat
echo call conda activate maskrcnn_gpu >> run_gpu.bat
echo python app_gpu.py >> run_gpu.bat
echo pause >> run_gpu.bat

echo.
echo Environment setup complete!
echo To run the application with GPU support:
echo 1. Make sure NVIDIA drivers are up to date
echo 2. Run run_gpu.bat to start the application
echo.
echo Press any key to exit...
pause 