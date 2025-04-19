# GPU-Accelerated Mask R-CNN for Object Detection and Segmentation

This README provides instructions on how to set up and use the GPU-accelerated version of the Mask R-CNN application for improved performance.

## System Requirements

- Windows 10 or newer
- NVIDIA GPU (RTX 3050 4GB or better)
- Up-to-date NVIDIA drivers
- Anaconda or Miniconda installed

## Setup Instructions

### 1. Install NVIDIA Drivers and CUDA Toolkit

Ensure you have the latest NVIDIA drivers installed for your GPU. You can download them from the [NVIDIA Driver Download](https://www.nvidia.com/Download/index.aspx) page.

### 2. Set Up the GPU Environment

Run the provided setup script to create a dedicated conda environment with GPU support:

```bash
setup_gpu_env.bat
```

This script will:
- Create a new conda environment named `maskrcnn_gpu`
- Install PyTorch with CUDA support
- Install all necessary dependencies
- Create a run script for the GPU-accelerated application
- Verify CUDA is properly installed and available

### 3. Running the Application

After setup is complete, you can start the GPU-accelerated version by running:

```bash
run_gpu.bat
```

This will launch the application using your GPU for inference, which should significantly speed up image and video processing compared to the CPU-only version.

## Performance Comparison

The GPU-accelerated version includes performance metrics that show:
- Inference time per frame (in milliseconds)
- Frames per second (FPS)
- GPU memory usage

You should see a significant performance improvement compared to the CPU version, especially for video processing and real-time camera analysis.

## Troubleshooting

If you encounter issues with GPU acceleration:

1. **CUDA not available**: Make sure your NVIDIA drivers are up-to-date
2. **Out of memory errors**: 
   - Try processing smaller images or videos
   - Reduce the detection confidence threshold to process fewer objects
   - Select specific classes to detect instead of all classes

3. **Slow performance**: 
   - Close other GPU-intensive applications
   - Restart the application to free GPU memory

## Switching Between CPU and GPU Versions

You can easily switch between the CPU and GPU versions depending on your needs:

- For CPU version: Use the original `run.bat` script
- For GPU version: Use the `run_gpu.bat` script

Both versions can exist side-by-side without interfering with each other.

## Additional Information

The GPU version of the application includes:
- Real-time performance metrics displayed on processed images and videos
- A `/gpu_status` endpoint that provides detailed information about GPU usage
- Visual indicators showing which device (CPU/GPU) is being used for processing

## Known Limitations

- The RTX 3050 with 4GB of VRAM may struggle with processing very high-resolution videos or images
- For large videos, the application will process only the first 30 seconds to preserve memory
- Real-time camera processing may be limited by available GPU memory 