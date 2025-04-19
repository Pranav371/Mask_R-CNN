# Mask R-CNN Web Interface for Instance Segmentation

This is a Flask web application that provides a user-friendly interface for performing instance segmentation using the pre-trained Mask R-CNN model from PyTorch's torchvision library.

## Features

- Upload and process images (JPG, PNG, JPEG)
- Upload and process videos (MP4, AVI, MOV)
- Interactive confidence threshold adjustment
- Display of segmented objects with colored masks
- Summary of detected objects for images
- Side-by-side comparison of original and processed files
- Download capabilities for processed results

## Requirements

The application requires Python 3.8+ and the following packages:
- Flask
- PyTorch
- torchvision
- OpenCV
- NumPy

You can install all dependencies using:

```bash
pip install -r requirements.txt
```
Or on Windows, you can use the provided batch file:
```
install_dependencies.bat
```

## Setup and Running

1. Clone this repository:

```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. **Important Note on Model Weights**: 
   - The Mask R-CNN model weights file (`mask_rcnn_coco.h5`, ~246MB) is not included in the repository due to its size.
   - You don't need to download this file separately when using our app, as it automatically uses PyTorch's pre-trained model.
   - If you're working with the TensorFlow implementation, you can download the weights from the [Mask R-CNN releases page](https://github.com/matterport/Mask_RCNN/releases).

4. Run the Flask application:

```bash
python app.py
```
Or on Windows, you can use the provided batch file:
```
run.bat
```

5. Open your web browser and navigate to:

```
http://localhost:5000
```

## Project Structure

```
├── app.py                  # Main Flask application
├── script.py               # Utility script for processing
├── requirements.txt        # Package dependencies
├── install_dependencies.bat # Windows batch file for installing dependencies
├── run.bat                 # Windows batch file for running the application
├── static/                 # Static files directory
│   ├── uploads/            # Stores uploaded image and video files
│   └── results/            # Stores processed image and video files
└── templates/              # HTML templates for the web interface
```

## File Management

- All uploaded and processed files are stored in the `static/uploads/` and `static/results/` directories respectively
- These media files (images and videos) are excluded from version control via .gitignore
- Only the directory structure is maintained in the repository using .gitkeep files

## Usage

1. On the main page, upload an image or video file using the drag-and-drop interface or file browser
2. Adjust the confidence threshold slider if needed (default is 0.7)
3. Click "Process" to start the instance segmentation
4. View the results on the results page, where you can:
   - Switch between the processed and original views
   - See a summary of detected objects (for images)
   - Download the processed image (for images)
   - Process another file

## Model Information

This application uses Mask R-CNN with a ResNet-50-FPN backbone, pre-trained on the COCO dataset. The model can detect and segment 80 different object categories.

### About the Model Weights
- The PyTorch implementation automatically downloads the required model weights when first used
- The weights files are excluded from Git tracking via .gitignore

## Technical Details

- The application uses PyTorch's torchvision implementation of Mask R-CNN
- For video processing, we process frame-by-frame with a limit on the number of frames
- Segmentation masks are visualized with semi-transparent colored overlays
- Object detection confidence threshold can be adjusted from 0.1 to 1.0

## Limitations

- Video processing is limited to approximately 30 seconds (900 frames at 30 fps) to avoid excessive processing time
- The processing happens on the server, so performance depends on your hardware
- For optimal performance, a GPU with CUDA support is recommended for faster inference

## GPU Acceleration

For faster processing and improved performance, this application supports GPU acceleration using CUDA. Using a GPU can significantly reduce processing times, especially for videos and real-time camera detection.

**To use the GPU-accelerated version:**

1. Ensure you have a compatible NVIDIA GPU (RTX 3050 or better recommended)
2. Follow the detailed setup and usage instructions in the [GPU README](README_GPU.md)
3. Run the application using the `run_gpu.bat` script instead of the standard one

The GPU version includes:
- Significantly faster processing times
- Real-time performance metrics 
- Detailed information about GPU usage
- Visual indicators showing GPU utilization

For complete instructions, requirements, and troubleshooting for GPU acceleration, please refer to [README_GPU.md](README_GPU.md).

## License

This project is licensed under the MIT License - see the LICENSE file for details. 