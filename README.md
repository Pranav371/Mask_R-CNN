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
- **NEW: Annotation Editor for creating and editing object annotations**
- **NEW: Export annotations in multiple formats (COCO JSON, Pascal VOC, YOLO)**
- **NEW: Real-time annotation editing with visual tools**
- **NEW: Scene Understanding and Relationship Analysis**
- **NEW: Visual Question Answering (VQA) for images**

## Annotation Editor

The new Annotation Editor feature allows you to:

1. **Edit and create object annotations:**
   - View automatically generated annotations from the Mask R-CNN model
   - Add new annotations manually
   - Edit existing bounding boxes and object classifications
   - Delete unwanted annotations

2. **Annotation Tools:**
   - Move Tool: Pan and navigate around the image
   - Box Tool: Create and modify bounding boxes
   - Mask Tool: Edit segmentation masks
   - Eraser Tool: Remove parts of segmentation masks
   - Zoom controls for detailed editing

3. **Export Annotations in Multiple Formats:**
   - **COCO JSON:** Standard format for instance segmentation tasks
   - **Pascal VOC:** XML-based format used by many object detection frameworks
   - **YOLO:** Simple text format for real-time object detection

4. **Cross-Platform Support:**
   - Works on desktop and mobile devices
   - Responsive design adapts to different screen sizes
   - GPU-accelerated for faster processing when available

To use the Annotation Editor:
1. Upload and process an image
2. On the results page, click the "Edit Annotations" button
3. Use the editing tools to modify annotations as needed
4. Select your preferred export format and click "Export Annotations"

This feature is particularly useful for:
- Creating custom training datasets
- Correcting model predictions
- Exporting annotations for use in other machine learning projects
- Fine-tuning detection results before further processing

## Scene Understanding

The new Scene Understanding feature enhances image analysis by going beyond simple object detection:

1. **Relationship Detection:**
   - Automatically identifies spatial relationships between objects (above, below, inside, next to)
   - Recognizes common interaction patterns (e.g., person sitting on chair, book placed on table)
   - Visualizes connections between objects in the scene

2. **Natural Language Scene Descriptions:**
   - Generates detailed textual descriptions of the entire scene
   - Highlights key objects and their relationships
   - Provides comprehensive captions about the scene content

3. **Visual Question Answering:**
   - Ask any question about the image content
   - Get AI-powered answers about objects, activities, and relationships
   - Includes common pre-set questions and the ability to ask custom questions

4. **GPU-Accelerated Analysis:**
   - Utilizes GPU (when available) for faster processing
   - Shows detailed performance metrics
   - Optimized for both CPU and GPU environments

To use the Scene Understanding feature:
1. Process an image with Mask R-CNN detection
2. On the results page, click the "Scene Understanding" button
3. Explore the detected relationships, descriptions, and ask questions about the image

This feature is particularly valuable for:
- Content analysis and understanding
- Accessibility applications (image descriptions for visually impaired)
- Research and data analysis of image content
- Automated image captioning and tagging

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

## Instructions for Mac and Linux Users

While the project provides .bat files for Windows users, Mac and Linux users can follow these steps to install and run the application:

### Installation

1. Create a virtual environment (recommended):
```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac

# Or using conda
conda create -n maskrcnn python=3.8
conda activate maskrcnn
```

2. Install the required dependencies:
```bash
pip3 install -r requirements.txt
```

### Running the Application

1. Start the Flask server:
```bash
# Standard CPU version
python3 app.py

# GPU-accelerated version (if you have a compatible NVIDIA GPU)
python3 app_gpu.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

### GPU Acceleration on Mac/Linux

For GPU acceleration on Linux:
1. Ensure you have CUDA and cuDNN properly installed
2. Install PyTorch with CUDA support:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For Mac users with Apple Silicon (M1/M2/M3):
1. PyTorch now supports GPU acceleration on Apple Silicon
2. Install the appropriate version with MPS (Metal Performance Shaders) support:
```bash
pip3 install torch torchvision torchaudio
```
3. The code will automatically use MPS if available when running on Apple Silicon

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
   - Access the Annotation Editor to modify or export annotations
   - Use Scene Understanding to analyze relationships between objects
   - Process another file
5. Annotation Editor functionality:
   - When viewing an image result, click the "Edit Annotations" button
   - Use the provided tools to modify existing annotations or add new ones
   - Export annotations in your preferred format (COCO JSON, Pascal VOC, or YOLO)
   - Download the exported annotations for use in other ML projects
6. Scene Understanding functionality:
   - When viewing an image result, click the "Scene Understanding" button
   - Review automatically generated scene descriptions and relationships
   - Explore spatial relationships and interactions between objects
   - Ask questions about the image content using the Visual QA feature

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
- All annotation and export features available in the CPU version, accelerated by GPU

**Note:** The annotation and export functionality is fully implemented in both the CPU (`app.py`) and GPU (`app_gpu.py`) versions, providing a consistent experience regardless of which version you choose to run. The GPU version offers faster processing for generating the initial annotations.

For complete instructions, requirements, and troubleshooting for GPU acceleration, please refer to [README_GPU.md](README_GPU.md).

## License

This project is licensed under the MIT License - see the LICENSE file for details. 