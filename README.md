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
   - If you're working with the TensorFlow implementation (in the Mask-RCNN-TF2 directory), you can download the weights from the [Mask R-CNN releases page](https://github.com/matterport/Mask_RCNN/releases).

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
- The original TensorFlow implementation requires the `mask_rcnn_coco.h5` weights file (~246MB)
- These weight files are excluded from Git tracking via .gitignore

## Technical Details

- The application uses PyTorch's torchvision implementation of Mask R-CNN
- For video processing, we process frame-by-frame with a limit on the number of frames
- Segmentation masks are visualized with semi-transparent colored overlays
- Object detection confidence threshold can be adjusted from 0.1 to 1.0

## Limitations

- Video processing is limited to approximately 30 seconds (900 frames at 30 fps) to avoid excessive processing time
- The processing happens on the server, so performance depends on your hardware
- For optimal performance, a GPU with CUDA support is recommended for faster inference

## License

This project is licensed under the MIT License - see the LICENSE file for details. 