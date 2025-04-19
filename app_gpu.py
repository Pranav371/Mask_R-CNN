import os
import time
import cv2
import numpy as np
import torch
import torchvision
import logging
import subprocess
from torchvision.transforms import functional as F
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, Response, jsonify
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO, emit
import base64
from threading import Thread
import json
import io

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Log GPU information at the start
def log_gpu_info():
    logger.info("Checking GPU availability...")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"CUDA is available with {device_count} device(s)")
        for i in range(device_count):
            logger.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
            # Get device properties for more detailed information
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  Total memory: {props.total_memory / 1024**3:.2f} GB")
            logger.info(f"  CUDA Capability: {props.major}.{props.minor}")
    else:
        logger.warning("CUDA is not available. Running on CPU only.")

# Log GPU information at startup
log_gpu_info()

# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

app = Flask(__name__)

# Use absolute paths for directories
base_dir = os.path.abspath(os.path.dirname(__file__))
static_dir = os.path.join(base_dir, 'static')
app.config['UPLOAD_FOLDER'] = os.path.join(static_dir, 'uploads')
app.config['RESULT_FOLDER'] = os.path.join(static_dir, 'results')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
app.secret_key = 'mask_rcnn_segmentation'

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# Create necessary directories
os.makedirs(static_dir, exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load pre-trained Mask R-CNN model
# Updated to handle different PyTorch versions and move to GPU
try:
    # For newer PyTorch versions
    from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
except (ImportError, TypeError):
    # Fallback for older PyTorch versions
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

# Move model to GPU if available
model = model.to(device)
model.eval()

# COCO class labels (91 indices, with some gaps/N/A values)
# These exactly match the classes in the PyTorch torchvision implementation
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Create a list of valid class indices (excluding N/A classes)
VALID_CLASS_INDICES = [idx for idx, name in enumerate(COCO_CLASSES) if name != 'N/A' and name != '__background__']
VALID_CLASS_NAMES = [name for name in COCO_CLASSES if name != 'N/A' and name != '__background__']

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_image(image_path, threshold=0.7, selected_classes=None):
    """Process an image with Mask R-CNN using GPU acceleration."""
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor and add batch dimension
    image_tensor = F.to_tensor(image)
    image_tensor = image_tensor.unsqueeze(0)
    
    # Move tensor to GPU if available
    image_tensor = image_tensor.to(device)

    # Add a simple timing mechanism for performance comparison
    start_time = time.time()
    
    # Make prediction
    with torch.no_grad():
        predictions = model(image_tensor)
    
    inference_time = time.time() - start_time
    logger.debug(f"Inference time on {device}: {inference_time:.4f} seconds")

    # Process predictions - move results back to CPU for OpenCV processing
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    masks = predictions[0]['masks'].squeeze(1).cpu().numpy()

    # Filter predictions based on confidence and selected classes
    if selected_classes:
        selected_classes = [int(cls) for cls in selected_classes]
        keep = (scores >= threshold) & np.isin(labels, selected_classes)
    else:
        # If no specific classes selected, use all valid classes (exclude N/A)
        keep = (scores >= threshold) & np.isin(labels, VALID_CLASS_INDICES)
    
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    masks = masks[keep]

    # Visualization
    output_image = image.copy()
    
    # Count detected objects
    detected_objects = {}
    
    for i, (box, label, score, mask) in enumerate(zip(boxes, labels, scores, masks)):
        # Skip N/A classes (shouldn't happen due to earlier filtering, but just in case)
        if COCO_CLASSES[label] == 'N/A':
            continue
            
        # Generate a random color for this instance
        color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
        
        # Draw bounding box
        cv2.rectangle(output_image, 
                    (int(box[0]), int(box[1])), 
                    (int(box[2]), int(box[3])), 
                    color, 2)
        
        # Draw mask
        mask = (mask > 0.5).astype(np.uint8)
        colored_mask = np.zeros_like(output_image)
        colored_mask[:] = color
        masked = cv2.bitwise_and(colored_mask, colored_mask, mask=mask)
        output_image = cv2.addWeighted(output_image, 1, masked, 0.5, 0)
        
        # Draw label
        class_name = COCO_CLASSES[label]
        label_text = f"{class_name}: {score:.2f}"
        cv2.putText(output_image, label_text, 
                (int(box[0]), int(box[1]-5)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Count objects by class
        if class_name in detected_objects:
            detected_objects[class_name] += 1
        else:
            detected_objects[class_name] = 1

    # Convert back to BGR for saving
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    
    return output_image, detected_objects, inference_time 

def convert_video_for_web(input_path, output_path=None):
    """
    Convert video to a web-compatible format using FFmpeg if available,
    or fall back to OpenCV for conversion.
    """
    if output_path is None:
        # Use the same filename but with .mp4 extension
        base_path = os.path.splitext(input_path)[0]
        output_path = f"{base_path}_web.mp4"
    
    try:
        # Try FFmpeg first (preferred method)
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        
        # Run FFmpeg to convert the video to H.264 + AAC in MP4 container
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-c:v', 'libx264',  # H.264 video codec
            '-preset', 'fast',   # Speed/compression trade-off
            '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
            '-movflags', '+faststart',  # Optimize for web streaming
            '-y',  # Overwrite output file if it exists
            output_path
        ]
        
        logger.debug(f"Running FFmpeg conversion: {' '.join(cmd)}")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg conversion failed: {result.stderr.decode()}")
            # Fall back to OpenCV method
            return convert_video_using_opencv(input_path, output_path)
        
        logger.debug(f"Video successfully converted to web format using FFmpeg: {output_path}")
        return os.path.basename(output_path)
    
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logger.warning(f"FFmpeg not available, falling back to OpenCV: {str(e)}")
        return convert_video_using_opencv(input_path, output_path)

def convert_video_using_opencv(input_path, output_path):
    """
    Fallback method to convert video using OpenCV.
    This may be less efficient but doesn't require external dependencies.
    """
    try:
        # Open the input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error(f"Could not open input video: {input_path}")
            return None
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Try multiple codecs for maximum compatibility
        for codec_attempt in ['avc1', 'H264', 'mp4v', 'XVID']:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_attempt)
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                if out.isOpened():
                    logger.debug(f"Using codec {codec_attempt} for video conversion")
                    break
                else:
                    out.release()
            except Exception as e:
                logger.warning(f"Codec {codec_attempt} failed: {str(e)}")
        
        if not out.isOpened():
            logger.error("All codecs failed. Could not create output video writer.")
            return None
        
        # Process each frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Write the frame to the output video
            out.write(frame)
        
        # Release resources
        cap.release()
        out.release()
        
        # Check if the output file was created and has content
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            logger.error(f"OpenCV conversion failed: Output file is missing or empty: {output_path}")
            return None
            
        logger.debug(f"Video successfully converted to web format using OpenCV: {output_path}")
        return os.path.basename(output_path)
        
    except Exception as e:
        logger.error(f"Error in OpenCV video conversion: {str(e)}")
        return None

def process_video(video_path, threshold=0.7, selected_classes=None):
    """Process a video with Mask R-CNN using GPU acceleration."""
    try:
        # Open the video file
        logger.debug(f"Attempting to open video file: {video_path}")
        video = cv2.VideoCapture(video_path)
        
        if not video.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return None
        
        # Get video properties
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Check if the video has valid properties
        if width <= 0 or height <= 0 or fps <= 0:
            logger.error(f"Invalid video properties: width={width}, height={height}, fps={fps}")
            return None
        
        logger.debug(f"Video properties: width={width}, height={height}, fps={fps}, total_frames={total_frames}")
        
        # Generate output filename
        base_filename = os.path.basename(video_path)
        output_filename = f"processed_{base_filename}"
        temp_output_path = os.path.join(app.config['RESULT_FOLDER'], "temp_" + output_filename)
        final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
        
        # Create video writer with proper codec
        # Try different codecs if needed
        try:
            # Use avc1 (H.264) codec as default which has better browser compatibility
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
            out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                # Try mp4v codec if avc1 failed
                logger.warning("Failed to create video writer with avc1 codec, trying mp4v")
                out.release()
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
                
                if not out.isOpened():
                    # Try XVID as last resort
                    logger.warning("Failed to create video writer with mp4v codec, trying XVID")
                    out.release()
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out = cv2.VideoWriter(temp_output_path + '.avi', fourcc, fps, (width, height))
                    # Update output path if we switched to AVI
                    if out.isOpened():
                        temp_output_path = temp_output_path + '.avi'
                        output_filename = output_filename + '.avi'
        except Exception as e:
            logger.error(f"Error creating video writer: {str(e)}")
            return None
        
        if not out.isOpened():
            logger.error("Failed to create video writer with any codec")
            return None
        
        # Process video frame by frame
        frame_count = 0
        max_frames = min(900, total_frames)  # Limit processing to about 30 seconds at 30 fps or total frames, whichever is smaller
        
        logger.debug(f"Will process {max_frames} frames out of {total_frames} total frames")
        
        # Convert selected_classes to integers if provided
        if selected_classes:
            selected_classes = [int(cls) for cls in selected_classes]
        
        # Set up performance tracking
        total_inference_time = 0
        
        # Frame processing loop
        while True:
            ret, frame = video.read()
            if not ret or frame_count >= max_frames:
                break
            
            # Log progress every 10 frames
            if frame_count % 10 == 0:
                logger.debug(f"Processing frame {frame_count}/{max_frames} ({(frame_count/max_frames)*100:.1f}%)")
            
            try:
                # Convert to RGB for processing
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to tensor
                frame_tensor = F.to_tensor(rgb_frame)
                frame_tensor = frame_tensor.unsqueeze(0)
                
                # Move tensor to GPU if available
                frame_tensor = frame_tensor.to(device)
                
                # Start the timer
                start_time = time.time()
                
                # Make prediction
                with torch.no_grad():
                    predictions = model(frame_tensor)
                
                # Update the total inference time
                frame_inference_time = time.time() - start_time
                total_inference_time += frame_inference_time
                
                # Process predictions
                boxes = predictions[0]['boxes'].cpu().numpy()
                scores = predictions[0]['scores'].cpu().numpy()
                labels = predictions[0]['labels'].cpu().numpy()
                masks = predictions[0]['masks'].squeeze(1).cpu().numpy()
                
                # Filter predictions based on confidence and selected classes
                if selected_classes:
                    keep = (scores >= threshold) & np.isin(labels, selected_classes)
                else:
                    # If no specific classes selected, use all valid classes (exclude N/A)
                    keep = (scores >= threshold) & np.isin(labels, VALID_CLASS_INDICES)
                
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
                masks = masks[keep]
                
                # Visualization
                output_frame = rgb_frame.copy()
                
                for i, (box, label, score, mask) in enumerate(zip(boxes, labels, scores, masks)):
                    # Skip N/A classes (shouldn't happen due to earlier filtering, but just in case)
                    if COCO_CLASSES[label] == 'N/A':
                        continue
                        
                    # Generate a random color for this instance
                    color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
                    
                    # Draw bounding box
                    cv2.rectangle(output_frame, 
                                (int(box[0]), int(box[1])), 
                                (int(box[2]), int(box[3])), 
                                color, 2)
                    
                    # Draw mask
                    mask = (mask > 0.5).astype(np.uint8)
                    colored_mask = np.zeros_like(output_frame)
                    colored_mask[:] = color
                    masked = cv2.bitwise_and(colored_mask, colored_mask, mask=mask)
                    output_frame = cv2.addWeighted(output_frame, 1, masked, 0.5, 0)
                    
                    # Draw label
                    label_text = f"{COCO_CLASSES[label]}: {score:.2f}"
                    cv2.putText(output_frame, label_text, 
                            (int(box[0]), int(box[1]-5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Convert back to BGR for saving
                output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
                
                # Write the frame to the output video
                out.write(output_frame)
                
            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {str(e)}")
                # If there's an error, use the original frame
                out.write(frame)
            
            frame_count += 1
        
        # Calculate average inference time
        if frame_count > 0:
            avg_inference_time = total_inference_time / frame_count
            logger.info(f"GPU Video Processing: Average inference time per frame: {avg_inference_time:.4f} seconds")
        
        # Release video resources
        video.release()
        out.release()
        
        # Verify the output file was created correctly
        if not os.path.exists(temp_output_path):
            logger.error(f"Output video file was not created: {temp_output_path}")
            return None
            
        if os.path.getsize(temp_output_path) == 0:
            logger.error(f"Output video file is empty: {temp_output_path}")
            os.remove(temp_output_path)  # Delete the empty file
            return None
        
        logger.debug(f"Video processing complete: {frame_count} frames processed")
        logger.debug(f"Output video saved to: {temp_output_path}")
        logger.debug(f"Output file exists: {os.path.exists(temp_output_path)}")
        logger.debug(f"Output file size: {os.path.getsize(temp_output_path)} bytes")
        
        # After video processing is complete, try to convert it to a web-compatible format
        web_compatible_filename = convert_video_for_web(temp_output_path, final_output_path)
        if web_compatible_filename:
            logger.debug(f"Using web-compatible video: {web_compatible_filename}")
            return os.path.basename(final_output_path)
        
        # If conversion failed, return the original processed file
        return output_filename
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return None 

def process_frame(frame, threshold=0.7, selected_classes=None):
    """Process a single frame with Mask R-CNN for real-time processing using GPU acceleration."""
    # Convert BGR to RGB (OpenCV loads images in BGR, but PyTorch expects RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor and add batch dimension
    frame_tensor = F.to_tensor(frame_rgb)
    frame_tensor = frame_tensor.unsqueeze(0)
    
    # Move tensor to GPU if available
    frame_tensor = frame_tensor.to(device)

    # Start the timer
    start_time = time.time()
    
    # Make prediction
    with torch.no_grad():
        predictions = model(frame_tensor)
    
    # Calculate inference time
    inference_time = time.time() - start_time

    # Process predictions - move results back to CPU for OpenCV processing
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    masks = predictions[0]['masks'].squeeze(1).cpu().numpy()

    # Filter predictions based on confidence and selected classes
    if selected_classes:
        selected_classes = [int(cls) for cls in selected_classes]
        keep = (scores >= threshold) & np.isin(labels, selected_classes)
    else:
        # If no specific classes selected, use all valid classes (exclude N/A)
        keep = (scores >= threshold) & np.isin(labels, VALID_CLASS_INDICES)
    
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    masks = masks[keep]

    # Visualization
    output_frame = frame_rgb.copy()
    
    # Count detected objects
    detected_objects = {}
    
    for i, (box, label, score, mask) in enumerate(zip(boxes, labels, scores, masks)):
        # Skip N/A classes (shouldn't happen due to earlier filtering, but just in case)
        if COCO_CLASSES[label] == 'N/A':
            continue
            
        # Generate a random color for this instance
        color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
        
        # Draw bounding box
        cv2.rectangle(output_frame, 
                    (int(box[0]), int(box[1])), 
                    (int(box[2]), int(box[3])), 
                    color, 2)
        
        # Draw mask
        mask = (mask > 0.5).astype(np.uint8)
        colored_mask = np.zeros_like(output_frame)
        colored_mask[:] = color
        masked = cv2.bitwise_and(colored_mask, colored_mask, mask=mask)
        output_frame = cv2.addWeighted(output_frame, 1, masked, 0.5, 0)
        
        # Draw label
        class_name = COCO_CLASSES[label]
        label_text = f"{class_name}: {score:.2f}"
        cv2.putText(output_frame, label_text, 
                (int(box[0]), int(box[1]-5)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Count objects by class
        if class_name in detected_objects:
            detected_objects[class_name] += 1
        else:
            detected_objects[class_name] = 1
    
    # Add some performance metrics to the frame
    fps_text = f"Inference time: {inference_time*1000:.1f}ms ({1/inference_time:.1f} FPS)"
    cv2.putText(output_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    device_text = f"Using: {device}"
    cv2.putText(output_frame, device_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Convert back to BGR for OpenCV display
    output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
    
    return output_frame, detected_objects, inference_time 

@app.route('/')
def index():
    # Pass the valid classes to the template
    class_data = [(idx, name) for idx, name in enumerate(COCO_CLASSES) 
                 if name != 'N/A' and name != '__background__']
    # Pass GPU information to the template
    gpu_info = {
        'available': torch.cuda.is_available(),
        'device': str(device),
        'name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
    }
    return render_template('index.html', class_data=class_data, gpu_info=gpu_info)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    threshold = float(request.form.get('threshold', 0.7))
    selected_classes = request.form.getlist('selected_classes[]')
    
    # If no classes selected, process all valid classes
    if not selected_classes:
        selected_classes = None
        selected_class_names = VALID_CLASS_NAMES
        filtered_classes = False
    else:
        # Get class names for the selected classes
        selected_class_names = [COCO_CLASSES[int(cls)] for cls in selected_classes 
                              if cls.isdigit() and int(cls) < len(COCO_CLASSES) 
                              and COCO_CLASSES[int(cls)] != 'N/A']
        filtered_classes = True
        
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        file_extension = filename.rsplit('.', 1)[1].lower()
        
        # Process based on file type
        if file_extension in {'png', 'jpg', 'jpeg'}:
            # Process image
            start_time = time.time()
            output_image, detected_objects, inference_time = process_image(file_path, threshold, selected_classes)
            processing_time = round(time.time() - start_time, 2)
            
            # Save the processed image
            output_filename = f"processed_{filename}"
            output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
            cv2.imwrite(output_path, output_image)
            
            logger.debug(f"Original file saved at: {file_path}")
            logger.debug(f"Processed file saved at: {output_path}")
            logger.debug(f"Original file exists: {os.path.exists(file_path)}")
            logger.debug(f"Processed file exists: {os.path.exists(output_path)}")
            
            # Calculate total objects
            total_objects = sum(detected_objects.values()) if detected_objects else 0
            
            # Add GPU performance metrics
            gpu_metrics = {
                'device': str(device),
                'inference_time': round(inference_time * 1000, 2),  # Convert to milliseconds
                'fps': round(1 / inference_time, 1) if inference_time > 0 else 0
            }
            
            return render_template('result.html', 
                                  original_file=filename,
                                  result_file=output_filename,
                                  file_type='image',
                                  class_counts=detected_objects,
                                  threshold=threshold,
                                  selected_classes=selected_class_names,
                                  total_classes=len(VALID_CLASS_NAMES),
                                  total_objects=total_objects,
                                  processing_time=processing_time,
                                  gpu_metrics=gpu_metrics)
        
        elif file_extension in {'mp4', 'avi', 'mov'}:
            # Process video
            try:
                # Before processing, check if the video file is valid
                video_check = cv2.VideoCapture(file_path)
                if not video_check.isOpened():
                    logger.error(f"Could not open video file for checking: {file_path}")
                    flash("Could not open the uploaded video file. The file may be corrupted or in an unsupported format.")
                    return redirect(url_for('index'))
                
                # Check video properties to warn about large videos
                width = int(video_check.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(video_check.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = video_check.get(cv2.CAP_PROP_FPS)
                total_frames = int(video_check.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames / fps if fps > 0 else 0
                
                logger.debug(f"Video properties check: {width}x{height}, {fps} fps, {total_frames} frames, {duration:.1f} seconds")
                
                # Release the video check
                video_check.release()
                
                # If the video is very large, warn the user
                if total_frames > 900:  # More than our max_frames
                    logger.info(f"Large video detected: {total_frames} frames, will process only first 900 frames")
                    flash(f"Your video is {duration:.1f} seconds long. For performance reasons, only the first 30 seconds will be processed.")
                
                start_time = time.time()
                output_filename = process_video(file_path, threshold, selected_classes)
                processing_time = round(time.time() - start_time, 2)
                
                if output_filename:
                    logger.debug(f"Video processing completed in {processing_time} seconds")
                    
                    # Check if output file exists and has content
                    output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
                    if not os.path.exists(output_path):
                        logger.error(f"Output file doesn't exist: {output_path}")
                        flash("Error: The processed video file was not created. Please try again with a different video.")
                        return redirect(url_for('index'))
                    
                    if os.path.getsize(output_path) == 0:
                        logger.error(f"Output file is empty: {output_path}")
                        flash("Error: The processed video file is empty. Please try again with a different video.")
                        return redirect(url_for('index'))
                    
                    # Add GPU metrics
                    gpu_metrics = {
                        'device': str(device),
                        'total_time': processing_time,
                        'fps': round(total_frames / processing_time, 1) if processing_time > 0 else 0
                    }
                    
                    # For video, we don't have object counts
                    return render_template('result.html',
                                        original_file=filename,
                                        result_file=output_filename,
                                        file_type='video',
                                        class_counts={},
                                        threshold=threshold,
                                        selected_classes=selected_class_names,
                                        total_classes=len(VALID_CLASS_NAMES),
                                        total_objects=0,
                                        processing_time=processing_time,
                                        gpu_metrics=gpu_metrics)
                else:
                    logger.error("Video processing returned None")
                    flash('Error processing video. Please check the server logs for details.')
                    return redirect(url_for('index'))
                    
            except Exception as e:
                logger.error(f"Exception during video processing: {str(e)}")
                flash(f'Error processing video: {str(e)}')
                return redirect(url_for('index'))
    
    flash('Invalid file type')
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    logger.debug(f"Accessing uploaded file: {file_path}")
    logger.debug(f"File exists: {os.path.exists(file_path)}")
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    file_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    logger.debug(f"Accessing result file: {file_path}")
    logger.debug(f"File exists: {os.path.exists(file_path)}")
    
    # Set the correct MIME type for videos
    if filename.lower().endswith('.mp4'):
        return send_from_directory(app.config['RESULT_FOLDER'], filename, mimetype='video/mp4')
    elif filename.lower().endswith(('.avi')):
        return send_from_directory(app.config['RESULT_FOLDER'], filename, mimetype='video/x-msvideo')
    elif filename.lower().endswith('.mov'):
        return send_from_directory(app.config['RESULT_FOLDER'], filename, mimetype='video/quicktime')
    
    # Default behavior for other file types
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/camera')
def camera():
    # Pass the valid classes to the template
    class_data = [(idx, name) for idx, name in enumerate(COCO_CLASSES) 
                 if name != 'N/A' and name != '__background__']
    # Pass GPU information to the template
    gpu_info = {
        'available': torch.cuda.is_available(),
        'device': str(device),
        'name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
    }
    return render_template('camera.html', class_data=class_data, gpu_info=gpu_info)

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')
    # Send GPU info to client
    emit('gpu_info', {
        'available': torch.cuda.is_available(),
        'device': str(device),
        'name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
    })

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

@socketio.on('process_frame')
def handle_process_frame(data):
    try:
        # Get the threshold and selected classes from the data
        threshold = float(data.get('threshold', 0.7))
        selected_classes = data.get('classes', None)
        
        # Ensure selected_classes is properly formatted for processing
        if selected_classes and len(selected_classes) > 0:
            # Make sure all class values are integers
            try:
                selected_classes = [int(cls) for cls in selected_classes if cls and cls.isdigit()]
                # Filter out any invalid class indices
                selected_classes = [cls for cls in selected_classes if cls < len(COCO_CLASSES) and COCO_CLASSES[cls] != 'N/A']
                if not selected_classes:  # If none valid after filtering, use all valid classes
                    selected_classes = VALID_CLASS_INDICES
            except (ValueError, TypeError) as e:
                logger.error(f"Error processing class indices: {str(e)}")
                selected_classes = VALID_CLASS_INDICES  # Fall back to all valid classes
        else:
            selected_classes = VALID_CLASS_INDICES  # Use all valid classes if none provided
        
        # Decode the base64 image
        image_data = data.get('image', '')
        if not image_data:
            raise ValueError("No image data received")
            
        # Handle potential formats: with or without the data URL prefix
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        try:
            # First attempt with regular decoding
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except cv2.error as e:
            logger.warning(f"Memory error during decoding, trying lower quality: {str(e)}")
            # Fallback 1: Try to decode at reduced size
            try:
                # Decode at reduced size using IMREAD_REDUCED_COLOR_2 flag (half resolution)
                frame = cv2.imdecode(nparr, cv2.IMREAD_REDUCED_COLOR_2)
            except Exception:
                # Fallback 2: Skip this frame entirely
                logger.error("Failed to decode image even at reduced size, skipping frame")
                emit('error', {'message': "Image decoding failed. Try lowering camera resolution or image quality."})
                return
        
        if frame is None:
            raise ValueError("Failed to decode image")
        
        # Resize frame if it's too large (memory optimization)
        max_dim = 480  # Maximum dimension for processing
        height, width = frame.shape[:2]
        if height > max_dim or width > max_dim:
            # Calculate new dimensions while maintaining aspect ratio
            if height > width:
                new_height = max_dim
                new_width = int(width * (max_dim / height))
            else:
                new_width = max_dim
                new_height = int(height * (max_dim / width))
                
            # Resize the frame
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logger.debug(f"Resized frame from {width}x{height} to {new_width}x{new_height}")
        
        try:
            # Process the frame
            processed_frame, detected_objects, inference_time = process_frame(frame, threshold, selected_classes)
            
            # Convert the processed frame to base64 for sending back to client
            _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])  # Lower quality (70 instead of 80)
            processed_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Send the processed frame and detected objects back to the client
            emit('processed_frame', {
                'image': processed_base64,
                'detected_objects': detected_objects,
                'performance': {
                    'device': str(device),
                    'inference_time_ms': round(inference_time * 1000, 2),
                    'fps': round(1 / inference_time, 1) if inference_time > 0 else 0
                }
            })
        except cv2.error as e:
            if "Failed to allocate" in str(e) or "Insufficient memory" in str(e):
                logger.error(f"Memory allocation error during processing: {str(e)}")
                emit('error', {'message': "Not enough memory to process this frame. Try reducing resolution or quality."})
            else:
                raise  # Re-raise if it's a different OpenCV error
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        # Include traceback for more detailed debugging
        import traceback
        logger.error(traceback.format_exc())
        emit('error', {'message': str(e)})

# GPU status endpoint
@app.route('/gpu_status')
def gpu_status():
    """Return GPU status information as JSON"""
    if torch.cuda.is_available():
        # Get GPU memory information
        total_memory = torch.cuda.get_device_properties(0).total_memory
        reserved_memory = torch.cuda.memory_reserved(0)
        allocated_memory = torch.cuda.memory_allocated(0)
        free_memory = total_memory - allocated_memory
        
        # Return JSON response
        return jsonify({
            'available': True,
            'device_name': torch.cuda.get_device_name(0),
            'device': str(device),
            'cuda_version': torch.version.cuda,
            'total_memory_gb': round(total_memory / (1024**3), 2),
            'free_memory_gb': round(free_memory / (1024**3), 2),
            'reserved_memory_gb': round(reserved_memory / (1024**3), 2),
            'allocated_memory_gb': round(allocated_memory / (1024**3), 2),
            'memory_usage_percent': round((allocated_memory / total_memory) * 100, 2)
        })
    else:
        return jsonify({
            'available': False,
            'device': 'cpu'
        })

if __name__ == '__main__':
    logger.info(f"Starting server with device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        
        # Get GPU memory information
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        logger.info(f"Total GPU memory: {total_memory:.2f} GB")
    
    socketio.run(app, debug=True)