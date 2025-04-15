import os
import time
import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
app.secret_key = 'mask_rcnn_segmentation'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load pre-trained Mask R-CNN model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# COCO class labels (81 classes including background)
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

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_image(image_path, threshold=0.7):
    """Process an image with Mask R-CNN."""
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor and add batch dimension
    image_tensor = F.to_tensor(image)
    image_tensor = image_tensor.unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        predictions = model(image_tensor)

    # Process predictions
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    masks = predictions[0]['masks'].cpu().numpy()

    # Filter predictions based on confidence
    keep = scores >= threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    masks = masks[keep]

    # Visualization
    output_image = image.copy()
    
    # Count detected objects
    detected_objects = {}
    
    for i, (box, label, score, mask) in enumerate(zip(boxes, labels, scores, masks)):
        # Generate a random color for this instance
        color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
        
        # Draw bounding box
        cv2.rectangle(output_image, 
                    (int(box[0]), int(box[1])), 
                    (int(box[2]), int(box[3])), 
                    color, 2)
        
        # Draw mask
        mask = mask[0]  # Remove channel dimension
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
    
    return output_image, detected_objects

def process_video(video_path, threshold=0.7):
    """Process a video with Mask R-CNN."""
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        return None
    
    # Get video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # Generate output filename
    base_filename = os.path.basename(video_path)
    output_filename = f"processed_{base_filename}"
    output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process video frame by frame
    frame_count = 0
    max_frames = 900  # Limit processing to about 30 seconds at 30 fps
    
    while True:
        ret, frame = video.read()
        if not ret or frame_count >= max_frames:
            break
            
        # Convert to RGB for processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        frame_tensor = F.to_tensor(rgb_frame)
        frame_tensor = frame_tensor.unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            predictions = model(frame_tensor)
        
        # Process predictions
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        masks = predictions[0]['masks'].cpu().numpy()
        
        # Filter predictions based on confidence
        keep = scores >= threshold
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        masks = masks[keep]
        
        # Visualization
        output_frame = rgb_frame.copy()
        
        for i, (box, label, score, mask) in enumerate(zip(boxes, labels, scores, masks)):
            # Generate a random color for this instance
            color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
            
            # Draw bounding box
            cv2.rectangle(output_frame, 
                        (int(box[0]), int(box[1])), 
                        (int(box[2]), int(box[3])), 
                        color, 2)
            
            # Draw mask
            mask = mask[0]  # Remove channel dimension
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
        
        frame_count += 1
    
    # Release video resources
    video.release()
    out.release()
    
    return output_filename

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    threshold = float(request.form.get('threshold', 0.7))
    
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
            output_image, detected_objects = process_image(file_path, threshold)
            
            # Save the processed image
            output_filename = f"processed_{filename}"
            output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
            cv2.imwrite(output_path, output_image)
            
            return render_template('result.html', 
                                  original=filename,
                                  processed=output_filename,
                                  is_video=False,
                                  objects=detected_objects)
        
        elif file_extension in {'mp4', 'avi', 'mov'}:
            # Process video
            output_filename = process_video(file_path, threshold)
            
            if output_filename:
                return render_template('result.html',
                                      original=filename,
                                      processed=output_filename,
                                      is_video=True,
                                      objects=None)
            else:
                flash('Error processing video')
                return redirect(url_for('index'))
    
    flash('Invalid file type')
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True) 