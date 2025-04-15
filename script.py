import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
import numpy as np

# Load pre-trained Mask R-CNN model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load and preprocess image
image = cv2.imread("Mask-RCNN-TF2/images/12283150_12d37e6389_z.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_tensor = F.to_tensor(image)
image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

# Make prediction
with torch.no_grad():
    predictions = model(image_tensor)

# Process predictions
threshold = 0.7  #Confidence threshold
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

# Visualization
output_image = image.copy()
for i, (box, label, score, mask) in enumerate(zip(boxes, labels, scores, masks)):
    # Draw bounding box
    color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
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
    label_text = f"{COCO_CLASSES[label]}: {score:.2f}"
    cv2.putText(output_image, label_text, 
               (int(box[0]), int(box[1]-5)), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Save result
cv2.imwrite("output_image.jpg", cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))