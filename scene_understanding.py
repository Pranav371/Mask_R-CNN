import numpy as np
import torch
import spacy
from collections import defaultdict
import logging
import os
import json
from transformers import AutoProcessor, BlipForQuestionAnswering, AutoTokenizer, AutoModelForSeq2SeqLM

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load spaCy for natural language generation
try:
    nlp = spacy.load("en_core_web_sm")
except:
    logger.warning("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Global variables for model storage
vqa_model = None
vqa_processor = None
caption_model = None
caption_tokenizer = None

# Constants
SPATIAL_RELATIONS = {
    'above': lambda obj1, obj2: obj1['center_y'] < obj2['center_y'] and overlap_horizontal(obj1, obj2),
    'below': lambda obj1, obj2: obj1['center_y'] > obj2['center_y'] and overlap_horizontal(obj1, obj2),
    'left of': lambda obj1, obj2: obj1['center_x'] < obj2['center_x'] and overlap_vertical(obj1, obj2),
    'right of': lambda obj1, obj2: obj1['center_x'] > obj2['center_x'] and overlap_vertical(obj1, obj2),
    'inside': lambda obj1, obj2: is_contained(obj1, obj2),
    'containing': lambda obj1, obj2: is_contained(obj2, obj1),
    'on top of': lambda obj1, obj2: is_on_top(obj1, obj2),
    'next to': lambda obj1, obj2: is_adjacent(obj1, obj2)
}

INTERACTION_RULES = {
    ('person', 'chair'): 'sitting on',
    ('person', 'bicycle'): 'riding',
    ('person', 'car'): 'driving',
    ('person', 'cell phone'): 'using',
    ('person', 'book'): 'reading',
    ('person', 'bottle'): 'drinking from',
    ('person', 'cup'): 'drinking from',
    ('person', 'sports ball'): 'playing with',
    ('person', 'laptop'): 'working on',
    ('cup', 'table'): 'placed on',
    ('bottle', 'table'): 'placed on',
    ('food items', 'table'): 'served on',
    ('book', 'table'): 'placed on',
    ('laptop', 'table'): 'placed on'
}

# Food items for grouping
FOOD_ITEMS = ['apple', 'banana', 'orange', 'sandwich', 'orange', 'broccoli', 'carrot', 
              'hot dog', 'pizza', 'donut', 'cake']

# Helper functions
def overlap_horizontal(obj1, obj2):
    return max(0, min(obj1['x'] + obj1['width'], obj2['x'] + obj2['width']) - max(obj1['x'], obj2['x'])) > 0

def overlap_vertical(obj1, obj2):
    return max(0, min(obj1['y'] + obj1['height'], obj2['y'] + obj2['height']) - max(obj1['y'], obj2['y'])) > 0

def is_contained(obj1, obj2):
    return (obj1['x'] >= obj2['x'] and 
            obj1['y'] >= obj2['y'] and 
            obj1['x'] + obj1['width'] <= obj2['x'] + obj2['width'] and 
            obj1['y'] + obj1['height'] <= obj2['y'] + obj2['height'])

def is_on_top(obj1, obj2):
    # Check if obj1 is on top of obj2 (overlap and obj1's bottom is close to obj2's top)
    horizontal_overlap = overlap_horizontal(obj1, obj2)
    bottom_of_obj1 = obj1['y'] + obj1['height']
    top_of_obj2 = obj2['y']
    
    return horizontal_overlap and abs(bottom_of_obj1 - top_of_obj2) < 20

def is_adjacent(obj1, obj2):
    # Check if two objects are next to each other
    distance_threshold = 50
    
    centers_distance = np.sqrt(
        (obj1['center_x'] - obj2['center_x'])**2 + 
        (obj1['center_y'] - obj2['center_y'])**2
    )
    
    return centers_distance < distance_threshold and not is_contained(obj1, obj2) and not is_contained(obj2, obj1)

def prepare_objects_data(coco_data):
    """Convert COCO format data to a more usable format for relationship detection"""
    objects = []
    
    for annotation in coco_data['annotations']:
        bbox = annotation['bbox']
        category_id = annotation['category_id']
        
        # Find category name
        category_name = "unknown"
        for cat in coco_data['categories']:
            if cat['id'] == category_id:
                category_name = cat['name']
                break
        
        x, y, width, height = bbox
        center_x = x + width / 2
        center_y = y + height / 2
        
        objects.append({
            'id': annotation['id'],
            'category': category_name,
            'x': x,
            'y': y,
            'width': width,
            'height': height,
            'center_x': center_x,
            'center_y': center_y,
            'area': width * height
        })
    
    return objects

def detect_spatial_relationships(objects):
    """Detect spatial relationships between objects"""
    relationships = []
    
    for i, obj1 in enumerate(objects):
        for j, obj2 in enumerate(objects):
            if i == j:
                continue
            
            for relation_name, relation_check in SPATIAL_RELATIONS.items():
                if relation_check(obj1, obj2):
                    relationships.append({
                        'subject': obj1['id'],
                        'subject_name': obj1['category'],
                        'relation': relation_name,
                        'object': obj2['id'],
                        'object_name': obj2['category'],
                        'confidence': 0.85  # Default confidence
                    })
    
    return relationships

def predict_interactions(objects):
    """Predict potential interactions between objects"""
    interactions = []
    
    # Group objects by category for easier lookup
    category_objects = defaultdict(list)
    for obj in objects:
        category_objects[obj['category']].append(obj)
    
    # Check predefined interaction rules
    for (subject_cat, object_cat), interaction in INTERACTION_RULES.items():
        if subject_cat in category_objects and object_cat in category_objects:
            for subj in category_objects[subject_cat]:
                for obj in category_objects[object_cat]:
                    # Check if the objects are close to each other
                    if is_adjacent(subj, obj) or is_on_top(subj, obj) or is_on_top(obj, subj):
                        interactions.append({
                            'subject': subj['id'],
                            'subject_name': subj['category'],
                            'interaction': interaction,
                            'object': obj['id'],
                            'object_name': obj['category'],
                            'confidence': 0.75
                        })
    
    # Special case for food items - group them
    food_objects = []
    for food in FOOD_ITEMS:
        if food in category_objects:
            food_objects.extend(category_objects[food])
    
    if food_objects and 'table' in category_objects:
        for table in category_objects['table']:
            interactions.append({
                'subject': food_objects[0]['id'],
                'subject_name': 'food items',
                'interaction': 'served on',
                'object': table['id'],
                'object_name': 'table',
                'confidence': 0.8,
                'group': [obj['id'] for obj in food_objects]
            })
            
    return interactions

def generate_scene_description(objects, relationships, interactions):
    """Generate a natural language description of the scene"""
    
    if not objects:
        return "No objects detected in the scene."
    
    # Start with a summary of detected objects
    object_counts = defaultdict(int)
    for obj in objects:
        object_counts[obj['category']] += 1
    
    scene_parts = []
    
    # Scene overview
    overview = "I can see "
    object_descriptions = []
    
    for category, count in object_counts.items():
        if count == 1:
            object_descriptions.append(f"a {category}")
        else:
            object_descriptions.append(f"{count} {category}s")
    
    if len(object_descriptions) == 1:
        overview += object_descriptions[0]
    elif len(object_descriptions) == 2:
        overview += f"{object_descriptions[0]} and {object_descriptions[1]}"
    else:
        overview += ", ".join(object_descriptions[:-1]) + f", and {object_descriptions[-1]}"
    
    overview += " in this scene."
    scene_parts.append(overview)
    
    # Add key relationships
    if relationships:
        important_relations = []
        added_relations = set()
        
        for rel in relationships:
            relation_key = (rel['subject'], rel['object'], rel['relation'])
            if relation_key not in added_relations:
                important_relations.append(f"The {rel['subject_name']} is {rel['relation']} the {rel['object_name']}.")
                added_relations.add(relation_key)
                
                # Limit to most important relationships to avoid verbose descriptions
                if len(important_relations) >= min(5, len(objects)):
                    break
        
        if important_relations:
            scene_parts.append(" ".join(important_relations))
    
    # Add interactions
    if interactions:
        interaction_descriptions = []
        
        for interact in interactions:
            if 'group' in interact:
                interaction_descriptions.append(f"There are several food items {interact['interaction']} the {interact['object_name']}.")
            else:
                interaction_descriptions.append(f"The {interact['subject_name']} is {interact['interaction']} the {interact['object_name']}.")
        
        if interaction_descriptions:
            scene_parts.append(" ".join(interaction_descriptions[:3]))  # Limit to top 3
    
    # Add a conclusion or interesting observation if possible
    if len(objects) > 3:
        scene_parts.append(f"It's a complex scene with {len(objects)} different objects interacting with each other.")
    elif 'person' in object_counts:
        scene_parts.append("The scene appears to involve human activity.")
    
    return " ".join(scene_parts)

def load_vqa_model(device):
    """Load the Visual Question Answering model"""
    global vqa_model, vqa_processor
    
    if vqa_model is None:
        logger.info("Loading VQA model...")
        try:
            processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
            model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
            model = model.to(device)
            
            vqa_processor = processor
            vqa_model = model
            logger.info("VQA model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading VQA model: {e}")
            return None, None
    
    return vqa_model, vqa_processor

def load_caption_model(device):
    """Load the image captioning model"""
    global caption_model, caption_tokenizer
    
    if caption_model is None:
        logger.info("Loading captioning model...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("microsoft/git-base")
            model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/git-base")
            model = model.to(device)
            
            caption_tokenizer = tokenizer
            caption_model = model
            logger.info("Captioning model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading captioning model: {e}")
            return None, None
    
    return caption_model, caption_tokenizer

def answer_question(image, question, device):
    """Answer a question about the image"""
    model, processor = load_vqa_model(device)
    
    if model is None or processor is None:
        return "Sorry, the VQA model is not available."
    
    try:
        # Prepare inputs
        inputs = processor(images=image, text=question, return_tensors="pt").to(device)
        
        # Generate answer
        with torch.no_grad():
            outputs = model.generate(**inputs)
        
        # Decode the answer
        answer = processor.decode(outputs[0], skip_special_tokens=True)
        return answer
    except Exception as e:
        logger.error(f"Error in VQA: {e}")
        return "Sorry, I couldn't process that question."

def generate_detailed_caption(image, device):
    """Generate a detailed caption for the image"""
    model, tokenizer = load_caption_model(device)
    
    if model is None or tokenizer is None:
        return "Image captioning model not available."
    
    try:
        from PIL import Image
        if not isinstance(image, Image.Image):
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            else:
                return "Invalid image format for captioning."
        
        # Resize if too large to avoid CUDA out of memory
        max_size = 800
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process image
        pixel_values = tokenizer(images=image, return_tensors="pt").pixel_values.to(device)
        
        # Generate caption
        with torch.no_grad():
            generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
            generated_caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_caption
    except Exception as e:
        logger.error(f"Error in image captioning: {e}")
        return "Could not generate a caption for this image."

def analyze_scene(image_path, coco_data, device='cpu'):
    """
    Main function to analyze a scene from an image and its COCO annotations
    
    Args:
        image_path: Path to the image file
        coco_data: COCO annotations for the image
        device: 'cpu' or 'cuda' for GPU acceleration
    
    Returns:
        Dictionary with scene analysis results
    """
    try:
        # Check if we can use CUDA
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = 'cpu'
        
        # Prepare objects data
        objects = prepare_objects_data(coco_data)
        
        if not objects:
            return {
                "error": "No objects detected in the scene",
                "scene_description": "The scene appears to be empty or objects couldn't be detected."
            }
        
        # Detect spatial relationships
        relationships = detect_spatial_relationships(objects)
        
        # Predict potential interactions
        interactions = predict_interactions(objects)
        
        # Generate scene description
        scene_description = generate_scene_description(objects, relationships, interactions)
        
        # Load image for captioning and VQA
        from PIL import Image
        image = Image.open(image_path)
        
        # Generate detailed caption
        detailed_caption = generate_detailed_caption(image, device)
        
        # Prepare common questions and answers
        common_questions = [
            "What objects are in this image?",
            "What is happening in this scene?",
            "Are there any people in this image?",
            "What is the main subject of this image?"
        ]
        
        vqa_examples = {}
        for question in common_questions:
            vqa_examples[question] = answer_question(image, question, device)
        
        # Prepare the results
        scene_analysis = {
            "objects": objects,
            "spatial_relationships": relationships,
            "potential_interactions": interactions,
            "scene_description": scene_description,
            "detailed_caption": detailed_caption,
            "vqa_examples": vqa_examples
        }
        
        return scene_analysis
        
    except Exception as e:
        logger.error(f"Error analyzing scene: {e}")
        return {
            "error": str(e),
            "scene_description": "An error occurred while analyzing this scene."
        }

def get_simplified_scene_analysis(scene_analysis):
    """
    Create a simplified version of the scene analysis for displaying in the UI
    """
    if "error" in scene_analysis:
        return {
            "success": False,
            "error": scene_analysis["error"],
            "scene_description": scene_analysis["scene_description"]
        }
    
    # Extract the essential information
    simplified = {
        "success": True,
        "scene_description": scene_analysis["scene_description"],
        "detailed_caption": scene_analysis["detailed_caption"],
        "object_count": len(scene_analysis["objects"]),
        "relationship_count": len(scene_analysis["spatial_relationships"]),
        "interaction_count": len(scene_analysis["potential_interactions"]),
        "vqa_examples": scene_analysis.get("vqa_examples", {})
    }
    
    # Add key relationships (top 5)
    key_relationships = []
    for rel in scene_analysis["spatial_relationships"][:5]:
        key_relationships.append(f"{rel['subject_name']} is {rel['relation']} {rel['object_name']}")
    simplified["key_relationships"] = key_relationships
    
    # Add key interactions (top 3)
    key_interactions = []
    for interact in scene_analysis["potential_interactions"][:3]:
        key_interactions.append(f"{interact['subject_name']} {interact['interaction']} {interact['object_name']}")
    simplified["key_interactions"] = key_interactions
    
    return simplified 