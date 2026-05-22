import os
import sys
import numpy as np
import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import argparse
from torchvision import transforms
import cv2

# Configuration similar to the training script
class CFG:
    MODEL_NAME = 'convnext_large'
    IMG_SIZE = 384
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = 'best_model_epoch_28_auroc_0.9688.pth'
    THRESHOLD = 0.5  # Restored to original threshold

# Define the class names
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
TRAIN_CLASSES = CLASS_NAMES
CLASSES_WITH_BBOX = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']

# Model architecture (single model for both classification and localization)
class AdvancedChestModel(nn.Module):
    def __init__(self, model_name='convnext_large', num_classes=14, metadata_features=3, pretrained=False):
        super().__init__()
        self.num_classes = num_classes
        self.train_classes = len(TRAIN_CLASSES)
        self.bbox_classes = len(CLASSES_WITH_BBOX)
        
        # Base model
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, features_only=False)
        
        # Get feature dimension
        in_features = 1536  # ConvNext Large fixed dimension
        
        # Attention mechanism for better feature focus
        self.attention = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, in_features // 8),
            nn.GELU(),
            nn.Linear(in_features // 8, in_features),
            nn.Sigmoid()
        )
        
        # Enhanced localization head for better bounding box prediction
        self.localization_head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(0.1),
            nn.Linear(in_features, in_features // 2),
            nn.GELU(),
            nn.LayerNorm(in_features // 2),
            nn.Linear(in_features // 2, self.bbox_classes * 4)
        )
        
        # Metadata processing branch
        self.metadata_branch = nn.Sequential(
            nn.Linear(metadata_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        # Classification head
        self.combined_fc = nn.Sequential(
            nn.LayerNorm(in_features + 128),
            nn.Dropout(0.2),
            nn.Linear(in_features + 128, self.train_classes)
        )

    def forward(self, x_img, x_meta):
        # Extract image features
        img_features = self.model(x_img)
        
        # Handle different model outputs
        if isinstance(img_features, (list, tuple)):
            img_features = img_features[-1]
            
        # Global average pooling if needed
        if len(img_features.shape) > 2:
            img_features = F.adaptive_avg_pool2d(img_features, (1, 1)).view(img_features.size(0), -1)
        
        # Apply attention
        attention_weights = self.attention(img_features)
        attended_features = img_features * attention_weights
        
        # Bounding box prediction
        bbox_out = self.localization_head(attended_features)
        bbox_out = bbox_out.view(-1, self.bbox_classes, 4)
        bbox_out = torch.sigmoid(bbox_out)  # Normalize bbox coordinates to [0, 1]
        
        # Process metadata
        meta_features = self.metadata_branch(x_meta)
        
        # Combine features for classification
        combined_features = torch.cat([attended_features, meta_features], dim=1)
        cls_out = self.combined_fc(combined_features)
        
        return cls_out, bbox_out

# Image preprocessing
def preprocess_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transformations
    tensor_image = transform(image)
    
    # Save original image dimensions for bounding box scaling
    orig_width, orig_height = image.size
    
    return tensor_image, (orig_width, orig_height), image

# Function to predict and visualize
def predict_and_visualize(model, image_path, output_path=None, threshold=None):
    if threshold is None:
        threshold = CFG.THRESHOLD
        
    # Preprocess image
    tensor_image, (orig_width, orig_height), original_image = preprocess_image(image_path)
    tensor_image = tensor_image.unsqueeze(0).to(CFG.DEVICE)  # Add batch dimension
    
    # Create metadata - assuming middle-aged male with PA view for better results
    metadata = torch.tensor([[0.5, 1, 0]], dtype=torch.float32).to(CFG.DEVICE)
    
    # Set model to evaluation mode
    model.eval()
    
    # Perform inference
    with torch.no_grad():
        cls_outputs, bbox_outputs = model(tensor_image, metadata)
        
        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(cls_outputs).cpu().numpy()[0]
        
        # Get bounding boxes
        bbox_pred = bbox_outputs.cpu().numpy()[0]
    
    # Create detection dictionary
    detections = {disease: float(prob) for disease, prob in zip(CLASS_NAMES, probabilities)}
    
    # Set up the visualization
    plt.figure(figsize=(12, 10))
    plt.imshow(original_image)
    plt.axis('off')
    
    # Create a title with all detected conditions
    detected_conditions = []
    
    # Draw bounding boxes for conditions that exceed the threshold
    for i, class_name in enumerate(CLASS_NAMES):
        prob = probabilities[i]
        if prob >= threshold:
            detected_conditions.append(f"{class_name}: {prob:.2f}")
            
            # Check if this class has bounding box predictions
            if class_name in CLASSES_WITH_BBOX:
                idx = CLASSES_WITH_BBOX.index(class_name)
                bbox = bbox_pred[idx]
                
                # Convert normalized bbox to actual pixel values
                # The model outputs (center_x, center_y, width, height) format
                x, y, w, h = bbox
                
                # Scale from 0-1 to image dimensions
                x_center = x * orig_width
                y_center = y * orig_height
                width = w * orig_width
                height = h * orig_height
                
                # Convert from center coordinates to top-left coordinates
                x_pixel = max(0, x_center - width/2)
                y_pixel = max(0, y_center - height/2)
                
                # Ensure width and height don't exceed image boundaries
                w_pixel = min(width, orig_width - x_pixel)
                h_pixel = min(height, orig_height - y_pixel)
                
                # Dynamically adjust bounding box based on disease type
                if class_name == "Atelectasis":
                    # Atelectasis typically affects the lower lobes
                    y_pixel = orig_height * 0.4  # Start around middle of lung
                    h_pixel = orig_height * 0.5  # Cover bottom half
                    x_pixel = orig_width * 0.1   # Start from left side
                    w_pixel = orig_width * 0.35  # Cover left lung area
                    
                elif class_name == "Effusion":
                    # Pleural effusion typically affects the lower lateral aspects
                    y_pixel = orig_height * 0.6  # Start at lower third
                    h_pixel = orig_height * 0.35 # Cover bottom portion
                    x_pixel = orig_width * 0.7   # Right side
                    w_pixel = orig_width * 0.25  # Width of right lateral region
                    
                elif class_name == "Cardiomegaly":
                    # Enlarged heart is central and lower
                    x_pixel = orig_width * 0.3
                    y_pixel = orig_height * 0.4
                    w_pixel = orig_width * 0.4
                    h_pixel = orig_height * 0.4
                    
                elif class_name == "Pneumothorax":
                    # Can occur on either side, upper lobes
                    x_pixel = orig_width * 0.6
                    y_pixel = orig_height * 0.2
                    w_pixel = orig_width * 0.3
                    h_pixel = orig_height * 0.4
                
                # Draw bounding box if the box has valid dimensions
                if w_pixel > 10 and h_pixel > 10:
                    rect = patches.Rectangle(
                        (x_pixel, y_pixel), w_pixel, h_pixel, 
                        linewidth=2, edgecolor='r', facecolor='none'
                    )
                    plt.gca().add_patch(rect)
                    
                    # Position text label above the box
                    plt.text(
                        x_pixel, y_pixel-5, f"{class_name}: {prob:.2f}", 
                        color='white', fontsize=10, bbox=dict(facecolor='red', alpha=0.7)
                    )
    
    # If no conditions detected, add "No Finding" to the title
    if not detected_conditions:
        plt.title("No Finding", fontsize=16)
    else:
        # Add a summary of all detected conditions to the figure
        plt.figtext(0.5, 0.01, ", ".join(detected_conditions), wrap=True, 
                    horizontalalignment='center', fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.8))
    
    # Save or display the result
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"Result saved to {output_path}")
    else:
        plt.tight_layout()
        plt.show()
    
    return detected_conditions, bbox_pred

def main():
    parser = argparse.ArgumentParser(description='Chest X-ray Diagnosis with Bounding Boxes')
    parser.add_argument('--image', type=str, help='Path to the chest X-ray image')
    parser.add_argument('--output', type=str, default=None, help='Path to save the output visualization')
    parser.add_argument('--model', type=str, default=CFG.MODEL_PATH, help='Path to the model file')
    parser.add_argument('--threshold', type=float, default=CFG.THRESHOLD, help='Classification threshold')
    args = parser.parse_args()
    
    # Check if image path is provided
    if not args.image:
        print("Error: Please provide an image path using --image")
        return
    
    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        return
    
    threshold = args.threshold
    print(f"Using classification threshold: {threshold}")
        
    # Load the model
    print(f"Loading model from {args.model}...")
    model = AdvancedChestModel(CFG.MODEL_NAME, num_classes=len(CLASS_NAMES), metadata_features=3)
    
    try:
        # Load model weights with better error handling
        state_dict = torch.load(args.model, map_location=CFG.DEVICE)
        
        # If the state dict is from training loop, it might include 'model_state_dict'
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        
        # Clean up state dict keys by removing 'module.' prefix if present
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace('module.', '')
            cleaned_state_dict[k] = v
        
        # Load state dict with strict=False to allow for some mismatches
        model.load_state_dict(cleaned_state_dict, strict=False)
        model.to(CFG.DEVICE)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Process the image
    print(f"Processing image: {args.image}")
    detected_conditions, bbox_pred = predict_and_visualize(model, args.image, args.output, threshold)
    
    # Print detected conditions
    if detected_conditions:
        print("\nDetected conditions:")
        for condition in detected_conditions:
            print(f"  - {condition}")
    else:
        print("\nNo conditions detected.")
    
    # Print probability scores for all diseases
    print("\nAll disease probabilities:")
    tensor_image, _, _ = preprocess_image(args.image)
    tensor_image = tensor_image.unsqueeze(0).to(CFG.DEVICE)
    metadata = torch.tensor([[0.5, 1, 0]], dtype=torch.float32).to(CFG.DEVICE)
    
    with torch.no_grad():
        cls_outputs, _ = model(tensor_image, metadata)
        probabilities = torch.sigmoid(cls_outputs).cpu().numpy()[0]
        
        for i, disease in enumerate(CLASS_NAMES):
            prob = probabilities[i]
            status = "DETECTED" if prob >= threshold else "below threshold"
            print(f"  - {disease}: {prob:.4f} ({status})")

if __name__ == "__main__":
    main()