import os
import torch
import numpy as np
import argparse
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
import timm
import torch.nn as nn
import torch.nn.functional as F
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configuration
MODEL_PATH = 'fine_tuned_model_best_auroc_0.9688.pth'  # Update with your best model path
IMG_SIZE = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD = 0.45  # Threshold for classification
TRAIN_CLASSES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
CLASSES_WITH_BBOX = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']

# Model definition
class AdvancedChestModel(nn.Module):
    def __init__(self, model_name='convnext_large', num_classes=14, metadata_features=3, pretrained=False):
        super().__init__()
        self.num_classes = num_classes
        self.train_classes = len(TRAIN_CLASSES)  # Use actual number of training classes
        self.bbox_classes = len(CLASSES_WITH_BBOX)  # Number of classes with bounding boxes
        
        # Base model
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, features_only=False)
        
        # Get feature dimension
        in_features = 1536  # ConvNext Large fixed dimension
        
        # Enhanced attention mechanism for better feature focus
        self.attention = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, in_features // 8),  # Wider attention module
            nn.GELU(),
            nn.Dropout(0.05),  # Light dropout in attention
            nn.Linear(in_features // 8, in_features),
            nn.Sigmoid()
        )
        
        # Enhanced localization head for better bounding box prediction
        self.localization_head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(0.05),  # Light dropout
            nn.Linear(in_features, in_features // 2),
            nn.GELU(),
            nn.LayerNorm(in_features // 2),
            nn.Dropout(0.05),  # Additional dropout layer
            nn.Linear(in_features // 2, in_features // 4),  # Additional layer
            nn.GELU(),
            nn.LayerNorm(in_features // 4),
            nn.Linear(in_features // 4, self.bbox_classes * 4)  # 4 coordinates per bbox class
        )
        
        # Metadata processing branch
        self.metadata_branch = nn.Sequential(
            nn.Linear(metadata_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        # Classification head
        self.combined_fc = nn.Sequential(
            nn.LayerNorm(in_features + 128),
            nn.Dropout(0.1),
            nn.Linear(in_features + 128, in_features // 2),  # Additional layer
            nn.GELU(),
            nn.LayerNorm(in_features // 2),
            nn.Dropout(0.05),  # Reduced dropout
            nn.Linear(in_features // 2, self.train_classes)
        )

    def forward(self, x_img, x_meta):
        # Extract image features
        img_features = self.model(x_img)
        
        # Apply attention
        attention_weights = self.attention(img_features)
        attended_features = img_features * attention_weights
        
        # Bounding box prediction
        bbox_out = self.localization_head(attended_features)
        bbox_out = bbox_out.view(-1, self.bbox_classes, 4)
        bbox_out = torch.sigmoid(bbox_out)  # Normalize bbox coordinates to [0, 1]
        
        # Process metadata
        if x_meta.shape[1] != self.metadata_branch[0].in_features:
            if x_meta.shape[1] < self.metadata_branch[0].in_features:
                padding = torch.zeros(x_meta.shape[0], 
                                   self.metadata_branch[0].in_features - x_meta.shape[1], 
                                   device=x_meta.device)
                x_meta = torch.cat([x_meta, padding], dim=1)
            else:
                x_meta = x_meta[:, :self.metadata_branch[0].in_features]
        
        meta_features = self.metadata_branch(x_meta)
        
        # Combine features for classification
        combined_features = torch.cat([attended_features, meta_features], dim=1)
        cls_out = self.combined_fc(combined_features)
        
        return cls_out, bbox_out

# Preprocessing functions
def get_transform():
    return A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def preprocess_image(image_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply transforms
    transform = get_transform()
    augmented = transform(image=img)
    img_tensor = augmented['image']
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor, img

def load_model(model_path):
    # Initialize model
    model = AdvancedChestModel(model_name='convnext_large', metadata_features=3, pretrained=False)
    
    # Load weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Handle potential module prefix
        if next(iter(state_dict)).startswith('module.'):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        
        # Load state dict
        model.load_state_dict(state_dict, strict=False)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    model.to(DEVICE)
    model.eval()
    return model

def diagnose_image(model, image_path, output_path=None, show_image=True):
    # Preprocess image
    img_tensor, original_img = preprocess_image(image_path)
    img_tensor = img_tensor.to(DEVICE)
    
    # Create dummy metadata (age, gender, view position)
    # Using default values for inference
    metadata = torch.tensor([[0.5, 0.5, 0.5]]).to(DEVICE)  # Default values
    
    # Get predictions
    with torch.no_grad():
        cls_logits, bbox_preds = model(img_tensor, metadata)
        cls_probs = torch.sigmoid(cls_logits)
    
    # Convert to numpy
    cls_probs = cls_probs.cpu().numpy()[0]
    bbox_preds = bbox_preds.cpu().numpy()[0]
    
    # Get predictions above threshold
    positive_classes = []
    for i, (cls_name, prob) in enumerate(zip(TRAIN_CLASSES, cls_probs)):
        if prob >= THRESHOLD:
            positive_classes.append((cls_name, prob))
    
    # Sort by probability
    positive_classes.sort(key=lambda x: x[1], reverse=True)
    
    # Check for "No Finding"
    if not positive_classes:
        positive_classes.append(("No Finding", 1.0 - max(cls_probs)))
    
    # Create visualization
    height, width, _ = original_img.shape
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(original_img)
    
    # Draw bounding boxes for classes with localization
    for cls_name, prob in positive_classes:
        if cls_name in CLASSES_WITH_BBOX:
            idx = CLASSES_WITH_BBOX.index(cls_name)
            bbox = bbox_preds[idx]
            
            # Convert normalized coordinates to pixel coordinates
            x, y, w, h = bbox
            x1 = int(x * width)
            y1 = int(y * height)
            w = int(w * width)
            h = int(h * height)
            
            # Only draw if the box has reasonable dimensions
            if w > 10 and h > 10:
                rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1-10, f"{cls_name}: {prob:.2f}", color='white', 
                        bbox=dict(facecolor='red', alpha=0.7))
    
    # Add text for all positive findings
    findings_text = "\n".join([f"{cls}: {prob:.2f}" for cls, prob in positive_classes])
    ax.text(10, 30, findings_text, color='white', bbox=dict(facecolor='black', alpha=0.7), fontsize=12)
    
    # Set title
    ax.set_title(f"Chest X-ray Diagnosis", fontsize=16)
    ax.axis('off')
    
    # Save or show
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved diagnosis to {output_path}")
    
    if show_image:
        plt.show()
    else:
        plt.close()
    
    return positive_classes

def main():
    parser = argparse.ArgumentParser(description='Diagnose chest X-ray images')
    parser.add_argument('--image', type=str, required=True, help='Path to the X-ray image')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to the model weights')
    parser.add_argument('--output', type=str, default=None, help='Path to save the output image')
    parser.add_argument('--no-display', action='store_true', help='Do not display the image')
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model)
    
    # Diagnose image
    findings = diagnose_image(model, args.image, args.output, not args.no_display)
    
    # Print findings
    print("\nDiagnosis Results:")
    print("=" * 30)
    for cls_name, prob in findings:
        print(f"{cls_name}: {prob:.4f} ({prob*100:.1f}%)")
    
    # Print recommendations
    if any(cls in [c[0] for c in findings] for cls in ['Pneumonia', 'Pneumothorax', 'Effusion']):
        print("\nRECOMMENDATION: Urgent clinical attention recommended.")
    else:
        print("\nRECOMMENDATION: Routine clinical review recommended.")

if __name__ == "__main__":
    main() 