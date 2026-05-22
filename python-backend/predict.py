import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import seaborn as sns
import warnings
import sys

warnings.filterwarnings('ignore')

# تعريف الفئات
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
               'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 
               'Pleural_Thickening', 'Hernia']

TRAIN_CLASSES = [c for c in CLASS_NAMES if c != 'No Finding']
CLASSES_WITH_BBOX = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
                    'Nodule', 'Pneumonia', 'Pneumothorax']

# تكوين النموذج
class CFG:
    MODEL_NAME = "convnext_large"
    IMG_SIZE = 512
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    THRESHOLD = 0.45  # عتبة التنبؤ الافتراضية

# تعريف النموذج
class ChestXrayModel(nn.Module):
    def __init__(self, model_name='convnext_large', num_classes=14, metadata_features=3, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False, num_classes=0, features_only=False)
        
        # Get feature dimension
        in_features = 1536  # ConvNext Large
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, in_features // 8),
            nn.GELU(),
            nn.Linear(in_features // 8, in_features),
            nn.Sigmoid()
        )
    
        # Metadata processing - exact match to saved dimensions
        self.metadata_branch = nn.Sequential(
            nn.Linear(metadata_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        # Localization head
        self.localization_head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, len(CLASSES_WITH_BBOX) * 4)
        )
        
        # Final classification
        self.combined_fc = nn.Sequential(
            nn.LayerNorm(in_features + 128),
            nn.Dropout(0.25),
            nn.Linear(in_features + 128, num_classes)
        )

    def forward(self, x, metadata):
        # Get base features
        features = self.model(x)
        
        # Handle different model outputs
        if isinstance(features, (list, tuple)):
            features = features[-1]
        
        # Global average pooling if needed
        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
            
        # Apply channel attention
        attention_weights = self.channel_attention(features)
        features_attended = features * attention_weights
        
        # Process metadata
        meta_features = self.metadata_branch(metadata)
        
        # Combine features
        combined_features = torch.cat([features_attended, meta_features], dim=1)
        
        # Classification
        cls_out = self.combined_fc(combined_features)
        
        # Localization
        bbox_out = self.localization_head(features_attended)
        bbox_out = bbox_out.view(-1, len(CLASSES_WITH_BBOX), 4)
        
        return cls_out, bbox_out, attention_weights

# دالة تحويل الصورة
def get_transforms(img_size):
    return A.Compose([
        A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
        
# دالة معالجة الصورة
def process_image(image_path, transform):
    try:
        # Try PIL first
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        transformed = transform(image=image)
        return transformed['image'].unsqueeze(0)
    except Exception as e:
        print(f"Error loading image with PIL: {e}")
        try:
            # Try OpenCV as fallback
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transformed = transform(image=image)
            return transformed['image'].unsqueeze(0)
        except Exception as e:
            print(f"Error loading image with OpenCV: {e}")
            return None

# دالة الرسم
def plot_results(image_path, predictions, bboxes, attention_map, threshold, output_path):
    fig = plt.figure(figsize=(15, 10))
    
    # Display original image with bounding boxes
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title("Original Image with Detections")
    
    # Draw bounding boxes
    height, width = image.shape[:2]
    for i, cls_name in enumerate(CLASSES_WITH_BBOX):
        if predictions[TRAIN_CLASSES.index(cls_name)] >= threshold:
            bbox = bboxes[i]
            x1, y1 = int(bbox[0] * width), int(bbox[1] * height)
            x2, y2 = int((bbox[0] + bbox[2]) * width), int((bbox[1] + bbox[3]) * height)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            plt.text(x1, y1-10, f"{cls_name}: {predictions[TRAIN_CLASSES.index(cls_name)]:.2f}", 
                    color='red', fontsize=8)
    
    # Display attention map
    plt.subplot(2, 2, 2)
    attention_np = attention_map.squeeze().cpu().numpy()
    attention_np = cv2.resize(attention_np, (width, height), interpolation=cv2.INTER_LINEAR)
    plt.imshow(attention_np, cmap='jet', alpha=0.5)
    plt.title("Attention Map")
    
    # Display prediction results
    plt.subplot(2, 1, 2)
    classes = TRAIN_CLASSES + ['No Finding']
    y_pos = np.arange(len(classes))
    colors = ['red' if pred >= threshold else 'blue' for pred in predictions]
    plt.barh(y_pos, predictions, color=colors)
    plt.yticks(y_pos, classes)
    plt.xlabel('Probability')
    plt.title('Diagnosis Results')
    
    # Add values on the plot
    for i, v in enumerate(predictions):
        plt.text(v + 0.01, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def load_model(model_path, device='cuda'):
    """Load the trained model with proper security handling"""
    print("Loading model...")
    try:
        # First try loading with weights_only=True for security
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        print("Successfully loaded checkpoint with weights_only=True")
    except Exception as e:
        print(f"Warning: Could not load with weights_only=True: {e}")
        print("Attempting to load with weights_only=False...")
        try:
            # If weights_only fails, try loading with pickle
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            print("Successfully loaded checkpoint with weights_only=False")
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    # Create model instance
    model = ChestXrayModel(num_classes=len(CLASS_NAMES))
    
    # Extract state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DataParallel)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Load state dict with strict=False to allow for architecture differences
    try:
        model.load_state_dict(state_dict, strict=False)
        print("Model state dictionary loaded successfully")
    except Exception as e:
        print(f"Error loading state dictionary: {e}")
        return None

    model = model.to(device)
    model.eval()
    return model

def predict_image(image_path, model, device='cuda', threshold=0.5):
    """Predict on a single image"""
    # Use the same transforms as in get_transforms
    transform = A.Compose([
        A.Resize(height=CFG.IMG_SIZE, width=CFG.IMG_SIZE, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Load and preprocess image
    image_tensor = process_image(image_path, transform)
    if image_tensor is None:
        print(f"Failed to load image: {image_path}")
        return None, None, None
    
    try:
        image_tensor = image_tensor.to(device)
        # Create dummy metadata (zeros)
        metadata = torch.zeros((1, 3)).to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs, bbox_outputs, attention_weights = model(image_tensor, metadata)
            probabilities = torch.sigmoid(outputs)
        
        return probabilities.cpu().numpy()[0], bbox_outputs.cpu().numpy()[0], attention_weights.cpu().numpy()[0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None, None

def visualize_predictions(image_path, probabilities, class_names, threshold=0.5, output_dir='predictions'):
    """Visualize predictions with a bar plot"""
    if probabilities is None:
        print("No predictions available to visualize")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot probabilities
        sns.barplot(x=probabilities, y=class_names)
        plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
        
        # Customize plot
        plt.title('Disease Probabilities')
        plt.xlabel('Probability')
        plt.ylabel('Disease')
        plt.legend()
        
        # Save plot
        output_path = os.path.join(output_dir, os.path.basename(image_path).replace('.png', '_predictions.png'))
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        # Print predictions
        print("\nPredictions:")
        for name, prob in zip(class_names, probabilities):
            if prob >= threshold:
                print(f"{name}: {prob:.4f}")
    except Exception as e:
        print(f"Error visualizing predictions: {e}")

def main():
    parser = argparse.ArgumentParser(description='Predict chest X-ray conditions')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold')
    parser.add_argument('--output_dir', type=str, default='predictions', help='Output directory for visualizations')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    # Verify image path exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        return
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = load_model(args.model_path, device)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Make prediction
    probabilities, bbox_outputs, attention_weights = predict_image(
        args.image_path, model, device, args.threshold
    )
    
    # Visualize results
    visualize_predictions(args.image_path, probabilities, CLASS_NAMES, args.threshold, args.output_dir)

if __name__ == '__main__':
    main() 