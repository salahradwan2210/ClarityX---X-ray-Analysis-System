import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import timm
from torchvision import transforms
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_INSTALLED = True
except ImportError:
    ALBUMENTATIONS_INSTALLED = False
import argparse
import matplotlib.cm as cm

# Set environment variables to prevent any internet access attempts
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TIMM_OFFLINE'] = '1'
os.environ['PYTHONHTTPSVERIFY'] = '0'  # Disable SSL verification

# Configuration
class CFG:
    SEED = 42
    MODEL_NAME = 'convnext_large'
    IMG_SIZE = 512
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CHECKPOINT_PATH = 'last_output/best_model_epoch_7_auroc_0.9690.pth'
    DROPOUT_HEAD = 0.2
    DROPOUT_META = 0.1
    THRESHOLD = 0.45
    SHOW_TOP_N = 3
    USE_ATTENTION_MAPS = True

# Disease Classes
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding']
TRAIN_CLASSES = [c for c in CLASS_NAMES if c != 'No Finding']

# Override timm's model creation to work offline
original_create_model = timm.create_model
def offline_create_model(model_name, pretrained=True, **kwargs):
    # Always disable pretrained for offline use
    print(f"Creating model {model_name} with pretrained=False (offline mode)")
    return original_create_model(model_name, pretrained=False, **kwargs)

# Replace timm's create_model with our offline version
timm.create_model = offline_create_model

# The model definition
class OfflineChestModel(nn.Module):
    def __init__(self, model_name, num_classes, metadata_features=3):
        super().__init__()
        self.num_classes = num_classes
        self.train_classes = num_classes - 1
        
        # Create the model without pretrained weights
        self.model = timm.create_model(model_name, pretrained=False, num_classes=0, features_only=False)
        
        # ConvNext Large has 1536 features
        in_features = 1536
        print(f"Using fixed feature size for {model_name}: {in_features}")
        
        # Attention module
        self.attention = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, in_features // 8),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(in_features // 8, in_features),
            nn.Sigmoid()
        )
        
        # Localization head
        self.localization_head = nn.Sequential(
            nn.LayerNorm(in_features), 
            nn.Dropout(0.1),
            nn.Linear(in_features, in_features // 2),
            nn.GELU(),
            nn.Linear(in_features // 2, self.train_classes * 4)
        )
        
        # Metadata branch
        self.metadata_branch = nn.Sequential(
            nn.Linear(metadata_features, 128),
            nn.LayerNorm(128), 
            nn.ReLU(), 
            nn.Dropout(CFG.DROPOUT_META), 
            nn.Linear(128, 256),
            nn.LayerNorm(256), 
            nn.ReLU()
        )
        
        # Final classification layer
        self.combined_fc = nn.Sequential(
            nn.LayerNorm(in_features + 256),
            nn.Dropout(CFG.DROPOUT_HEAD), 
            nn.Linear(in_features + 256, in_features // 2),
            nn.GELU(),
            nn.Dropout(CFG.DROPOUT_HEAD / 2),
            nn.Linear(in_features // 2, self.train_classes)
        )
    
    def forward(self, x_img, x_meta):
        img_features = self.model(x_img)
        attention_weights = self.attention(img_features)
        img_features = img_features * attention_weights
        
        bbox_out = self.localization_head(img_features)
        bbox_out = bbox_out.view(bbox_out.size(0), self.train_classes, 4)
        
        if x_meta.shape[1] != self.metadata_branch[0].in_features:
            if x_meta.shape[1] < self.metadata_branch[0].in_features:
                padding = torch.zeros(x_meta.shape[0], self.metadata_branch[0].in_features - x_meta.shape[1], device=x_meta.device)
                x_meta = torch.cat([x_meta, padding], dim=1)
            elif x_meta.shape[1] > self.metadata_branch[0].in_features:
                x_meta = x_meta[:, :self.metadata_branch[0].in_features]
                
        meta_features = self.metadata_branch(x_meta)
        combined_features = torch.cat([img_features, meta_features], dim=1)
        combined_cls_out = self.combined_fc(combined_features)
        
        return combined_cls_out, bbox_out, attention_weights

# Helper functions
def get_transforms(img_size):
    if ALBUMENTATIONS_INSTALLED:
        return A.Compose([
            A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_AREA),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        print("Warning: Albumentations not installed. Falling back to basic Torchvision transforms.")
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])

def load_model(checkpoint_path, device, metadata_features=3):
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = OfflineChestModel(CFG.MODEL_NAME, num_classes=len(CLASS_NAMES), metadata_features=metadata_features)
    
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            
            # Remove 'module.' prefix if present (from DataParallel)
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k[len("module."):] if k.startswith('module.') else k: v for k, v in state_dict.items()}
            
            # Load with strict=False to allow for architecture differences
            result = model.load_state_dict(state_dict, strict=False)
            
            print(f"Model loaded from '{checkpoint_path}'")
            if result.missing_keys:
                print(f"Missing keys: {result.missing_keys}")
            if result.unexpected_keys:
                print(f"Unexpected keys: {result.unexpected_keys}")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"WARNING: Checkpoint '{checkpoint_path}' not found. Model will use random weights.")
    
    model.to(device)
    model.eval()
    return model

def predict_image(image_path, model, device, threshold=CFG.THRESHOLD, debug=False):
    # Read and process image
    try:
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
        transform = get_transforms(CFG.IMG_SIZE)
        
        # Apply transforms
        if ALBUMENTATIONS_INSTALLED:
            transformed = transform(image=img_np)
            img_tensor = transformed['image']
        else:
            img_tensor = transform(img_np)
        
        # Default metadata (age, gender, view position)
        metadata = torch.tensor([0.6, 0.5, 0.5], dtype=torch.float32)
        
        # Predict
        with torch.no_grad():
            img_tensor = img_tensor.unsqueeze(0).to(device)
            metadata = metadata.unsqueeze(0).to(device)
            outputs, bbox_outputs, attention_map = model(img_tensor, metadata)
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
        
        # Organize results
        disease_results = []
        for i, cls_name in enumerate(TRAIN_CLASSES):
            disease_results.append({'disease': cls_name, 'probability': float(probabilities[i])})
        
        # Sort by probability
        disease_results = sorted(disease_results, key=lambda x: x['probability'], reverse=True)
        
        # Determine if normal
        max_prob = max(probabilities)
        is_normal = max_prob < threshold
        
        # Return results
        results = {
            'predictions': disease_results,
            'is_normal': is_normal,
            'max_probability': float(max_prob),
            'attention_map': attention_map.cpu().numpy()[0] if CFG.USE_ATTENTION_MAPS else None
        }
        
        if debug:
            print(f"Max Probability: {max_prob:.4f}")
            print(f"Is Normal: {is_normal}")
            for item in disease_results[:5]:
                print(f"{item['disease']}: {item['probability']:.4f}")
                
        return results, img_np
    
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def generate_visualization(image, attention_map, predictions, threshold=CFG.THRESHOLD, top_n=CFG.SHOW_TOP_N):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Add attention map
    if attention_map is not None:
        # Resize attention map to match image
        try:
            heatmap = cv2.resize(attention_map, (image.shape[1], image.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Convert color format
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Overlay image with attention map
            superimposed_img = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
            axes[1].imshow(superimposed_img)
            axes[1].set_title('Attention Map')
        except Exception as e:
            print(f"Error generating attention map: {e}")
            axes[1].imshow(image)
            axes[1].set_title('Attention Map (Failed)')
    else:
        # Fallback to original image if attention map is not available
        axes[1].imshow(image)
        axes[1].set_title('Processing')
    axes[1].axis('off')
    
    # Add diagnosis classifications
    is_normal = predictions['is_normal']
    disease_preds = predictions['predictions']
    
    plt.figtext(0.5, 0.01, 'DIAGNOSIS RESULTS', fontsize=16, ha='center', weight='bold')
    
    if is_normal:
        plt.figtext(0.5, 0.05, 'NO FINDINGS DETECTED', fontsize=14, ha='center', color='green')
    else:
        plt.figtext(0.5, 0.05, 'POTENTIAL FINDINGS DETECTED:', fontsize=14, ha='center', color='red')
        # Show top N results
        for i, pred in enumerate(disease_preds[:top_n]):
            if pred['probability'] > threshold:
                color = 'red'
            else:
                color = 'grey'
            
            plt.figtext(0.5, 0.1 + i * 0.03, 
                      f"{pred['disease']}: {pred['probability']:.2f}",
                      fontsize=12, ha='center', color=color)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    return fig

def main():
    parser = argparse.ArgumentParser(description='Offline Chest X-ray Disease Diagnosis')
    parser.add_argument('image_path', type=str, help='Path to the X-ray image')
    parser.add_argument('--threshold', type=float, default=CFG.THRESHOLD, help='Threshold for positive predictions')
    parser.add_argument('--checkpoint', type=str, default=CFG.CHECKPOINT_PATH, help='Path to model checkpoint')
    parser.add_argument('--save', type=str, default=None, help='Save visualization to specified file path')
    parser.add_argument('--debug', action='store_true', help='Print debug information')
    
    args = parser.parse_args()
    print("OFFLINE MODE: Running without internet connection")
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        return
    
    # Load model
    model = load_model(args.checkpoint, CFG.DEVICE)
    
    # Predict
    results, img = predict_image(args.image_path, model, CFG.DEVICE, args.threshold, args.debug)
    if results is None:
        print("Failed to process the image.")
        return
    
    # Generate visualization
    fig = generate_visualization(img, results['attention_map'], results)
    
    # Save or display result
    if args.save:
        plt.savefig(args.save, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {args.save}")
    else:
        plt.show()
    
    # Print results
    print("\n=== DIAGNOSIS RESULTS ===")
    if results['is_normal']:
        print("NO FINDINGS DETECTED")
    else:
        print("POTENTIAL FINDINGS DETECTED:")
        for i, pred in enumerate(results['predictions'][:CFG.SHOW_TOP_N]):
            if pred['probability'] > args.threshold:
                print(f"{pred['disease']}: {pred['probability']:.4f} (POSITIVE)")
            else:
                print(f"{pred['disease']}: {pred['probability']:.4f}")

if __name__ == "__main__":
    main() 