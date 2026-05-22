# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd # Keep if CFG or class names are extensive
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms
import argparse
import warnings

# Filter out annoying UserWarnings (Use with caution)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Configuration (Minimal needed for Inference) ---
class CFG:
    MODEL_NAME = 'convnext_large' # Should match your trained model
    IMG_SIZE = 512              # Should match your trained model's input size
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # These are needed if your AdvancedChestModel __init__ uses them directly from CFG
    DROPOUT_HEAD = 0.6
    DROPOUT_META = 0.4


# --- Class Definitions ---
# These are defined directly in the script
CLASS_NAMES_FULL = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding']
TRAIN_CLASSES = [c for c in CLASS_NAMES_FULL if c != 'No Finding'] # Should be 14 classes
# For BBox display, if needed
CLASSES_WITH_BBOX = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']


# --- Model Definition (Must be IDENTICAL to the one used for training) ---
class AdvancedChestModel(nn.Module):
    def __init__(self, model_name, num_classes, metadata_features=3, pretrained=False):
        super().__init__()
        self.num_classes_config = num_classes # Full number of classes including "No Finding" if CLASS_NAMES_FULL is used
        self.train_classes = len(TRAIN_CLASSES) # Number of output neurons for disease prediction

        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, features_only=False)
        # self.model.set_grad_checkpointing(enable=False) # No checkpointing for inference

        try:
            if hasattr(self.model, 'num_features'): in_features = self.model.num_features
            elif hasattr(self.model.head, 'in_features'): in_features = self.model.head.in_features
            else: raise AttributeError("Cannot find standard feature attribute")
        except AttributeError:
            try:
                if hasattr(self.model, 'fc') and hasattr(self.model.fc, 'in_features'): in_features = self.model.fc.in_features
                elif hasattr(self.model, 'classifier') and hasattr(self.model.classifier, 'in_features'): in_features = self.model.classifier.in_features
                else:
                    print("Attempting fallback in_features detection with dummy input...")
                    with torch.no_grad(): dummy_input = torch.randn(1, 3, 32, 32); dummy_output = self.model(dummy_input); in_features = dummy_output.shape[-1]
            except Exception:
                in_features = 1536 if 'convnext_large' in model_name else (_ for _ in ()).throw(ValueError("Cannot determine in_features for model head"))
        # print(f"Model: {model_name}, Detected in_features: {in_features}") # Less verbose

        self.localization_head = nn.Sequential(nn.LayerNorm(in_features), nn.Linear(in_features, self.train_classes * 4))
        self.metadata_branch = nn.Sequential(nn.Linear(metadata_features, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(CFG.DROPOUT_META), nn.Linear(64, 128), nn.LayerNorm(128), nn.ReLU())
        self.combined_fc = nn.Sequential(nn.LayerNorm(in_features + 128), nn.Dropout(CFG.DROPOUT_HEAD), nn.Linear(in_features + 128, self.train_classes))

    def forward(self, x_img, x_meta):
        img_features = self.model(x_img)
        bbox_out = self.localization_head(img_features); bbox_out = bbox_out.view(bbox_out.size(0), self.train_classes, 4)

        if x_meta is not None:
            if x_meta.shape[1] != self.metadata_branch[0].in_features:
                # Basic padding/truncating for metadata mismatch
                if x_meta.shape[1] < self.metadata_branch[0].in_features:
                    padding_size = self.metadata_branch[0].in_features - x_meta.shape[1]
                    padding = torch.zeros(x_meta.shape[0], padding_size, device=x_meta.device)
                    x_meta = torch.cat([x_meta, padding], dim=1)
                elif x_meta.shape[1] > self.metadata_branch[0].in_features:
                    x_meta = x_meta[:, :self.metadata_branch[0].in_features]
            meta_features = self.metadata_branch(x_meta)
            combined_features = torch.cat([img_features, meta_features], dim=1)
            combined_cls_out = self.combined_fc(combined_features)
        else: # Handle case where no metadata is provided (model might need adjustment for this or always require it)
            # This assumes combined_fc uses only img_features if no meta. Simpler: always require meta.
            # For this script, we will always provide dummy metadata if not given.
            # So the above 'if x_meta is not None' might not be strictly needed if predict() always passes it
            print("Warning: Metadata not provided to forward pass. This might be an issue if model expects it.")
            # Fallback: if model structure allows for no metadata (e.g. another head)
            # combined_cls_out = self.some_other_head_for_no_meta(img_features)
            # For now, we assume combined_fc is always used.
            # To make it work, we need to ensure x_meta is always valid.
            # Let's ensure dummy metadata has correct features for the branch if metadata is None.
            # This logic is better handled in the predict function.
            pass # Metadata will be prepared with correct features in predict function

        combined_cls_out = self.combined_fc(combined_features) # This line was duplicated, removing one
        return combined_cls_out, bbox_out

# --- Transforms for Inference (Validation Transforms) ---
def get_inference_transforms(img_size):
    # Using Torchvision for simplicity if Albumentations causes issues or isn't main focus
    # Ensure these match validation transforms used during training
    try:
        import albumentations as A_ # Alias to avoid conflict if A already used
        from albumentations.pytorch import ToTensorV2 as ToTensorV2_A # Alias
        import cv2 as cv2_A # Alias
        # print("Using Albumentations for inference transforms.")
        return A_.Compose([
            A_.Resize(height=img_size, width=img_size, interpolation=cv2_A.INTER_LANCZOS4),
            A_.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2_A()
        ])
    except ImportError:
        # print("Albumentations not found, using Torchvision for inference transforms.")
        return transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

# --- Prediction Function ---
def predict_single_image(model, image_path, metadata_tensor, transform_fn, device):
    try:
        image_pil = Image.open(image_path).convert('RGB')
        if isinstance(transform_fn, A.Compose): # Check if it's Albumentations
            image_np = np.array(image_pil)
            augmented = transform_fn(image=image_np)
            image_tensor = augmented['image'].unsqueeze(0).to(device)
        else: # Assume Torchvision
            image_tensor = transform_fn(image_pil).unsqueeze(0).to(device)

        metadata_tensor = metadata_tensor.to(device)
        if metadata_tensor.ndim == 1: metadata_tensor = metadata_tensor.unsqueeze(0)

        model.eval()
        with torch.no_grad():
            cls_logits, bbox_coords = model(image_tensor, metadata_tensor)

        cls_probabilities = torch.sigmoid(cls_logits)
        cls_probs_np = cls_probabilities.cpu().numpy().squeeze()
        bbox_coords_np = bbox_coords.cpu().numpy().squeeze()
        return cls_probs_np, bbox_coords_np
    except FileNotFoundError: print(f"Error: Image file not found at {image_path}"); return None, None
    except Exception as e: print(f"An error occurred for {image_path}: {e}"); return None, None

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Chest X-ray findings using a trained ConvNeXt model.")
    # <<< MODIFIED >>> Image paths are now positional and can be multiple
    parser.add_argument("image_paths", nargs='+', help="Path(s) to the input X-ray image(s).")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to trained model weights (.pth).")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Probability threshold for display (0.0 to 1.0).")
    # Simplified metadata input for this script
    parser.add_argument("--age", type=float, default=60.0, help="Patient Age (default: 60).")
    parser.add_argument("--gender", type=str, default="M", choices=["M", "F"], help="Patient Gender (M/F, default: M).")
    parser.add_argument("--view", type=str, default="PA", choices=["PA", "AP", "LL", "RL"], help="View Position (default: PA).")


    args = parser.parse_args()

    print(f"--- Prediction Script ---")
    print(f"Using device: {CFG.DEVICE}")
    print(f"Loading model: {CFG.MODEL_NAME} with image size {CFG.IMG_SIZE}")

    # --- Prepare Model ---
    # Ensure metadata_features matches what model expects (usually 3: Age, Gender, View)
    METADATA_FEATURES_COUNT = 3
    model = AdvancedChestModel(CFG.MODEL_NAME, num_classes=len(CLASS_NAMES_FULL), metadata_features=METADATA_FEATURES_COUNT, pretrained=False)

    print(f"Loading trained weights from: {args.model_path}")
    try:
        # <<< MODIFIED >>> Added weights_only=True for security and to suppress warning
        state_dict = torch.load(args.model_path, map_location=CFG.DEVICE, weights_only=True)
        if next(iter(state_dict)).startswith('module.'): state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        load_result = model.load_state_dict(state_dict, strict=True)
        print("Model weights loaded successfully.")
    except FileNotFoundError: print(f"Error: Model weights file not found at {args.model_path}"); exit()
    except RuntimeError as e: # Catch state_dict loading errors specifically
        print(f"Error loading model weights (state_dict mismatch?): {e}")
        print("Ensure the model definition in this script matches the trained model architecture.")
        exit()
    except Exception as e: print(f"An unexpected error occurred loading model weights: {e}"); exit()

    model.to(CFG.DEVICE); model.eval()

    # --- Prepare Transforms ---
    inference_transform = get_inference_transforms(CFG.IMG_SIZE)

    # --- Prepare (Dummy/Example) Metadata ---
    # This needs proper LabelEncoding matching your training to be truly accurate
    # For now, a simplified mapping:
    age_normalized = args.age / 100.0
    gender_encoded = 0 if args.gender == "F" else 1 # Example: Female=0, Male=1 (MUST MATCH TRAINING ENCODING)
    # View position encoding also needs to match training
    # Example: PA=0, AP=1, etc. This is a placeholder.
    view_pos_map = {"PA": 0, "AP": 1, "LL": 2, "RL": 3} # Example map
    view_encoded = view_pos_map.get(args.view.upper(), 0) # Default to 0 if unknown
    # The order must be: Age, Gender, View Position (as per metadata_cols in training)
    metadata_input_tensor = torch.tensor([[age_normalized, float(gender_encoded), float(view_encoded)]], dtype=torch.float32)
    print(f"Using metadata: Age={args.age}({age_normalized:.2f}), Gender={args.gender}({gender_encoded}), View={args.view}({view_encoded})")


    # --- Process Each Image ---
    for img_path in args.image_paths:
        print(f"\n--- Analyzing Image: {img_path} ---")
        probabilities, bboxes = predict_single_image(model, img_path, metadata_input_tensor, inference_transform, CFG.DEVICE)

        if probabilities is not None:
            print("\nPredicted Probabilities (above threshold):")
            found_findings = False
            results_for_print = sorted([(TRAIN_CLASSES[i], prob) for i, prob in enumerate(probabilities)], key=lambda x: x[1], reverse=True)

            for class_name, prob_val in results_for_print:
                if prob_val >= args.threshold:
                    print(f"- {class_name}: {prob_val:.4f}")
                    found_findings = True
                    if class_name in CLASSES_WITH_BBOX:
                        try:
                            cls_idx_for_bbox = TRAIN_CLASSES.index(class_name)
                            bbox_coords = bboxes[cls_idx_for_bbox]
                            print(f"    Raw BBox [x, y, w, h] (normalized): [{bbox_coords[0]:.3f}, {bbox_coords[1]:.3f}, {bbox_coords[2]:.3f}, {bbox_coords[3]:.3f}]")
                        except IndexError: print(f"    Error accessing bbox index for {class_name}")
                        except Exception as e_bbox: print(f"    Error processing bbox for {class_name}: {e_bbox}")
            if not found_findings: print(f"No findings detected above threshold {args.threshold}.")
        else: print("Prediction failed for this image.")
    print("\n--- Prediction complete ---")