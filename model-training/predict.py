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
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Dict, Optional, List, Any

# FastAPI app initialization
app = FastAPI(title="Chest X-Ray Analysis API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    THRESHOLD = 0.50
    CLS_MODEL_PATH = "best_model_epoch_28_auroc_0.9688.pth"
    BBOX_MODEL_PATH = "best_bbox_model.pth"

# Global variables
cls_model = None
bbox_model = None
transform = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# تعريف نموذج التصنيف
class ChestXrayModel(nn.Module):
    def __init__(self, model_name='convnext_large', num_classes=14, metadata_features=3, pretrained=False):
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
    
        # Metadata processing
        self.metadata_branch = nn.Sequential(
            nn.Linear(metadata_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU()
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
        
        return cls_out, attention_weights

# تعريف نموذج الـ Bounding Box
class BBoxModel(nn.Module):
    def __init__(self, model_name='convnext_large', num_classes=8, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False, num_classes=0, features_only=False)
        
        # Get feature dimension
        in_features = 1536  # ConvNext Large
        
        # Localization head with improved architecture
        self.localization_head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes * 4)
        )

    def forward(self, x):
        # Get base features
        features = self.model(x)
        
        # Handle different model outputs
        if isinstance(features, (list, tuple)):
            features = features[-1]
        
        # Global average pooling if needed
        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
        
        # Get bounding boxes
        bbox_out = self.localization_head(features)
        bbox_out = bbox_out.view(-1, len(CLASSES_WITH_BBOX), 4)
        
        return bbox_out

def get_transforms(img_size):
    return A.Compose([
        A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def initialize_models():
    """Initialize and load both classification and bbox models"""
    try:
        # Create classification model
        cls_model = ChestXrayModel(
            model_name=CFG.MODEL_NAME,
            num_classes=len(CLASS_NAMES),
            metadata_features=3,
            pretrained=False
        )
        
        # Create bbox model
        bbox_model = BBoxModel(
            model_name=CFG.MODEL_NAME,
            num_classes=len(CLASSES_WITH_BBOX),
            pretrained=False
        )
        
        # Load classification model weights
        cls_checkpoint = torch.load(CFG.CLS_MODEL_PATH, map_location=device)
        if isinstance(cls_checkpoint, dict) and 'model_state_dict' in cls_checkpoint:
            cls_state_dict = cls_checkpoint['model_state_dict']
        else:
            cls_state_dict = cls_checkpoint
        cls_state_dict = {k.replace('module.', ''): v for k, v in cls_state_dict.items()}
        cls_model.load_state_dict(cls_state_dict, strict=False)
        
        # Load bbox model weights
        bbox_checkpoint = torch.load(CFG.BBOX_MODEL_PATH, map_location=device)
        if isinstance(bbox_checkpoint, dict) and 'model_state_dict' in bbox_checkpoint:
            bbox_state_dict = bbox_checkpoint['model_state_dict']
        else:
            bbox_state_dict = bbox_checkpoint
        bbox_state_dict = {k.replace('module.', ''): v for k, v in bbox_state_dict.items()}
        bbox_model.load_state_dict(bbox_state_dict, strict=False)
        
        # Move models to device and set to eval mode
        cls_model = cls_model.to(device)
        bbox_model = bbox_model.to(device)
        cls_model.eval()
        bbox_model.eval()
        
        print(f"Both models loaded successfully on {device}")
        return cls_model, bbox_model
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

def process_image(image_data):
    """Process image data into model input format"""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        transformed = transform(image=img)
        return transformed['image'].unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def process_bbox(bbox_output, detections):
    """Process bounding boxes from model output"""
    boxes = {}
    
    for i, disease in enumerate(CLASSES_WITH_BBOX):
        if disease in detections and detections[disease] >= CFG.THRESHOLD:
            # Get raw coordinates
            x, y, w, h = bbox_output[i]
            
            # Normalize coordinates
            x = torch.sigmoid(torch.tensor(x)).item()
            y = torch.sigmoid(torch.tensor(y)).item()
            w = torch.sigmoid(torch.tensor(w)).item() * 0.8
            h = torch.sigmoid(torch.tensor(h)).item() * 0.8
            
            # Ensure box stays within image bounds
            x = max(0, min(1 - w, x))
            y = max(0, min(1 - h, y))
            
            boxes[disease] = {
                "x": float(x),
                "y": float(y),
                "width": float(w),
                "height": float(h)
            }
    
    return boxes

@app.on_event("startup")
async def startup_event():
    """Initialize models and transforms on startup"""
    global cls_model, bbox_model, transform
    
    # Initialize transforms
    transform = get_transforms(CFG.IMG_SIZE)
    
    # Load models
    cls_model, bbox_model = initialize_models()
    
    if cls_model is None or bbox_model is None:
        raise RuntimeError("Failed to load models")

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    age: int = Form(...),
    sex: int = Form(...),
    view_position: int = Form(...)
):
    if cls_model is None or bbox_model is None:
        await startup_event()
    
    try:
        # Read and process image
        image_data = await image.read()
        img_tensor = process_image(image_data)
        
        if img_tensor is None:
            raise ValueError("Failed to process image")
        
        # Prepare metadata
        metadata = torch.tensor([age/100, sex, view_position], 
                              dtype=torch.float32).unsqueeze(0).to(device)
        
        # Make predictions
        with torch.no_grad():
            # Get classification predictions
            cls_outputs, attention_weights = cls_model(img_tensor, metadata)
            probabilities = torch.sigmoid(cls_outputs)
            
            # Get bbox predictions
            bbox_outputs = bbox_model(img_tensor)
            
            # Convert to numpy
            probs = probabilities.cpu().numpy()[0]
            bbox_output = bbox_outputs.cpu().numpy()[0]
            
            # Create response dictionary
            detections = {disease: float(prob) for disease, prob in zip(CLASS_NAMES, probs)}
            
            # Check for diseases above threshold
            detected_diseases = [d for d, p in detections.items() if p >= CFG.THRESHOLD]
            
            # Handle No Finding case
            if not detected_diseases:
                detections['No Finding'] = 1.0
            else:
                detections['No Finding'] = 0.0
            
            # Process bounding boxes
            boxes = process_bbox(bbox_output, detections)
            
            # Get attention map
            attention_map = attention_weights.cpu().numpy()[0]
            
            return {
                "detections": detections,
                "boxes": boxes,
                "attention_map": attention_map.tolist(),
                "threshold": CFG.THRESHOLD,
                "no_finding": len(detected_diseases) == 0
            }
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": str(e)}

@app.get("/healthcheck")
def healthcheck():
    """Endpoint to check if the server is running"""
    return {
        "status": "healthy", 
        "cls_model_loaded": cls_model is not None,
        "bbox_model_loaded": bbox_model is not None
    }

if __name__ == "__main__":
    uvicorn.run("predict:app", host="0.0.0.0", port=5000, reload=True) 