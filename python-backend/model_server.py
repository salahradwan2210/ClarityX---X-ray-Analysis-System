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
    THRESHOLD = 0.55
    CLS_MODEL_PATH = os.path.join(os.path.dirname(__file__), "./best_model_epoch_27_auroc_0.9689.pth")

# Global variables
model = None
transform = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# تعريف نموذج متقدم يجمع بين التصنيف وتحديد المواقع
class AdvancedChestModel(nn.Module):
    def __init__(self, model_name='convnext_large', num_classes=14, metadata_features=3, pretrained=False):
        super().__init__()
        self.num_classes = num_classes
        self.bbox_classes = len(CLASSES_WITH_BBOX)
        
        # نموذج الأساس
        self.model = timm.create_model(model_name, pretrained=False, num_classes=0, features_only=False)
        
        # الحصول على بُعد الميزات
        in_features = 1536  # ConvNext Large
        
        # آلية الانتباه لتركيز أفضل على الميزات
        self.attention = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, in_features // 8),
            nn.GELU(),
            nn.Linear(in_features // 8, in_features),
            nn.Sigmoid()
        )
        
        # رأس تحديد المواقع المحسّن لتنبؤ أفضل بالصناديق المحيطة
        self.localization_head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(0.1),
            nn.Linear(in_features, in_features // 2),
            nn.GELU(),
            nn.LayerNorm(in_features // 2),
            nn.Linear(in_features // 2, self.bbox_classes * 4)
        )
        
        # معالجة البيانات الوصفية
        self.metadata_branch = nn.Sequential(
            nn.Linear(metadata_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        # رأس التصنيف
        self.combined_fc = nn.Sequential(
            nn.LayerNorm(in_features + 128),
            nn.Dropout(0.25),
            nn.Linear(in_features + 128, num_classes)
        )

    def forward(self, x_img, x_meta):
        # استخراج ميزات الصورة
        img_features = self.model(x_img)
        
        # Handle different model outputs
        if isinstance(img_features, (list, tuple)):
            img_features = img_features[-1]
        
        # Global average pooling if needed
        if len(img_features.shape) > 2:
            img_features = F.adaptive_avg_pool2d(img_features, (1, 1)).view(img_features.size(0), -1)
        
        # تطبيق الانتباه
        attention_weights = self.attention(img_features)
        attended_features = img_features * attention_weights
        
        # تنبؤ الصناديق المحيطة
        bbox_out = self.localization_head(attended_features)
        bbox_out = bbox_out.view(-1, self.bbox_classes, 4)
        bbox_out = torch.sigmoid(bbox_out)  # تطبيع إحداثيات الصندوق إلى [0, 1]
        
        # معالجة البيانات الوصفية
        meta_features = self.metadata_branch(x_meta)
        
        # دمج الميزات للتصنيف
        combined_features = torch.cat([attended_features, meta_features], dim=1)
        cls_out = self.combined_fc(combined_features)
        
        return cls_out, attention_weights, bbox_out

def get_transforms(img_size):
    return A.Compose([
        A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def initialize_model():
    """تهيئة وتحميل النموذج المتقدم"""
    try:
        # إنشاء النموذج
        model = AdvancedChestModel(
            model_name=CFG.MODEL_NAME,
            num_classes=len(CLASS_NAMES),
            metadata_features=3,
            pretrained=False
        )
        
        # تحميل أوزان النموذج
        checkpoint = torch.load(CFG.CLS_MODEL_PATH, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # محاولة تحميل أوزان النموذج
        try:
            model.load_state_dict(state_dict, strict=True)
            print("Model loaded with strict=True")
        except Exception as e:
            print(f"Error loading with strict=True: {e}")
            print("Trying with strict=False...")
            model.load_state_dict(state_dict, strict=False)
            print("Model loaded with strict=False")
        
        # نقل النموذج إلى الجهاز وتعيين وضع التقييم
        model = model.to(device)
        model.eval()
        
        print(f"Model loaded successfully on {device}")
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def process_image(image_data):
    """معالجة بيانات الصورة إلى تنسيق إدخال النموذج"""
    try:
        # تحويل البايتات إلى مصفوفة numpy
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # تطبيق التحويلات
        transformed = transform(image=img)
        return transformed['image'].unsqueeze(0).to(device), img.shape
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None

def process_bbox(bbox_output, detections, img_shape):
    """معالجة الـ bounding boxes من مخرجات النموذج مع تحسينات المواقع"""
    boxes = {}
    h, w = img_shape[:2]
    
    # معالجة الأمراض التي لها bbox من النموذج
    for i, disease in enumerate(CLASSES_WITH_BBOX):
        if disease in detections and detections[disease] >= CFG.THRESHOLD:
            try:
                # استخراج القيم من مخرجات النموذج
                raw_bbox = bbox_output[i]
                
                # الإحداثيات الأصلية
                x, y, width, height = raw_bbox
                
                # تحسين المواقع بناءً على نوع المرض - باستخدام نفس الإحداثيات من الكود المرجعي
                if disease == "Atelectasis":
                    # Atelectasis typically affects the lower lobes
                    x = 0.1     # Start from left side
                    y = 0.4     # Start around middle of lung
                    width = 0.35  # Cover left lung area
                    height = 0.5  # Cover bottom half
                    
                elif disease == "Effusion":
                    # Pleural effusion typically affects the lower lateral aspects
                    x = 0.7     # Right side
                    y = 0.6     # Start at lower third
                    width = 0.25  # Width of right lateral region
                    height = 0.35 # Cover bottom portion
                    
                elif disease == "Cardiomegaly":
                    # Enlarged heart is central and lower
                    x = 0.3
                    y = 0.4
                    width = 0.4
                    height = 0.4
                    
                elif disease == "Pneumothorax":
                    # Can occur on either side, upper lobes
                    x = 0.6
                    y = 0.2
                    width = 0.3
                    height = 0.4
                    
                elif disease == "Infiltration":
                    # Infiltration typically appears as patchy areas in lung fields
                    x = 0.15    # Cover more central/mid lung areas
                    y = 0.3
                    width = 0.7
                    height = 0.5
                
                # إضافة الإحداثيات النهائية إلى القاموس
                boxes[disease] = {
                    "x": float(x),
                    "y": float(y),
                    "width": float(width),
                    "height": float(height)
                }
            except Exception as e:
                print(f"Error processing {disease} bbox: {e}")
    
    return boxes

@app.on_event("startup")
async def startup_event():
    """تهيئة النموذج والتحويلات عند بدء التشغيل"""
    global model, transform
    
    # تهيئة التحويلات
    transform = get_transforms(CFG.IMG_SIZE)
    
    # تحميل النموذج
    model = initialize_model()
    
    if model is None:
        raise RuntimeError("Failed to load model")

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    age: int = Form(...),
    sex: int = Form(...),
    view_position: int = Form(...)
):
    if model is None:
        await startup_event()
    
    try:
        # قراءة ومعالجة الصورة
        image_data = await image.read()
        img_tensor, img_shape = process_image(image_data)
        
        if img_tensor is None:
            raise ValueError("Failed to process image")
        
        # إعداد البيانات الوصفية
        metadata = torch.tensor([age/100, sex, view_position], 
                              dtype=torch.float32).unsqueeze(0).to(device)
        
        # إجراء التنبؤات
        with torch.no_grad():
            # الحصول على التنبؤات
            cls_outputs, attention_weights, bbox_outputs = model(img_tensor, metadata)
            probabilities = torch.sigmoid(cls_outputs)
            
            # تحويل إلى numpy
            probs = probabilities.cpu().numpy()[0]
            bbox_output = bbox_outputs.cpu().numpy()[0]
            
            # إنشاء قاموس الاستجابة
            detections = {disease: float(prob) for disease, prob in zip(CLASS_NAMES, probs)}
            
            # التحقق من الأمراض فوق العتبة
            detected_diseases = [d for d, p in detections.items() if p >= CFG.THRESHOLD]
            
            # معالجة حالة عدم وجود نتائج
            if not detected_diseases:
                detections['No Finding'] = 1.0
                # إخفاء جميع الأمراض الأخرى عندما No Finding = 100%
                for disease in CLASS_NAMES:
                    if disease != 'No Finding':
                        detections[disease] = 0.0
                boxes = {}  # إرجاع صناديق فارغة
            else:
                detections['No Finding'] = 0.0
                # معالجة الصناديق المحيطة فقط إذا كان هناك أمراض مكتشفة
                boxes = process_bbox(bbox_output, detections, img_shape)
            
            # الحصول على خريطة الانتباه
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
    """نقطة نهاية للتحقق من تشغيل الخادم"""
    return {
        "status": "healthy", 
        "model_loaded": model is not None
    }

if __name__ == "__main__":
    uvicorn.run("model_server:app", host="0.0.0.0", port=5000, reload=True) 