import os
import torch
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from typing import Dict, Optional, List, Any
import logging

# استيراد الملفات المخصصة
from config import Config
from models.convnext_model import ChestXrayModel, BBoxModel

# إعداد التسجيل
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# إنشاء تطبيق FastAPI
app = FastAPI(title="تحليل أشعة الصدر")

# إضافة CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# متغيرات عامة
cls_model = None
bbox_model = None
transform = None
device = Config.DEVICE

def get_transforms(img_size):
    """تحويلات الصورة للتنبؤ"""
    return A.Compose([
        A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def initialize_models():
    """تهيئة نماذج التنبؤ"""
    try:
        # إنشاء نموذج التصنيف
        cls_model = ChestXrayModel(
            model_name=Config.MODEL_NAME,
            num_classes=len(Config.CLASS_NAMES),
            metadata_features=3,
            pretrained=False
        )
        
        # إنشاء نموذج مربعات الحدود
        bbox_model = BBoxModel(
            model_name=Config.MODEL_NAME,
            num_classes=len(Config.CLASSES_WITH_BBOX),
            pretrained=False
        )
        
        # تحميل أوزان نموذج التصنيف
        cls_checkpoint = torch.load(Config.CLS_MODEL_PATH, map_location=device)
        if isinstance(cls_checkpoint, dict) and 'model_state_dict' in cls_checkpoint:
            cls_state_dict = cls_checkpoint['model_state_dict']
        else:
            cls_state_dict = cls_checkpoint
        cls_state_dict = {k.replace('module.', ''): v for k, v in cls_state_dict.items()}
        cls_model.load_state_dict(cls_state_dict, strict=False)
        
        # تحميل أوزان نموذج مربعات الحدود
        bbox_checkpoint = torch.load(Config.BBOX_MODEL_PATH, map_location=device)
        if isinstance(bbox_checkpoint, dict) and 'model_state_dict' in bbox_checkpoint:
            bbox_state_dict = bbox_checkpoint['model_state_dict']
        else:
            bbox_state_dict = bbox_checkpoint
        bbox_state_dict = {k.replace('module.', ''): v for k, v in bbox_state_dict.items()}
        bbox_model.load_state_dict(bbox_state_dict, strict=False)
        
        # نقل النماذج إلى الجهاز وضبطها على وضع التقييم
        cls_model = cls_model.to(device)
        bbox_model = bbox_model.to(device)
        cls_model.eval()
        bbox_model.eval()
        
        logger.info(f"تم تحميل النماذج بنجاح على الجهاز: {device}")
        return cls_model, bbox_model
        
    except Exception as e:
        logger.error(f"خطأ في تحميل النماذج: {e}")
        return None, None

def process_image(image_data):
    """معالجة بيانات الصورة لتناسب النموذج"""
    try:
        # تحويل البيانات الثنائية إلى مصفوفة numpy
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # تطبيق التحويلات
        transformed = transform(image=img)
        return transformed['image'].unsqueeze(0).to(device)
    except Exception as e:
        logger.error(f"خطأ في معالجة الصورة: {e}")
        return None

def process_bbox(bbox_output, detections):
    """معالجة مربعات الحدود من مخرجات النموذج"""
    boxes = {}
    
    for i, disease in enumerate(Config.CLASSES_WITH_BBOX):
        if disease in detections and detections[disease] >= Config.THRESHOLD:
            # الحصول على الإحداثيات
            x, y, w, h = bbox_output[i]
            
            # تطبيع الإحداثيات
            x = torch.sigmoid(torch.tensor(x)).item()
            y = torch.sigmoid(torch.tensor(y)).item()
            w = torch.sigmoid(torch.tensor(w)).item() * 0.8
            h = torch.sigmoid(torch.tensor(h)).item() * 0.8
            
            # التأكد من بقاء المربع داخل حدود الصورة
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
    """تهيئة النماذج والتحويلات عند بدء التشغيل"""
    global cls_model, bbox_model, transform
    
    # إنشاء المجلدات اللازمة
    Config.setup_directories()
    
    # تهيئة التحويلات
    transform = get_transforms(Config.IMG_SIZE)
    
    # تحميل النماذج
    cls_model, bbox_model = initialize_models()
    
    if cls_model is None or bbox_model is None:
        logger.error("فشل تحميل النماذج")
        raise RuntimeError("فشل تحميل النماذج")
    else:
        logger.info("تم تهيئة التطبيق بنجاح")

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    age: int = Form(...),
    sex: int = Form(...),
    view_position: int = Form(...),
    selected_disease: str = Form(None)  # معامل لاختيار المرض
):
    """
    التنبؤ بالأمراض من صورة أشعة الصدر
    
    المعلمات:
        image: صورة الأشعة
        age: عمر المريض
        sex: جنس المريض (0 للأنثى، 1 للذكر)
        view_position: وضعية التصوير (0 للأمامية، 1 للجانبية)
        selected_disease: اسم المرض المحدد لعرض خريطة الحرارة الخاصة به
    """
    if cls_model is None or bbox_model is None:
        await startup_event()
    
    try:
        # قراءة ومعالجة الصورة
        image_data = await image.read()
        img_tensor = process_image(image_data)
        
        if img_tensor is None:
            raise ValueError("فشل معالجة الصورة")
        
        # تجهيز البيانات الوصفية
        metadata = torch.tensor([age/100, sex, view_position], 
                              dtype=torch.float32).unsqueeze(0).to(device)
        
        # التنبؤ
        with torch.no_grad():
            # الحصول على تنبؤات التصنيف
            cls_outputs, attention_weights = cls_model(img_tensor, metadata)
            probabilities = torch.sigmoid(cls_outputs)
            
            # الحصول على تنبؤات مربعات الحدود
            bbox_outputs = bbox_model(img_tensor)
            
            # تحويل إلى numpy
            probs = probabilities.cpu().numpy()[0]
            bbox_output = bbox_outputs.cpu().numpy()[0]
            
            # إنشاء قاموس الاستجابة
            detections = {disease: float(prob) for disease, prob in zip(Config.CLASS_NAMES, probs)}
            
            # التحقق من الأمراض التي تتجاوز العتبة
            detected_diseases = [d for d, p in detections.items() if p >= Config.THRESHOLD]
            
            # معالجة حالة عدم وجود أمراض
            if not detected_diseases:
                detections['No Finding'] = 1.0
            else:
                detections['No Finding'] = 0.0
            
            # معالجة مربعات الحدود
            boxes = process_bbox(bbox_output, detections)
            
            # توليد خريطة حرارة خاصة بمرض محدد
            attention_map = attention_weights.cpu().numpy()[0]
            
            # إذا تم اختيار مرض معين، تعديل خريطة الحرارة
            if selected_disease and selected_disease in Config.CLASS_NAMES:
                disease_idx = Config.CLASS_NAMES.index(selected_disease)
                disease_prob = probabilities[0][disease_idx].item()
                
                # الحصول على الميزات من الطبقة الأخيرة
                features = attention_weights.clone()
                
                # ترجيح الميزات بناءً على احتمالية المرض
                weighted_features = features * disease_prob
                
                # تطبيق softmax للتطبيع
                import torch.nn.functional as F
                weighted_features = F.softmax(weighted_features.view(-1), dim=0)
                
                # إعادة تشكيل إلى الشكل الأصلي
                attention_map = weighted_features.view(attention_map.shape).cpu().numpy()
            
            return {
                "detections": detections,
                "boxes": boxes,
                "attention_map": attention_map.tolist(),
                "threshold": Config.THRESHOLD,
                "no_finding": len(detected_diseases) == 0,
                "selected_disease": selected_disease,
                "selected_disease_prob": float(detections.get(selected_disease, 0)) if selected_disease else None
            }
            
    except Exception as e:
        logger.error(f"خطأ أثناء التنبؤ: {e}")
        return {"error": str(e)}

@app.get("/diseases")
async def get_diseases():
    """استرجاع قائمة الأمراض المتاحة"""
    return {
        "diseases": Config.CLASS_NAMES,
        "diseases_with_bbox": Config.CLASSES_WITH_BBOX
    }

@app.get("/healthcheck")
def healthcheck():
    """نقطة نهاية للتحقق من صحة الخادم"""
    return {
        "status": "healthy", 
        "cls_model_loaded": cls_model is not None,
        "bbox_model_loaded": bbox_model is not None
    }

# استيراد الملفات الثابتة
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    # تشغيل الخادم
    uvicorn.run("app:app", host=Config.SERVER_HOST, port=Config.SERVER_PORT, reload=True) 