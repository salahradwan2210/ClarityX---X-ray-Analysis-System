import os
import torch

class Config:
    """تكوين عام للمشروع"""
    
    # أسماء الفئات
    CLASS_NAMES = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
        'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 
        'Pleural_Thickening', 'Hernia'
    ]
    
    # الفئات التي لها مربعات حدود
    CLASSES_WITH_BBOX = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
        'Nodule', 'Pneumonia', 'Pneumothorax'
    ]
    
    # إعدادات النموذج
    MODEL_NAME = "convnext_large"
    IMG_SIZE = 512
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    THRESHOLD = 0.50
    
    # مسارات ملفات النموذج
    MODEL_DIR = "models/checkpoints"
    CLS_MODEL_PATH = os.path.join(MODEL_DIR, "best_model_epoch_28_auroc_0.9688.pth")
    BBOX_MODEL_PATH = os.path.join(MODEL_DIR, "best_bbox_model.pth")
    
    # مسارات البيانات
    DATA_DIR = "data"
    TRAIN_CSV = os.path.join(DATA_DIR, "train_bbox.csv")
    VAL_CSV = os.path.join(DATA_DIR, "val_bbox.csv")
    
    # إعدادات التدريب
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 30
    DROPOUT_HEAD = 0.25
    DROPOUT_META = 0.2
    
    # إعدادات الخادم
    SERVER_HOST = "0.0.0.0"
    SERVER_PORT = 5000
    UPLOAD_FOLDER = "uploads"
    
    @staticmethod
    def setup_directories():
        """إنشاء المجلدات اللازمة إذا لم تكن موجودة"""
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True) 