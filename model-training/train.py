import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

# تحسينات CUDA
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.fastest = True

# استيراد النموذج الموحد
from models.chest_xray_models import create_model, prepare_demographic_data, DemographicDataset

# استيراد مكونات معالجة البيانات
from utils.dataset import ChestXrayDataset
from utils.transforms import train_transform, valid_transform

# تعريف الفئات
CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
    'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

def verify_data(loader):
    """
    التحقق من صحة البيانات
    
    المعلمات:
        loader (DataLoader): محمل البيانات
        
    الإرجاع:
        bool: صحة البيانات
    """
    print("Verifying data...")
    for batch in tqdm(loader, desc="Verifying"):
        # التحقق من وجود قيم NaN
        if isinstance(batch, tuple) and len(batch) >= 2:
            images, labels = batch[0], batch[1]
        else:
            images, labels = batch
            
        if torch.isnan(images).any():
            print("NaN values found in images!")
            return False
            
        if torch.isnan(labels).any():
            print("NaN values found in labels!")
            return False
            
        # التحقق من نطاق التسميات
        if (labels < 0).any() or (labels > 1).any():
            print("Labels out of range [0, 1]!")
            return False
    
    print("Data verification passed!")
    return True

def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler=None, num_epochs=5, device='cuda', use_demographics=False):
    """
    تدريب النموذج
    
    المعلمات:
        model (nn.Module): النموذج
        train_loader (DataLoader): محمل بيانات التدريب
        valid_loader (DataLoader): محمل بيانات التحقق
        criterion (nn.Module): دالة الخسارة
        optimizer (optim.Optimizer): المحسن
        scheduler (optim.lr_scheduler._LRScheduler): جدولة معدل التعلم
        num_epochs (int): عدد الدورات
        device (str): الجهاز
        use_demographics (bool): استخدام البيانات الديموغرافية
        
    الإرجاع:
        tuple: النموذج المدرب، قائمة خسائر التدريب، قائمة خسائر التحقق، قائمة AUROC
    """
    model.to(device)
    
    # قوائم لتتبع الخسائر والأداء
    train_losses = []
    valid_losses = []
    aurocs = []
    
    # أفضل AUROC
    best_auroc = 0.0
    
    # تمكين التدريب بدقة مختلطة
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # ===== مرحلة التدريب =====
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc="Training"):
            # تحميل البيانات
            if use_demographics and len(batch) == 3:
                images, labels, demographics = batch
                images = images.to(device)
                labels = labels.to(device)
                demographics = {
                    'age': demographics['age'].to(device),
                    'gender': demographics['gender'].to(device),
                    'view': demographics['view'].to(device)
                }
            else:
                if isinstance(batch, tuple) and len(batch) >= 2:
                    images, labels = batch[0], batch[1]
                else:
                    images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                demographics = None
            
            # تصفير التدرجات
            optimizer.zero_grad()
            
            # التمرير الأمامي مع دقة مختلطة
            with torch.cuda.amp.autocast():
                if use_demographics and demographics is not None:
                    outputs = model(images, demographics)
                else:
                    outputs = model(images)
                loss = criterion(outputs, labels)
            
            # التمرير الخلفي مع دقة مختلطة
            scaler.scale(loss).backward()
            
            # تقليم التدرجات لمنع انفجار التدرج
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # تحديث الأوزان
            scaler.step(optimizer)
            scaler.update()
            
            # تحديث جدولة معدل التعلم
            if scheduler is not None:
                scheduler.step()
            
            train_loss += loss.item()
        
        # حساب متوسط خسارة التدريب
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # ===== مرحلة التحقق =====
        model.eval()
        valid_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validation"):
                # تحميل البيانات
                if use_demographics and len(batch) == 3:
                    images, labels, demographics = batch
                    images = images.to(device)
                    labels = labels.to(device)
                    demographics = {
                        'age': demographics['age'].to(device),
                        'gender': demographics['gender'].to(device),
                        'view': demographics['view'].to(device)
                    }
                else:
                    if isinstance(batch, tuple) and len(batch) >= 2:
                        images, labels = batch[0], batch[1]
                    else:
                        images, labels = batch
                    images = images.to(device)
                    labels = labels.to(device)
                    demographics = None
                
                # التمرير الأمامي
                if use_demographics and demographics is not None:
                    outputs = model(images, demographics)
                else:
                    outputs = model(images)
                loss = criterion(outputs, labels)
                
                valid_loss += loss.item()
                
                # جمع التنبؤات والتسميات
                all_preds.append(torch.sigmoid(outputs).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # حساب متوسط خسارة التحقق
        valid_loss /= len(valid_loader)
        valid_losses.append(valid_loss)
        
        # حساب AUROC لكل فئة
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        class_aurocs = []
        for i in range(len(CLASS_NAMES)):
            if np.sum(all_labels[:, i]) > 0:  # تجنب الفئات التي ليس لها أمثلة إيجابية
                try:
                    auroc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                    class_aurocs.append(auroc)
                    print(f"{CLASS_NAMES[i]}: AUROC = {auroc:.4f}")
                except:
                    print(f"Could not calculate AUROC for {CLASS_NAMES[i]}")
        
        # حساب متوسط AUROC
        mean_auroc = np.mean(class_aurocs)
        aurocs.append(mean_auroc)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Mean AUROC: {mean_auroc:.4f}")
        
        # حفظ أفضل نموذج
        if mean_auroc > best_auroc:
            best_auroc = mean_auroc
            model_type = "demographic" if use_demographics else "base"
            torch.save(model.state_dict(), f"best_model_{model_type}_auroc_{best_auroc:.3f}.pth")
            print(f"Saved new best model with AUROC: {best_auroc:.4f}")
    
    return model, train_losses, valid_losses, aurocs

def parse_args():
    """تحليل معلمات سطر الأوامر"""
    parser = argparse.ArgumentParser(description='Train a chest X-ray classification model')
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to data directory')
    parser.add_argument('--data_entry', type=str, default='./data/data/Data_Entry_2017.csv', help='Path to Data_Entry_2017.csv')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--model', type=str, default='densenet121', 
                        choices=['densenet121', 'densenet169', 'resnet152', 'convnext', 'demographic'], 
                        help='Model architecture')
    parser.add_argument('--backbone', type=str, default='resnet50', 
                        choices=['resnet50', 'densenet121', 'convnext'], 
                        help='Backbone for demographic model')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--no_finding_ratio', type=float, default=0.2, help='Ratio of No Finding samples to use (0-1)')
    return parser.parse_args()

def main():
    """الدالة الرئيسية"""
    # تحليل المعلمات
    args = parse_args()
    
    # تحديد الجهاز
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # طباعة معلومات GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2:.1f} MB")
    
    # تحديد مسار البيانات
    data_path = os.path.join(args.data_dir, 'preprocessed4')
    
    # تحديد ما إذا كان النموذج يستخدم البيانات الديموغرافية
    use_demographics = args.model == 'demographic'
    
    # إنشاء مجموعات البيانات
    if use_demographics:
        # تحميل بيانات التدريب
        train_X_files = [f for f in os.listdir(data_path) if f.startswith('train_X_batch_')]
        train_y_files = [f for f in os.listdir(data_path) if f.startswith('train_y_batch_')]
        
        train_X_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        train_y_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        # تحميل بيانات التحقق
        valid_X_files = [f for f in os.listdir(data_path) if f.startswith('valid_X_batch_')]
        valid_y_files = [f for f in os.listdir(data_path) if f.startswith('valid_y_batch_')]
        
        valid_X_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        valid_y_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        # إعداد مسارات الصور والتسميات
        train_image_paths = []
        train_labels = []
        
        for i, (X_file, y_file) in enumerate(zip(train_X_files, train_y_files)):
            X_path = os.path.join(data_path, X_file)
            y_path = os.path.join(data_path, y_file)
            
            # تحميل البيانات
            X_batch = np.load(X_path)
            y_batch = np.load(y_path)
            
            # إضافة البيانات إلى القوائم
            for j in range(len(X_batch)):
                image_path = f"train_image_{i}_{j}.pt"
                train_image_paths.append(image_path)
                train_labels.append(y_batch[j])
                
                # حفظ الصورة بتنسيق PyTorch
                torch.save(torch.from_numpy(X_batch[j]), os.path.join(data_path, image_path))
        
        valid_image_paths = []
        valid_labels = []
        
        for i, (X_file, y_file) in enumerate(zip(valid_X_files, valid_y_files)):
            X_path = os.path.join(data_path, X_file)
            y_path = os.path.join(data_path, y_file)
            
            # تحميل البيانات
            X_batch = np.load(X_path)
            y_batch = np.load(y_path)
            
            # إضافة البيانات إلى القوائم
            for j in range(len(X_batch)):
                image_path = f"valid_image_{i}_{j}.pt"
                valid_image_paths.append(image_path)
                valid_labels.append(y_batch[j])
                
                # حفظ الصورة بتنسيق PyTorch
                torch.save(torch.from_numpy(X_batch[j]), os.path.join(data_path, image_path))
        
        # إعداد البيانات الديموغرافية
        print("Preparing demographic data...")
        
        # استخراج البيانات الديموغرافية من ملف Data_Entry_2017.csv
        train_demographics = prepare_demographic_data(args.data_entry, train_image_paths)
        valid_demographics = prepare_demographic_data(args.data_entry, valid_image_paths)
        
        # إنشاء مجموعات البيانات
        train_dataset = DemographicDataset(
            image_paths=[os.path.join(data_path, path) for path in train_image_paths],
            labels=train_labels,
            demographics=train_demographics
        )
        
        valid_dataset = DemographicDataset(
            image_paths=[os.path.join(data_path, path) for path in valid_image_paths],
            labels=valid_labels,
            demographics=valid_demographics
        )
    else:
        # استخدام مجموعات البيانات العادية
        train_dataset = ChestXrayDataset(data_path=data_path, train_or_valid="train", transform=train_transform, no_finding_ratio=args.no_finding_ratio)
        valid_dataset = ChestXrayDataset(data_path=data_path, train_or_valid="valid", transform=valid_transform, no_finding_ratio=args.no_finding_ratio)
    
    # إنشاء محملات البيانات
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # التحقق من صحة البيانات
    if not verify_data(train_loader) or not verify_data(valid_loader):
        print("Data verification failed. Exiting...")
        return
    
    # إنشاء النموذج
    if use_demographics:
        model = create_model('demographic', num_classes=len(CLASS_NAMES), backbone=args.backbone)
    else:
        model = create_model(args.model, num_classes=len(CLASS_NAMES))
    
    # تعريف دالة الخسارة والمحسن
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    
    # جدولة معدل التعلم
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate * 10,
        steps_per_epoch=len(train_loader),
        epochs=args.num_epochs
    )
    
    # تدريب النموذج
    model, train_losses, valid_losses, aurocs = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        device=device,
        use_demographics=use_demographics
    )
    
    # رسم منحنيات التدريب
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(aurocs, label='AUROC')
    plt.xlabel('Epoch')
    plt.ylabel('AUROC')
    plt.legend()
    plt.title('AUROC')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()
    
    print("Training completed!")

if __name__ == "__main__":
    main()