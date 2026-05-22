import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# استيراد النموذج ومجموعة البيانات
from models.detection_model import create_detection_model, DetectionDataset
from utils.bbox_utils import load_bbox_data, get_image_bboxes
from utils.transforms import train_transform, valid_transform

# تعريف الفئات
CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
    'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

def parse_args():
    """تحليل معلمات سطر الأوامر"""
    parser = argparse.ArgumentParser(description='Train a detection model for chest X-ray images')
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to data directory')
    parser.add_argument('--bbox_file', type=str, default='./data/BBox_List_2017.csv', help='Path to bounding box CSV file')
    parser.add_argument('--output_dir', type=str, default='./detection_outputs', help='Directory to save outputs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'densenet121', 'convnext'], help='Backbone model')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    return parser.parse_args()

def prepare_data(data_dir, bbox_file):
    """
    إعداد البيانات للتدريب
    
    المعلمات:
        data_dir (str): مسار مجلد البيانات
        bbox_file (str): مسار ملف مربعات الإحاطة
        
    الإرجاع:
        tuple: بيانات التدريب والتحقق
    """
    print("Loading data...")
    
    # تحميل بيانات مربعات الإحاطة
    bbox_data = load_bbox_data(bbox_file)
    
    # تحميل بيانات الصور والتسميات
    data_entry_path = os.path.join(data_dir, 'Data_Entry_2017.csv')
    meta_data = pd.read_csv(data_entry_path)
    
    # تحويل التسميات إلى تنسيق متعدد التسميات
    def parse_labels(label_str):
        labels = label_str.split('|')
        result = np.zeros(len(CLASS_NAMES))
        for label in labels:
            if label in CLASS_NAMES:
                result[CLASS_NAMES.index(label)] = 1
        return result
    
    meta_data['Labels_Array'] = meta_data['Finding Labels'].apply(parse_labels)
    
    # تقسيم البيانات إلى تدريب وتحقق
    np.random.seed(42)
    indices = np.random.permutation(len(meta_data))
    train_size = int(0.8 * len(meta_data))
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:]
    
    train_data = meta_data.iloc[train_indices]
    valid_data = meta_data.iloc[valid_indices]
    
    print(f"Train data: {len(train_data)} samples")
    print(f"Validation data: {len(valid_data)} samples")
    
    # إعداد مسارات الصور والتسميات ومربعات الإحاطة
    def prepare_dataset_inputs(data):
        image_paths = []
        labels = []
        bboxes = []
        
        for _, row in data.iterrows():
            image_name = row['Image Index']
            
            # البحث عن مسار الصورة
            image_path = None
            for i in range(1, 13):  # من 1 إلى 12
                folder_name = f'images_{i:03d}'  # مثال: images_001
                img_path = os.path.join(data_dir, folder_name, 'images', image_name)
                if os.path.exists(img_path):
                    image_path = img_path
                    break
            
            if image_path is None:
                continue
            
            # إضافة مسار الصورة والتسميات
            image_paths.append(image_path)
            labels.append(row['Labels_Array'])
            
            # الحصول على مربعات الإحاطة للصورة
            image_bboxes = get_image_bboxes(image_name, bbox_data)
            
            # تحويل التسميات إلى معرفات الفئات
            for bbox in image_bboxes:
                label = bbox['label']
                if label in CLASS_NAMES:
                    bbox['class_id'] = CLASS_NAMES.index(label) + 1  # +1 لأن 0 محجوز للخلفية
            
            bboxes.append(image_bboxes)
        
        return image_paths, labels, bboxes
    
    train_image_paths, train_labels, train_bboxes = prepare_dataset_inputs(train_data)
    valid_image_paths, valid_labels, valid_bboxes = prepare_dataset_inputs(valid_data)
    
    print(f"Prepared {len(train_image_paths)} training samples")
    print(f"Prepared {len(valid_image_paths)} validation samples")
    
    # إنشاء مجموعات البيانات
    train_dataset = DetectionDataset(
        image_paths=train_image_paths,
        labels=train_labels,
        bboxes=train_bboxes,
        transform=train_transform
    )
    
    valid_dataset = DetectionDataset(
        image_paths=valid_image_paths,
        labels=valid_labels,
        bboxes=valid_bboxes,
        transform=valid_transform
    )
    
    return train_dataset, valid_dataset

def collate_fn(batch):
    """
    دالة تجميع للـ DataLoader
    """
    return tuple(zip(*batch))

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    تدريب النموذج لدورة واحدة
    
    المعلمات:
        model (nn.Module): النموذج
        dataloader (DataLoader): محمل البيانات
        optimizer (optim.Optimizer): المحسن
        criterion (nn.Module): دالة الخسارة
        device (torch.device): الجهاز
        
    الإرجاع:
        float: متوسط الخسارة
    """
    model.train()
    total_loss = 0.0
    classification_loss = 0.0
    detection_loss = 0.0
    
    for images, targets in tqdm(dataloader, desc="Training"):
        # تحويل الصور والأهداف إلى الجهاز
        images = [image.to(device) for image in images]
        
        # التحقق من نوع الأهداف
        if isinstance(targets[0], dict):  # مربعات الإحاطة متوفرة
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            
            # التمرير الأمامي
            outputs = model(images, targets)
            
            # حساب خسارة التصنيف
            cls_loss = criterion(outputs['classification_logits'], torch.stack([t['labels'].float() for t in targets]))
            
            # حساب خسارة الكشف
            det_loss = sum(loss for loss in outputs['detection_losses'].values())
            
            # الخسارة الإجمالية
            loss = cls_loss + det_loss
            
            classification_loss += cls_loss.item()
            detection_loss += det_loss.item()
            
        else:  # مربعات الإحاطة غير متوفرة
            targets = [target.to(device) for target in targets]
            
            # التمرير الأمامي
            outputs = model(images)
            
            # حساب خسارة التصنيف فقط
            loss = criterion(outputs['classification_logits'], torch.stack(targets))
            
            classification_loss += loss.item()
        
        # التحسين
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = classification_loss / len(dataloader)
    avg_det_loss = detection_loss / len(dataloader)
    
    return avg_loss, avg_cls_loss, avg_det_loss

def validate(model, dataloader, criterion, device):
    """
    تقييم النموذج
    
    المعلمات:
        model (nn.Module): النموذج
        dataloader (DataLoader): محمل البيانات
        criterion (nn.Module): دالة الخسارة
        device (torch.device): الجهاز
        
    الإرجاع:
        float: متوسط الخسارة
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validation"):
            # تحويل الصور إلى الجهاز
            images = [image.to(device) for image in images]
            
            # التمرير الأمامي
            outputs = model(images)
            
            # التحقق من نوع الأهداف
            if isinstance(targets[0], dict):  # مربعات الإحاطة متوفرة
                # استخراج التسميات من الأهداف
                target_labels = [t['labels'] for t in targets]
                target_tensor = torch.zeros(len(target_labels), len(CLASS_NAMES))
                
                for i, labels in enumerate(target_labels):
                    for label in labels:
                        if label.item() - 1 < len(CLASS_NAMES):  # -1 لأننا أضفنا 1 سابقاً
                            target_tensor[i, label.item() - 1] = 1
                
                target_tensor = target_tensor.to(device)
                
                # حساب خسارة التصنيف
                loss = criterion(outputs['classification_logits'], target_tensor)
                
                # جمع التنبؤات والأهداف
                preds = torch.sigmoid(outputs['classification_logits'])
                all_preds.append(preds.cpu())
                all_targets.append(target_tensor.cpu())
                
            else:  # مربعات الإحاطة غير متوفرة
                targets = [target.to(device) for target in targets]
                target_tensor = torch.stack(targets)
                
                # حساب خسارة التصنيف
                loss = criterion(outputs['classification_logits'], target_tensor)
                
                # جمع التنبؤات والأهداف
                preds = torch.sigmoid(outputs['classification_logits'])
                all_preds.append(preds.cpu())
                all_targets.append(target_tensor.cpu())
            
            total_loss += loss.item()
    
    # حساب متوسط الخسارة
    avg_loss = total_loss / len(dataloader)
    
    # حساب AUROC لكل فئة
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    aurocs = []
    for i in range(len(CLASS_NAMES)):
        if np.sum(all_targets[:, i]) > 0:  # تجنب الفئات التي ليس لها أمثلة إيجابية
            try:
                from sklearn.metrics import roc_auc_score
                auroc = roc_auc_score(all_targets[:, i], all_preds[:, i])
                aurocs.append(auroc)
                print(f"{CLASS_NAMES[i]}: AUROC = {auroc:.4f}")
            except:
                print(f"Could not calculate AUROC for {CLASS_NAMES[i]}")
    
    mean_auroc = np.mean(aurocs)
    print(f"Mean AUROC: {mean_auroc:.4f}")
    
    return avg_loss, mean_auroc, aurocs

def visualize_predictions(model, dataloader, device, output_dir, num_samples=5):
    """
    تصور تنبؤات النموذج
    
    المعلمات:
        model (nn.Module): النموذج
        dataloader (DataLoader): محمل البيانات
        device (torch.device): الجهاز
        output_dir (str): مجلد الإخراج
        num_samples (int): عدد العينات للتصور
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    # الألوان لكل فئة
    colors = [
        (255, 0, 0),      # أحمر
        (0, 255, 0),      # أخضر
        (0, 0, 255),      # أزرق
        (255, 255, 0),    # أصفر
        (255, 0, 255),    # وردي
        (0, 255, 255),    # سماوي
        (128, 0, 0),      # بني محمر
        (0, 128, 0),      # أخضر داكن
        (0, 0, 128),      # أزرق داكن
        (128, 128, 0),    # زيتوني
        (128, 0, 128),    # أرجواني
        (0, 128, 128),    # فيروزي
        (128, 128, 128),  # رمادي
        (255, 128, 0)     # برتقالي
    ]
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader):
            if i >= num_samples:
                break
            
            # تحويل الصور إلى الجهاز
            images_device = [image.to(device) for image in images]
            
            # التنبؤ
            predictions = model.predict(images_device)
            
            for j, image in enumerate(images):
                # تحويل الصورة إلى نطاق 0-255
                img_np = image.permute(1, 2, 0).cpu().numpy() * 255
                img_np = img_np.astype(np.uint8)
                
                # تحويل إلى BGR لـ OpenCV
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
                # رسم مربعات الإحاطة المتنبأ بها
                for box, label, score in zip(
                    predictions['detection'][j]['boxes'].cpu().numpy(),
                    predictions['detection'][j]['labels'].cpu().numpy(),
                    predictions['detection'][j]['scores'].cpu().numpy()
                ):
                    if score > 0.5:  # عتبة الثقة
                        x1, y1, x2, y2 = map(int, box)
                        class_id = label - 1  # -1 لأننا أضفنا 1 سابقاً
                        
                        if class_id < len(CLASS_NAMES):
                            class_name = CLASS_NAMES[class_id]
                            color = colors[class_id % len(colors)]
                            
                            # رسم المربع
                            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
                            
                            # إضافة التسمية
                            cv2.putText(img_bgr, f"{class_name}: {score:.2f}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # رسم التنبؤات العامة
                cls_preds = predictions['classification']['probabilities'][j]
                y_pos = 30
                for k, (cls_name, prob) in enumerate(zip(CLASS_NAMES, cls_preds)):
                    if prob > 0.5:  # عتبة الثقة
                        color = colors[k % len(colors)]
                        cv2.putText(img_bgr, f"{cls_name}: {prob:.2f}", (10, y_pos),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        y_pos += 25
                
                # حفظ الصورة
                output_path = os.path.join(output_dir, f"pred_{i}_{j}.png")
                cv2.imwrite(output_path, img_bgr)

def main():
    """الدالة الرئيسية"""
    # تحليل المعلمات
    args = parse_args()
    
    # إنشاء مجلد الإخراج
    os.makedirs(args.output_dir, exist_ok=True)
    
    # تحديد الجهاز
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # إعداد البيانات
    train_dataset, valid_dataset = prepare_data(args.data_dir, args.bbox_file)
    
    # إنشاء محملات البيانات
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # إنشاء النموذج
    model = create_detection_model(
        num_classes=len(CLASS_NAMES),
        backbone_name=args.backbone,
        pretrained=True
    )
    model.to(device)
    
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
    
    # التدريب
    best_auroc = 0.0
    train_losses = []
    valid_losses = []
    aurocs = []
    
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        
        # تدريب دورة واحدة
        train_loss, train_cls_loss, train_det_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )
        
        # تقييم النموذج
        valid_loss, mean_auroc, epoch_aurocs = validate(
            model=model,
            dataloader=valid_loader,
            criterion=criterion,
            device=device
        )
        
        # تحديث جدولة معدل التعلم
        scheduler.step()
        
        # حفظ الخسائر والـ AUROC
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        aurocs.append(mean_auroc)
        
        print(f"Train Loss: {train_loss:.4f}, Classification Loss: {train_cls_loss:.4f}, Detection Loss: {train_det_loss:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}, Mean AUROC: {mean_auroc:.4f}")
        
        # حفظ أفضل نموذج
        if mean_auroc > best_auroc:
            best_auroc = mean_auroc
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"best_model_{args.backbone}_auroc_{best_auroc:.3f}.pth"))
            print(f"Saved new best model with AUROC: {best_auroc:.4f}")
        
        # تصور التنبؤات
        if (epoch + 1) % 5 == 0 or epoch == args.num_epochs - 1:
            visualize_predictions(
                model=model,
                dataloader=valid_loader,
                device=device,
                output_dir=os.path.join(args.output_dir, f"visualizations_epoch_{epoch+1}"),
                num_samples=5
            )
    
    # رسم منحنيات التدريب
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(aurocs, label='Mean AUROC')
    plt.xlabel('Epoch')
    plt.ylabel('AUROC')
    plt.legend()
    plt.title('Mean AUROC')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'))
    
    print("Training completed!")
    print(f"Best AUROC: {best_auroc:.4f}")

if __name__ == "__main__":
    main() 