import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import roc_auc_score

# استيراد النموذج وأدوات المساعدة
from models.densenet121_model import DenseNet121Model
from utils import (
    ChestXrayDataset,
    train_transform,
    valid_transform,
    analyze_class_distribution,
    balance_dataset,
    CLASS_NAMES
)

# تحسينات CUDA
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True

def train_densenet121(train_loader, valid_loader, num_classes=14, num_epochs=5, device='cuda', learning_rate=1e-4, load_checkpoint=None):
    """
    تدريب نموذج DenseNet121
    
    المعلمات:
        train_loader (DataLoader): محمل بيانات التدريب
        valid_loader (DataLoader): محمل بيانات التحقق
        num_classes (int): عدد الفئات
        num_epochs (int): عدد الدورات
        device (str): الجهاز المستخدم
        learning_rate (float): معدل التعلم
        load_checkpoint (str): مسار نقطة الاستئناف
        
    الإرجاع:
        model (nn.Module): النموذج المدرب
        best_auroc (float): أفضل قيمة AUROC
    """
    # إنشاء النموذج
    model = DenseNet121Model(num_classes=num_classes, pretrained=True)
    model = model.to(device)
    
    # تحديد دالة الخسارة والمحسن
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    
    # إنشاء مقياس التدرج للتدريب بدقة مختلطة
    scaler = GradScaler()
    
    # تهيئة المتغيرات
    best_auroc = 0.0
    train_losses = []
    valid_losses = []
    aurocs = []
    
    # تحميل نقطة الاستئناف إذا كانت موجودة
    start_epoch = 0
    if load_checkpoint:
        checkpoint = torch.load(load_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_auroc = checkpoint.get('best_auroc', 0.0)
        print(f"Loaded checkpoint from epoch {start_epoch} with AUROC {best_auroc:.4f}")
    
    # طباعة معلومات التدريب
    print(f"Training on {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # حلقة التدريب
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # وضع التدريب
        model.train()
        running_loss = 0.0
        
        # حلقة التدريب
        train_pbar = tqdm(train_loader, desc="Training")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # تصفير التدرجات
            optimizer.zero_grad()
            
            # التمرير الأمامي مع دقة مختلطة
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # التمرير الخلفي مع تدرج متدرج
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # تحديث الخسارة
            running_loss += loss.item() * inputs.size(0)
            train_pbar.set_postfix({"loss": loss.item()})
        
        # حساب متوسط خسارة التدريب
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # وضع التقييم
        model.eval()
        running_loss = 0.0
        all_labels = []
        all_outputs = []
        
        # حلقة التقييم
        valid_pbar = tqdm(valid_loader, desc="Validation")
        with torch.no_grad():
            for inputs, labels in valid_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # التمرير الأمامي
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # تحديث الخسارة
                running_loss += loss.item() * inputs.size(0)
                valid_pbar.set_postfix({"loss": loss.item()})
                
                # تجميع التنبؤات والتسميات
                all_labels.append(labels.cpu().numpy())
                all_outputs.append(torch.sigmoid(outputs).cpu().numpy())
        
        # حساب متوسط خسارة التحقق
        valid_loss = running_loss / len(valid_loader.dataset)
        valid_losses.append(valid_loss)
        
        # تجميع التنبؤات والتسميات
        all_labels = np.vstack(all_labels)
        all_outputs = np.vstack(all_outputs)
        
        # حساب AUROC لكل فئة
        aucs = []
        for i in range(num_classes):
            if len(np.unique(all_labels[:, i])) > 1:
                auc = roc_auc_score(all_labels[:, i], all_outputs[:, i])
                aucs.append(auc)
        
        # حساب متوسط AUROC
        mean_auc = np.mean(aucs)
        aurocs.append(mean_auc)
        
        # تحديث جدول معدل التعلم
        scheduler.step(mean_auc)
        
        # طباعة النتائج
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Mean AUROC: {mean_auc:.4f}")
        
        # حفظ أفضل نموذج
        if mean_auc > best_auroc:
            best_auroc = mean_auc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auroc': best_auroc
            }, f"best_densenet121_model_auroc_{best_auroc:.3f}.pth")
            print(f"Saved best model with AUROC {best_auroc:.4f}")
        
        # رسم منحنى AUROC لكل فئة
        if epoch == num_epochs - 1:
            plt.figure(figsize=(15, 10))
            for i in range(num_classes):
                if len(np.unique(all_labels[:, i])) > 1:
                    auc = roc_auc_score(all_labels[:, i], all_outputs[:, i])
                    plt.bar(i, auc)
                    plt.text(i, auc + 0.01, f"{auc:.3f}", ha='center')
            plt.xticks(range(num_classes), CLASS_NAMES, rotation=90)
            plt.ylim(0.5, 1.0)
            plt.title(f"Class-wise AUROC - Epoch {epoch+1}")
            plt.tight_layout()
            plt.savefig(f"densenet121_class_aurocs_epoch_{epoch+1}.png")
    
    # رسم منحنى الخسارة
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('densenet121_loss_curve.png')
    
    # رسم منحنى AUROC
    plt.figure(figsize=(10, 5))
    plt.plot(aurocs, label='Mean AUROC')
    plt.xlabel('Epoch')
    plt.ylabel('AUROC')
    plt.legend()
    plt.title('Mean AUROC')
    plt.savefig('densenet121_auroc_curve.png')
    
    return model, best_auroc

def load_image_list(list_file_path):
    """تحميل قائمة الصور من ملف"""
    with open(list_file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def parse_args():
    """تحليل معلمات سطر الأوامر"""
    parser = argparse.ArgumentParser(description='Train a DenseNet121 model for chest X-ray classification')
    parser.add_argument('--data_dir', type=str, default='.', help='Path to data directory')
    parser.add_argument('--data_entry', type=str, default='./data/Data_Entry_2017.csv', help='Path to Data_Entry_2017.csv')
    parser.add_argument('--train_val_list', type=str, default='./data/train_val_list.txt', help='Path to train_val_list.txt')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--no_finding_ratio', type=float, default=0.2, help='Ratio of No Finding samples to use (0-1)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to load')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading')
    parser.add_argument('--preprocessed_dir', type=str, default='./preprocessed4', help='Directory with preprocessed data')
    parser.add_argument('--use_preprocessed', action='store_true', help='Use preprocessed data instead of CSV')
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
    
    # إعداد محملات البيانات
    if args.use_preprocessed:
        # استخدام البيانات المعالجة مسبقًا
        print(f"Using preprocessed data from {args.preprocessed_dir}")
        
        train_dataset = ChestXrayDataset(
            data_path=args.preprocessed_dir,
            train_or_valid="train",
            transform=train_transform,
            no_finding_ratio=args.no_finding_ratio
        )
        
        valid_dataset = ChestXrayDataset(
            data_path=args.preprocessed_dir,
            train_or_valid="valid",
            transform=valid_transform,
            no_finding_ratio=args.no_finding_ratio
        )
    else:
        # استخدام البيانات من CSV
        print(f"Using data from CSV: {args.data_entry}")
        
        # تحميل قائمة الصور
        valid_images = set(load_image_list(args.train_val_list))
        print(f"Found {len(valid_images)} images in list file")
        
        # تحميل ومعالجة البيانات
        df = pd.read_csv(args.data_entry)
        df = df[df['Image Index'].isin(valid_images)]
        print(f"Total images after filtering: {len(df)}")
        
        # موازنة مجموعة البيانات
        df = balance_dataset(df, max_no_finding_ratio=args.no_finding_ratio)
        
        # تقسيم البيانات
        train_df = df.sample(frac=0.8, random_state=42)
        valid_df = df.drop(train_df.index)
        
        print("\nAnalyzing training set distribution:")
        analyze_class_distribution(train_df)
        
        print("\nAnalyzing validation set distribution:")
        analyze_class_distribution(valid_df)
        
        # تعريف فصل ChestXrayDataset المخصص لاستخدام البيانات من CSV
        class CSVChestXrayDataset(torch.utils.data.Dataset):
            def __init__(self, image_dir, df, transform=None):
                self.image_dir = image_dir
                self.df = df
                self.transform = transform
                
                # تحويل التسميات إلى ترميز one-hot
                self.labels = []
                for finding in self.df['Finding Labels'].values:
                    label = np.zeros(len(CLASS_NAMES), dtype=np.float32)
                    for cls in finding.split('|'):
                        if cls in CLASS_NAMES:
                            label[CLASS_NAMES.index(cls)] = 1
                    self.labels.append(label)
                self.labels = np.array(self.labels)
                
                print(f"Dataset size: {len(self.df)}")
            
            def __len__(self):
                return len(self.df)
            
            def __getitem__(self, idx):
                img_name = self.df.iloc[idx]['Image Index']
                img_path = self._find_image_path(img_name)
                if img_path is None:
                    raise FileNotFoundError(f"Image not found: {img_name}")
                    
                image = Image.open(img_path).convert('RGB')
                
                if self.transform:
                    image = self.transform(image)
                
                label = torch.from_numpy(self.labels[idx])
                return image, label
            
            def _find_image_path(self, image_name):
                for i in range(1, 13):  # من 1 إلى 12
                    folder_name = f'images_{i:03d}'  # مثال: images_001
                    image_path = os.path.join(self.image_dir, 'data', folder_name, 'images', image_name)
                    if os.path.exists(image_path):
                        return image_path
                return None
        
        # إنشاء مجموعات البيانات
        train_dataset = CSVChestXrayDataset(args.data_dir, train_df, transform=train_transform)
        valid_dataset = CSVChestXrayDataset(args.data_dir, valid_df, transform=valid_transform)
    
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
        batch_size=args.batch_size * 2,  # استخدام حجم دفعة أكبر للتحقق
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # تدريب النموذج
    model, best_auroc = train_densenet121(
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_classes=len(CLASS_NAMES),
        num_epochs=args.num_epochs,
        device=device,
        learning_rate=args.learning_rate,
        load_checkpoint=args.checkpoint
    )
    
    print(f"Training completed! Best AUROC: {best_auroc:.4f}")

if __name__ == "__main__":
    main() 