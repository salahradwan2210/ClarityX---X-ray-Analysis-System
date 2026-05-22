import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter

# استيراد النموذج
from models.densenet169_model import DenseNet169Model, train_densenet169

# تعريف الفئات
CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
    'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# تحسينات CUDA
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.fastest = True

# تعيين إعدادات مخصص الذاكرة
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

class ChestXrayDataset(torch.utils.data.Dataset):
    """فئة لتحميل ومعالجة بيانات الأشعة السينية للصدر"""
    
    def __init__(self, image_dir, df, transform=None, train=True):
        """
        تهيئة مجموعة البيانات
        
        المعلمات:
            image_dir (str): مسار مجلد الصور
            df (DataFrame): إطار البيانات الذي يحتوي على معلومات الصور
            transform (callable, optional): تحويلات لتطبيقها على الصور
            train (bool): ما إذا كانت مجموعة بيانات تدريب أم لا
        """
        self.image_dir = image_dir
        self.df = df
        self.transform = transform
        self.train = train
        
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
        """إرجاع عدد العناصر في مجموعة البيانات"""
        return len(self.df)
    
    def __getitem__(self, idx):
        """الحصول على عنصر من مجموعة البيانات"""
        img_name = self.df.iloc[idx]['Image Index']
        img_path = self.find_image_path(img_name)
        if img_path is None:
            raise FileNotFoundError(f"Image not found: {img_name}")
            
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.from_numpy(self.labels[idx])
        return image, label
    
    def find_image_path(self, image_name):
        """البحث عن الصورة في المجلدات الفرعية"""
        for i in range(1, 13):  # من 1 إلى 12
            folder_name = f'images_{i:03d}'  # مثال: images_001
            image_path = os.path.join(self.image_dir, 'data', folder_name, 'images', image_name)
            if os.path.exists(image_path):
                return image_path
        return None

def load_image_list(list_file_path):
    """تحميل قائمة الصور من ملف"""
    with open(list_file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def analyze_dataset(df):
    """تحليل وطباعة توزيع الفئات في مجموعة البيانات"""
    print("\nClass distribution in dataset:")
    for cls in CLASS_NAMES:
        count = len(df[df['Finding Labels'].str.contains(cls, regex=False)])
        print(f"{cls}: {count} samples")
    print(f"\nTotal samples: {len(df)}\n")
    return df

def balance_dataset(df, no_finding_ratio=0.2):
    """موازنة مجموعة البيانات عن طريق أخذ عينة من حالات No Finding"""
    print("\nBalancing dataset...")
    
    # فصل حالات No Finding والحالات الأخرى
    no_finding_df = df[df['Finding Labels'] == 'No Finding']
    other_cases_df = df[df['Finding Labels'] != 'No Finding']
    
    # حساب متوسط عدد العينات لكل حالة (باستثناء No Finding)
    conditions = [label for label in CLASS_NAMES if label != 'No Finding']
    samples_per_condition = []
    for condition in conditions:
        count = len(df[df['Finding Labels'].str.contains(condition, regex=False)])
        if count > 0:  # حساب الحالات التي لها عينات فقط
            samples_per_condition.append(count)
    
    # حساب الحجم المستهدف لعينات No Finding (استخدام متوسط الحالات الأخرى)
    avg_samples = int(np.mean(samples_per_condition))
    target_no_finding = int(avg_samples * no_finding_ratio)
    
    # أخذ عينة من حالات No Finding
    if len(no_finding_df) > target_no_finding:
        no_finding_df = no_finding_df.sample(n=target_no_finding, random_state=42)
    
    # دمج مجموعات البيانات المتوازنة
    balanced_df = pd.concat([other_cases_df, no_finding_df])
    
    # خلط مجموعة البيانات النهائية
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # طباعة التوزيع
    print("\nFinal class distribution after balancing:")
    for cls in CLASS_NAMES:
        count = len(balanced_df[balanced_df['Finding Labels'].str.contains(cls, regex=False)])
        print(f"{cls}: {count} samples")
    
    print(f"\nTotal samples after balancing: {len(balanced_df)}")
    return balanced_df

def parse_args():
    """تحليل معلمات سطر الأوامر"""
    parser = argparse.ArgumentParser(description='Train a DenseNet169 model for chest X-ray classification')
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
    
    # تحميل قائمة الصور
    valid_images = set(load_image_list(args.train_val_list))
    print(f"Found {len(valid_images)} images in list file")
    
    # تحميل ومعالجة البيانات
    df = pd.read_csv(args.data_entry)
    df = df[df['Image Index'].isin(valid_images)]
    print(f"Total images after filtering: {len(df)}")
    
    # موازنة مجموعة البيانات
    df = balance_dataset(df, args.no_finding_ratio)
    
    # تقسيم البيانات
    train_df = df.sample(frac=0.8, random_state=42)
    valid_df = df.drop(train_df.index)
    
    print("\nAnalyzing training set distribution:")
    analyze_dataset(train_df)
    
    print("\nAnalyzing validation set distribution:")
    analyze_dataset(valid_df)
    
    # إنشاء التحويلات
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # إنشاء مجموعات البيانات
    train_dataset = ChestXrayDataset(args.data_dir, train_df, transform=train_transform)
    valid_dataset = ChestXrayDataset(args.data_dir, valid_df, transform=valid_transform)
    
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
    model, best_auroc = train_densenet169(
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