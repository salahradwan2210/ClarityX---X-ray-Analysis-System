import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob

# استيراد النموذج
from models.convnext_large_model import ConvNextLargeModel

# تعريف الفئات
CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
    'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

class SimpleChestXrayDataset(Dataset):
    """فئة بسيطة لتحميل ومعالجة بيانات الأشعة السينية للصدر"""
    
    def __init__(self, image_paths, labels, transform=None):
        """
        تهيئة مجموعة البيانات
        
        المعلمات:
            image_paths (list): قائمة مسارات الصور
            labels (list): قائمة التسميات
            transform (callable, optional): تحويلات لتطبيقها على الصور
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        """إرجاع عدد العناصر في مجموعة البيانات"""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """الحصول على عنصر من مجموعة البيانات"""
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # تحميل الصورة
        image = Image.open(img_path).convert('RGB')
        
        # تطبيق التحويلات
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)

def find_images_with_labels(data_dir, csv_path, limit=10):
    """
    البحث عن الصور المتاحة مع تسمياتها
    
    المعلمات:
        data_dir (str): مسار مجلد البيانات
        csv_path (str): مسار ملف CSV
        limit (int): الحد الأقصى لعدد الصور
        
    الإرجاع:
        tuple: (قائمة مسارات الصور، قائمة التسميات)
    """
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from data entry file")
    
    image_paths = []
    labels = []
    
    # البحث عن الصور المتاحة
    for idx, row in tqdm(df.iterrows(), total=min(len(df), limit*10), desc="Finding images"):
        if len(image_paths) >= limit:
            break
            
        image_name = row['Image Index']
        
        # البحث عن الصورة في المجلدات الفرعية
        for i in range(1, 13):  # من 1 إلى 12
            folder_name = f'images_{i:03d}'  # مثال: images_001
            image_path = os.path.join(data_dir, 'data', folder_name, 'images', image_name)
            if os.path.exists(image_path):
                # تحويل التسمية إلى ترميز one-hot
                label = np.zeros(len(CLASS_NAMES), dtype=np.float32)
                for cls in row['Finding Labels'].split('|'):
                    if cls in CLASS_NAMES:
                        label[CLASS_NAMES.index(cls)] = 1
                
                image_paths.append(image_path)
                labels.append(label)
                break
    
    print(f"Found {len(image_paths)} images")
    return image_paths, labels

def load_model(model_path, num_classes=14, device='cuda'):
    """
    تحميل النموذج المدرب
    
    المعلمات:
        model_path (str): مسار ملف النموذج
        num_classes (int): عدد الفئات
        device (str): الجهاز المستخدم
        
    الإرجاع:
        model (nn.Module): النموذج المحمل
    """
    # إنشاء النموذج
    model = ConvNextLargeModel(num_classes=num_classes, pretrained=False)
    
    # تحميل الأوزان
    print(f"Loading model from: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        print("\nCheckpoint keys:", checkpoint.keys())
        
        # طريقة 1: تحميل من model_state_dict
        if 'model_state_dict' in checkpoint:
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("Loaded model from 'model_state_dict'")
                model = model.to(device)
                model.eval()
                return model
            except Exception as e:
                print(f"Error loading from model_state_dict: {e}")
        
        # طريقة 2: تحميل من state_dict
        if 'state_dict' in checkpoint:
            try:
                model.load_state_dict(checkpoint['state_dict'])
                print("Loaded model from 'state_dict'")
                model = model.to(device)
                model.eval()
                return model
            except Exception as e:
                print(f"Error loading from state_dict: {e}")
        
        # طريقة 3: تحميل النموذج مباشرة
        try:
            model.load_state_dict(checkpoint)
            print("Loaded model directly")
            model = model.to(device)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading model directly: {e}")
        
        # طريقة 4: تحميل من مفتاح model
        if 'model' in checkpoint:
            try:
                model.load_state_dict(checkpoint['model'])
                print("Loaded model from 'model' key")
                model = model.to(device)
                model.eval()
                return model
            except Exception as e:
                print(f"Error loading from model key: {e}")
        
        # طريقة 5: تجربة تعديل state_dict
        if 'model_state_dict' in checkpoint:
            try:
                state_dict = checkpoint['model_state_dict']
                # إزالة 'module.' من أسماء المفاتيح إذا كان موجوداً (يحدث عند استخدام DataParallel)
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k
                    if name.startswith('module.'):
                        name = name[7:]  # إزالة 'module.'
                    new_state_dict[name] = v
                
                model.load_state_dict(new_state_dict)
                print("Loaded model after removing 'module.' prefix")
                model = model.to(device)
                model.eval()
                return model
            except Exception as e:
                print(f"Error loading after removing 'module.' prefix: {e}")
        
        # طريقة 6: تجربة تحميل النموذج مع تجاهل بعض الطبقات
        if 'model_state_dict' in checkpoint:
            try:
                state_dict = checkpoint['model_state_dict']
                model_dict = model.state_dict()
                
                # تصفية state_dict للحصول على المفاتيح المتطابقة فقط
                filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
                model_dict.update(filtered_dict)
                
                model.load_state_dict(model_dict)
                print(f"Loaded model with partial state_dict ({len(filtered_dict)}/{len(state_dict)} keys)")
                model = model.to(device)
                model.eval()
                return model
            except Exception as e:
                print(f"Error loading with partial state_dict: {e}")
        
        # إذا وصلنا إلى هنا، فلم نتمكن من تحميل النموذج بأي طريقة
        print("Failed to load model with any method. Initializing with random weights.")
        model = model.to(device)
        model.eval()
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Initializing model with random weights.")
        model = model.to(device)
        model.eval()
        return model

def evaluate_model(model, data_loader, device, threshold=0.5):
    """
    تقييم النموذج على مجموعة البيانات
    
    المعلمات:
        model (nn.Module): النموذج المدرب
        data_loader (DataLoader): محمل البيانات
        device (str): الجهاز المستخدم
        threshold (float): عتبة التنبؤ
        
    الإرجاع:
        dict: نتائج التقييم
    """
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []
    all_outputs = []  # إضافة مخرجات النموذج الخام
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs >= threshold).astype(int)
            
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs)
            all_preds.append(preds)
            all_outputs.append(outputs.cpu().numpy())  # حفظ المخرجات الخام
    
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    all_preds = np.vstack(all_preds)
    all_outputs = np.vstack(all_outputs)  # تجميع المخرجات الخام
    
    return {
        'labels': all_labels,
        'probs': all_probs,
        'preds': all_preds,
        'outputs': all_outputs  # إضافة المخرجات الخام إلى النتائج
    }

def calculate_metrics(y_true, y_pred, classes):
    """
    حساب مقاييس الأداء لكل فئة
    
    المعلمات:
        y_true (numpy.ndarray): القيم الحقيقية
        y_pred (numpy.ndarray): القيم المتنبأ بها
        classes (list): أسماء الفئات
        
    الإرجاع:
        pandas.DataFrame: إطار بيانات يحتوي على المقاييس
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'Class': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': [],
        'Support': []
    }
    
    for i, cls_name in enumerate(classes):
        metrics['Class'].append(cls_name)
        metrics['Accuracy'].append(accuracy_score(y_true[:, i], y_pred[:, i]))
        metrics['Precision'].append(precision_score(y_true[:, i], y_pred[:, i], zero_division=0))
        metrics['Recall'].append(recall_score(y_true[:, i], y_pred[:, i], zero_division=0))
        metrics['F1-Score'].append(f1_score(y_true[:, i], y_pred[:, i], zero_division=0))
        metrics['Support'].append(np.sum(y_true[:, i]))
    
    return pd.DataFrame(metrics)

def parse_args():
    """تحليل معلمات سطر الأوامر"""
    parser = argparse.ArgumentParser(description='Simple evaluation of a trained model')
    parser.add_argument('--data_dir', type=str, default='.', help='Path to data directory')
    parser.add_argument('--data_entry', type=str, default='./data/Data_Entry_2017.csv', help='Path to Data_Entry_2017.csv')
    parser.add_argument('--model_path', type=str, default='best_convnext_large_model_auroc_0.742.pth', help='Path to the trained model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--threshold', type=float, default=0.5, help='Prediction threshold')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to evaluate')
    return parser.parse_args()

def main():
    """الدالة الرئيسية"""
    # تحليل المعلمات
    args = parse_args()
    
    # تحديد الجهاز
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # البحث عن الصور المتاحة مع تسمياتها
    image_paths, labels = find_images_with_labels(args.data_dir, args.data_entry, limit=args.num_samples)
    
    if len(image_paths) == 0:
        print("No images found. Exiting...")
        return
    
    # إنشاء التحويلات
    test_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # إنشاء مجموعة البيانات
    test_dataset = SimpleChestXrayDataset(image_paths, labels, transform=test_transform)
    
    # إنشاء محمل البيانات
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # تحميل النموذج
    model = load_model(args.model_path, num_classes=14, device=device)
    
    # تقييم النموذج
    print("Evaluating model...")
    results = evaluate_model(model, test_loader, device, args.threshold)
    
    # حساب المقاييس
    print("Calculating metrics...")
    metrics_df = calculate_metrics(results['labels'], results['preds'], CLASS_NAMES)
    print(metrics_df)
    
    # طباعة التنبؤات لكل صورة
    print("\nPredictions for each image:")
    for i, (image_path, label) in enumerate(zip(image_paths, results['labels'])):
        pred = results['preds'][i]
        prob = results['probs'][i]
        raw_output = results['outputs'][i]  # الحصول على المخرجات الخام
        
        print(f"\nImage: {os.path.basename(image_path)}")
        print("True labels:", end=" ")
        for j, cls_name in enumerate(CLASS_NAMES):
            if label[j] > 0:
                print(f"{cls_name}", end=", ")
        
        print("\nRaw outputs:", end=" ")
        for j, cls_name in enumerate(CLASS_NAMES):
            print(f"{cls_name}: {raw_output[j]:.4f}", end=", ")
        
        print("\nProbabilities:", end=" ")
        for j, cls_name in enumerate(CLASS_NAMES):
            print(f"{cls_name}: {prob[j]:.4f}", end=", ")
        
        print("\nPredicted labels:", end=" ")
        for j, cls_name in enumerate(CLASS_NAMES):
            if pred[j] > 0:
                print(f"{cls_name} ({prob[j]:.4f})", end=", ")
        print()

if __name__ == "__main__":
    main() 