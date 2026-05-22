import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import itertools
import glob
import random

# استيراد النموذج
from models.convnext_large_model import ConvNextLargeModel

# تعريف الفئات
CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
    'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

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
        self.df = df.copy()
        self.transform = transform
        self.train = train
        
        # التحقق من وجود الصور وإزالة الصور غير الموجودة
        valid_indices = []
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Verifying images"):
            img_path = self.find_image_path(row['Image Index'])
            if img_path is not None:
                valid_indices.append(idx)
        
        self.df = self.df.loc[valid_indices].reset_index(drop=True)
        print(f"Found {len(self.df)} valid images out of {len(df)} total images")
        
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
            # هذا لن يحدث لأننا تحققنا من وجود جميع الصور في الإنشاء
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

def find_available_images(data_dir, df):
    """
    البحث عن الصور المتاحة في مجلدات الصور والتحقق من وجودها فعليًا
    
    المعلمات:
        data_dir (str): مسار مجلد البيانات
        df (DataFrame): إطار البيانات الذي يحتوي على معلومات الصور
        
    الإرجاع:
        list: قائمة بأسماء الصور المتاحة
    """
    print(f"Finding available images in {data_dir}")
    available_images = []
    
    # التحقق من وجود الصور المذكورة في ملف CSV
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking images"):
        image_name = row['Image Index']
        image_found = False
        
        for i in range(1, 13):  # من 1 إلى 12
            folder_name = f'images_{i:03d}'  # مثال: images_001
            image_path = os.path.join(data_dir, 'data', folder_name, 'images', image_name)
            if os.path.exists(image_path):
                available_images.append(image_name)
                image_found = True
                break
    
    print(f"Found {len(available_images)} available images in total")
    return available_images

def load_image_list(list_file_path):
    """
    تحميل قائمة الصور من ملف
    
    المعلمات:
        list_file_path (str): مسار ملف القائمة
        
    الإرجاع:
        list: قائمة أسماء الصور
    """
    print(f"Loading image list from {list_file_path}")
    image_list = []
    try:
        with open(list_file_path, 'r') as f:
            for line in f:
                image_list.append(line.strip())
        print(f"Found {len(image_list)} images in list file")
        return image_list
    except FileNotFoundError:
        print(f"Warning: Image list file {list_file_path} not found")
        return []

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
    checkpoint = torch.load(model_path, map_location=device)
    
    # طباعة معلومات عن النموذج المحفوظ
    print("\nCheckpoint keys:", checkpoint.keys())
    
    # التعامل مع أنواع مختلفة من الملفات المحفوظة
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model from 'model_state_dict'")
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded model from 'state_dict'")
    else:
        # محاولة تحميل النموذج مباشرة
        try:
            model.load_state_dict(checkpoint)
            print("Loaded model directly")
        except Exception as e:
            print(f"Error loading model directly: {e}")
            print("Trying to load from 'model' key...")
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
                print("Loaded model from 'model' key")
    
    model = model.to(device)
    model.eval()
    return model

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues, figsize=(12, 10)):
    """
    رسم مصفوفة الارتباك
    
    المعلمات:
        cm (numpy.ndarray): مصفوفة الارتباك
        classes (list): أسماء الفئات
        normalize (bool): ما إذا كان يجب تطبيع المصفوفة
        title (str): عنوان الرسم
        cmap (matplotlib.colors.Colormap): خريطة الألوان
        figsize (tuple): حجم الرسم
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
    
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right', fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=8)
    
    plt.tight_layout()
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    plt.savefig(f"{title.lower().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curves(y_true, y_pred, classes, figsize=(15, 10)):
    """
    رسم منحنيات ROC لكل فئة
    
    المعلمات:
        y_true (numpy.ndarray): القيم الحقيقية
        y_pred (numpy.ndarray): القيم المتنبأ بها
        classes (list): أسماء الفئات
        figsize (tuple): حجم الرسم
    """
    plt.figure(figsize=figsize)
    
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    
    for i, cls_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{cls_name} (AUC = {roc_auc:.3f})')
        
        # حساب متوسط منحنى ROC
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
    
    # إضافة منحنى ROC المتوسط
    mean_tpr /= len(classes)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--', lw=2, label=f'Mean ROC (AUC = {mean_auc:.3f})')
    
    # إضافة خط الأساس
    plt.plot([0, 1], [0, 1], 'r--', lw=2, label='Chance')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=16)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True)
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_precision_recall_curves(y_true, y_pred, classes, figsize=(15, 10)):
    """
    رسم منحنيات الدقة والاستدعاء لكل فئة
    
    المعلمات:
        y_true (numpy.ndarray): القيم الحقيقية
        y_pred (numpy.ndarray): القيم المتنبأ بها
        classes (list): أسماء الفئات
        figsize (tuple): حجم الرسم
    """
    plt.figure(figsize=figsize)
    
    for i, cls_name in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
        avg_precision = average_precision_score(y_true[:, i], y_pred[:, i])
        plt.plot(recall, precision, lw=2, label=f'{cls_name} (AP = {avg_precision:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves', fontsize=16)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True)
    plt.savefig('precision_recall_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_class_metrics(metrics_df, figsize=(15, 8)):
    """
    رسم مقاييس الأداء لكل فئة
    
    المعلمات:
        metrics_df (pandas.DataFrame): إطار بيانات يحتوي على المقاييس
        figsize (tuple): حجم الرسم
    """
    plt.figure(figsize=figsize)
    
    metrics_df = metrics_df.sort_values('F1-Score', ascending=False)
    
    # رسم شريطي للمقاييس
    metrics_df.plot(kind='bar', figsize=figsize)
    plt.title('Performance Metrics by Class', fontsize=16)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('class_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

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
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs >= threshold).astype(int)
            
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs)
            all_preds.append(preds)
    
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    all_preds = np.vstack(all_preds)
    
    return {
        'labels': all_labels,
        'probs': all_probs,
        'preds': all_preds
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
    parser = argparse.ArgumentParser(description='Evaluate a trained model and generate performance metrics')
    parser.add_argument('--data_dir', type=str, default='.', help='Path to data directory')
    parser.add_argument('--data_entry', type=str, default='./data/Data_Entry_2017.csv', help='Path to Data_Entry_2017.csv')
    parser.add_argument('--model_path', type=str, default='best_convnext_large_model_auroc_0.742.pth', help='Path to the trained model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--threshold', type=float, default=0.5, help='Prediction threshold')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Directory to save evaluation results')
    parser.add_argument('--sample_size', type=int, default=1000, help='Number of samples to use for evaluation')
    parser.add_argument('--test_list', type=str, default='data/test_list.txt', help='Path to test list file')
    parser.add_argument('--use_test_list', action='store_true', help='Use test list file instead of random sampling')
    parser.add_argument('--use_available_images', action='store_true', help='Use available images instead of test list')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

def main():
    """الدالة الرئيسية"""
    # تحليل المعلمات
    args = parse_args()
    
    # تعيين البذرة العشوائية للتكرارية
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # إنشاء مجلد النتائج
    os.makedirs(args.output_dir, exist_ok=True)
    
    # تحديد الجهاز
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # تحميل ومعالجة البيانات
    data_entry_path = os.path.join(args.data_dir, 'data', 'Data_Entry_2017.csv')
    print(f"Loading data from {data_entry_path}")
    
    try:
        df = pd.read_csv(data_entry_path)
        print(f"Loaded {len(df)} records from data entry file")
    except FileNotFoundError:
        print(f"Error: Data entry file {data_entry_path} not found")
        data_entry_path = args.data_entry
        print(f"Trying alternative path: {data_entry_path}")
        df = pd.read_csv(data_entry_path)
        print(f"Loaded {len(df)} records from data entry file")
    
    # تحديد مجموعة الاختبار
    if args.use_available_images:
        # استخدام الصور المتاحة
        available_images = find_available_images(args.data_dir, df)
        if available_images:
            # أخذ عينة من الصور المتاحة
            if args.sample_size > 0 and args.sample_size < len(available_images):
                sampled_images = random.sample(available_images, args.sample_size)
            else:
                sampled_images = available_images
                
            test_df = df[df['Image Index'].isin(sampled_images)]
            print(f"Test set size from available images: {len(test_df)}")
        else:
            print("Warning: No available images found, falling back to random sampling")
            if args.sample_size > 0 and args.sample_size < len(df):
                test_df = df.sample(n=args.sample_size, random_state=args.seed)
            else:
                test_df = df.sample(frac=0.2, random_state=args.seed)
    elif args.use_test_list:
        # استخدام قائمة الاختبار
        test_list_path = args.test_list
        test_images = load_image_list(test_list_path)
        if test_images:
            test_df = df[df['Image Index'].isin(test_images)]
            print(f"Test set size from test list: {len(test_df)}")
        else:
            print("Warning: No images found in test list, falling back to random sampling")
            if args.sample_size > 0 and args.sample_size < len(df):
                test_df = df.sample(n=args.sample_size, random_state=args.seed)
            else:
                test_df = df.sample(frac=0.2, random_state=args.seed)
    else:
        # أخذ عينة عشوائية للتقييم
        if args.sample_size > 0 and args.sample_size < len(df):
            test_df = df.sample(n=args.sample_size, random_state=args.seed)
        else:
            test_df = df.sample(frac=0.2, random_state=args.seed)
    
    print(f"Initial test set size: {len(test_df)}")
    
    # إنشاء التحويلات
    test_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # إنشاء مجموعة البيانات
    test_dataset = ChestXrayDataset(args.data_dir, test_df, transform=test_transform, train=False)
    
    # إنشاء محمل البيانات
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # تحميل النموذج
    model_path = args.model_path
    model = load_model(model_path, num_classes=14, device=device)
    
    # الانتقال إلى مجلد النتائج
    os.chdir(args.output_dir)
    
    # تقييم النموذج
    print("Evaluating model...")
    results = evaluate_model(model, test_loader, device, args.threshold)
    
    # حساب المقاييس
    print("Calculating metrics...")
    metrics_df = calculate_metrics(results['labels'], results['preds'], CLASS_NAMES)
    metrics_df.to_csv('class_metrics.csv', index=False)
    print(metrics_df)
    
    # حساب مصفوفة الارتباك لكل فئة
    print("Generating confusion matrices...")
    for i, cls_name in enumerate(CLASS_NAMES):
        cm = confusion_matrix(results['labels'][:, i], results['preds'][:, i])
        plot_confusion_matrix(cm, ['Negative', 'Positive'], normalize=False, 
                             title=f'Confusion Matrix - {cls_name}')
        plot_confusion_matrix(cm, ['Negative', 'Positive'], normalize=True, 
                             title=f'Normalized Confusion Matrix - {cls_name}')
    
    # رسم منحنيات ROC
    print("Generating ROC curves...")
    plot_roc_curves(results['labels'], results['probs'], CLASS_NAMES)
    
    # رسم منحنيات الدقة والاستدعاء
    print("Generating precision-recall curves...")
    plot_precision_recall_curves(results['labels'], results['probs'], CLASS_NAMES)
    
    # رسم مقاييس الأداء لكل فئة
    print("Generating class metrics plot...")
    plot_class_metrics(metrics_df)
    
    print(f"Evaluation completed! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 