# شرح ملف test_train.py خطوة بخطوة

هذا الملف يمثل البايبلاين الكامل لبناء وتدريب وتقييم نموذج تصنيف وتوطين أمراض الصدر بالأشعة السينية باستخدام بايثون وPyTorch. فيما يلي شرح مفصل لكل جزء مع كود توضيحي وشرح تحته.

---

## 1. الاستيراد والإعدادات الأولية

```python
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import timm
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from PIL import Image
import warnings
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import math
```

**الشرح:**
- استيراد جميع المكتبات اللازمة لمعالجة البيانات، بناء النموذج، التدريب، التقييم، والرسم البياني.

---

## 2. إعدادات الضبط (Configuration)

```python
class CFG:
    SEED = 42
    MODEL_NAME = 'convnext_large'
    IMG_SIZE = 512
    BATCH_SIZE = 4
    ACCUM_STEPS = 8
    NUM_WORKERS = 2
    EPOCHS = 50
    LR = 5e-6
    HEAD_LR = 1e-5
    WEIGHT_DECAY = 0.01
    SCHEDULER = 'CosineAnnealingLR'
    T_MAX = EPOCHS
    MIN_LR = 1e-7
    WARMUP_EPOCHS = 1
    LOSS_FN = 'FocalLoss'
    FOCAL_ALPHA = 0.6
    FOCAL_GAMMA = 2.0
    LABEL_SMOOTHING = 0.1
    PATIENCE = 20
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    GRAD_CLIP_NORM = 1.0
    DROPOUT_HEAD = 0.1
    DROPOUT_META = 0.1
    BASE_PATH = 'data'
    IMAGE_DIR = BASE_PATH
    DATA_ENTRY_PATH = os.path.join(BASE_PATH, 'Data_Entry_2017.csv')
    BBOX_PATH = os.path.join(BASE_PATH, 'BBox_List_2017.csv')
    CHECKPOINT_LOAD_PATH = 'best_model_epoch_28_auroc_0.9688.pth'
    START_FROM_SCRATCH = False
    LOAD_STRICT = True
    MIN_AUROC = 0.8
    AUROC_IMPROVE_THRESHOLD = 0.0001
    CONFUSION_MATRIX_THRESHOLD = 0.5
    MIXUP_ALPHA = 0.2
    BBOX_LOSS_WEIGHT = 1.0
```

**الشرح:**
- جميع الإعدادات الخاصة بالتجربة (اسم النموذج، حجم الصورة، معدل التعلم، عدد العصور، إلخ) في كلاس واحد لسهولة التعديل.

---

## 3. تعريف الفئات (Classes) والأمراض

```python
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', ... , 'No Finding']
TRAIN_CLASSES = [c for c in CLASS_NAMES if c != 'No Finding']
CLASSES_WITH_BBOX = ['Atelectasis', 'Cardiomegaly', ...]
```

**الشرح:**
- تحديد أسماء الأمراض التي سيتم تصنيفها، وتحديد الأمراض التي لديها توطين (Bounding Box).

---

## 4. دوال الخسارة (Loss Functions)

### Focal Loss
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=CFG.FOCAL_ALPHA, gamma=CFG.FOCAL_GAMMA, label_smoothing=CFG.LABEL_SMOOTHING, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    def forward(self, inputs, targets):
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing / inputs.shape[1]
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduction == 'mean':
            return F_loss.mean()
        else:
            return F_loss
```

**الشرح:**
- دالة خسارة متقدمة تعالج مشكلة عدم توازن الفئات وتمنع النموذج من التركيز على الفئات السهلة فقط.

---

## 5. دوال مساعدة (Helper Functions)

### البحث عن الصورة
```python
def find_image_path(image_name, base_folder):
    for i in range(1, 13):
        image_path = os.path.join(base_folder, f'images_{i:03d}', 'images', image_name)
        if os.path.exists(image_path):
            return image_path
    image_path_flat = os.path.join(base_folder, 'images', image_name)
    if os.path.exists(image_path_flat):
        return image_path_flat
    return None
```

**الشرح:**
- تبحث عن الصورة في جميع المجلدات الفرعية أو في مجلد images الرئيسي.

---

## 6. كلاس البيانات (Dataset)

```python
class ChestXrayDataset(Dataset):
    def __init__(self, image_dir, df, bbox_dict, transform=None, train=True):
        ...
    def __len__(self):
        ...
    def __getitem__(self, idx):
        ...
```

**الشرح:**
- كلاس مسؤول عن تجهيز كل صورة ووسومها وصندوقها وبياناتها الوصفية عند الطلب (On-the-fly)، مع تطبيق التحسينات.

---

## 7. تعريف النموذج (Model)

```python
class AdvancedChestModel(nn.Module):
    def __init__(self, model_name='convnext_large', num_classes=14, metadata_features=3, pretrained=False):
        ...
    def forward(self, x_img, x_meta):
        ...
```

**الشرح:**
- النموذج يعتمد على ConvNeXt-Large، ويحتوي على فروع للانتباه، التوطين، ودمج البيانات الوصفية مع ميزات الصورة.

---

## 8. التحسينات (Augmentation)

```python
def get_transforms(img_size, is_train=True):
    if ALBUMENTATIONS_INSTALLED:
        if is_train:
            return A.Compose([
                A.Resize(...),
                A.HorizontalFlip(p=0.5),
                ...
            ])
        else:
            return A.Compose([
                A.Resize(...),
                ...
            ])
    else:
        ...
```

**الشرح:**
- دالة تعيد سلسلة التحسينات المناسبة حسب ما إذا كان التدريب أو التحقق.

---

## 9. حساب الخسارة (Loss Calculation)

```python
def calculate_loss(cls_outputs, bbox_outputs, labels, bboxes, criterion_cls, criterion_bbox, device, mixup=False, y_a=None, y_b=None, lam=1.0):
    ...
    total_loss = cls_loss + CFG.BBOX_LOSS_WEIGHT * bbox_loss
    return total_loss, cls_loss, bbox_loss
```

**الشرح:**
- تجمع بين خسارة التصنيف وخسارة التوطين (L1 + IoU)، وتدعم Mixup.

---

## 10. تحميل النموذج من نقطة حفظ (Checkpoint)

```python
def safe_load_checkpoint(model, optimizer, scheduler, checkpoint_path, device, load_strict):
    ...
    return start_epoch, best_auroc, metrics, reset_optimizer_scheduler, epochs_no_improve
```

**الشرح:**
- لتحميل النموذج والمحسن والجدولة من ملف نقطة حفظ، مع معالجة الأخطاء.

---

## 11. دوال التدريب والتحقق (Training/Validation)

### تدريب عصر واحد
```python
def train_one_epoch(model, loader, optimizer, criterion_cls, criterion_bbox, device, scaler, epoch, scheduler):
    ...
    return avg_loss, avg_cls_loss, avg_bbox_loss
```

### تحقق عصر واحد
```python
def validate_one_epoch(model, loader, criterion_cls, criterion_bbox, device, epoch):
    ...
    return avg_loss, avg_cls_loss, avg_bbox_loss, mean_auroc, aurocs, all_targets, all_outputs, best_no_finding_accuracy, best_no_finding_threshold, confusion_matrices
```

**الشرح:**
- تدريب النموذج على دفعة واحدة من البيانات، وتحقق الأداء على بيانات التحقق، مع حساب جميع المقاييس المهمة.

---

## 12. دوال التصور (Plotting)

```python
def plot_training_progress(metrics, num_epochs, save_path='training_progress.png'):
    ...
def plot_class_metrics(class_aurocs, save_path='class_aurocs.png'):
    ...
def plot_roc_curves(targets, outputs, class_names, epoch, mean_auc, save_path='roc_curves.png'):
    ...
def plot_confusion_matrices(confusion_matrices, class_names, epoch, save_path='confusion_matrices'):
    ...
```

**الشرح:**
- رسم منحنيات التدريب، منحنيات ROC، مصفوفات الالتباس، إلخ.

---

## 13. التنفيذ الرئيسي (Main Execution)

```python
if __name__ == '__main__':
    ...
    # تحميل البيانات
    df_main = pd.read_csv(CFG.DATA_ENTRY_PATH)
    ...
    # معالجة البيانات
    df_processed, _, _ = preprocess_metadata(df_processed)
    ...
    # تقسيم البيانات
    train_df, valid_df = train_test_split(df_processed, test_size=0.2, random_state=CFG.SEED)
    ...
    # تحميل الصناديق
    bbox_dict = load_bounding_boxes(CFG.BBOX_PATH)
    ...
    # إنشاء Datasets و DataLoaders
    train_dataset = ChestXrayDataset(...)
    valid_dataset = ChestXrayDataset(...)
    ...
    # تهيئة النموذج والمحسن والجدولة
    model = AdvancedChestModel(...)
    optimizer = torch.optim.AdamW(...)
    scheduler = ...
    ...
    # تحميل نقطة حفظ إذا وجدت
    if not CFG.START_FROM_SCRATCH:
        start_epoch, best_auroc, metrics, reset_optimizer_scheduler, epochs_no_improve = safe_load_checkpoint(...)
    ...
    # حلقة التدريب
    for epoch in range(start_epoch, CFG.EPOCHS):
        train_loss, train_cls_loss, train_bbox_loss = train_one_epoch(...)
        val_loss, val_cls_loss, val_bbox_loss, mean_auroc, ... = validate_one_epoch(...)
        ...
        # حفظ أفضل نموذج
        if mean_auroc > 0.88:
            ...
            torch.save(model.state_dict(), best_model_path)
        ...
        # التوقف المبكر
        if epochs_no_improve >= CFG.PATIENCE or mean_auroc < CFG.MIN_AUROC:
            ...
            break
    ...
```

**الشرح:**
- يبدأ البرنامج من هنا: تحميل البيانات، تجهيزها، تقسيمها، تحميل الصناديق، إنشاء Datasets وLoaders، تهيئة النموذج، التدريب والتحقق، حفظ أفضل نموذج، رسم النتائج، التوقف المبكر.

---

## ملخص
- الملف عبارة عن بايبلاين متكامل لبناء وتدريب وتقييم نموذج تصنيف وتوطين أمراض الصدر بالأشعة السينية.
- كل جزء من الكود مسؤول عن خطوة محددة: التحميل، التحسين، التدريب، التحقق، التصور، الحفظ.
- يدعم تقنيات متقدمة مثل Attention, Mixup, Multi-task Learning, Metadata Fusion, Early Stopping, Visualization.
- كل شيء منظم وقابل للتعديل بسهولة.

---

> إذا أردت شرح أعمق لأي جزء أو كود معين، أخبرني بذلك! 