import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

"""
شرح استخدام WeightedRandomSampler في PyTorch لمعالجة عدم توازن البيانات في مجموعات البيانات متعددة التصنيفات
مثل مجموعة بيانات الأشعة السينية للصدر
"""

class ChestXrayDataset(Dataset):
    """
    فئة Dataset للأشعة السينية للصدر
    """
    def __init__(self, csv_file, img_dir, transform=None):
        """
        تهيئة مجموعة البيانات
        
        المعلمات:
        - csv_file: مسار ملف CSV يحتوي على أسماء الصور والتصنيفات
        - img_dir: مجلد الصور
        - transform: تحويلات لتطبيقها على الصور
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # قائمة بالأمراض الـ 14
        self.diseases = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
            'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 
            'Pleural_Thickening', 'Hernia'
        ]
        
    def __len__(self):
        return len(self.data_frame)
        
    def __getitem__(self, idx):
        img_name = self.data_frame.iloc[idx, 0]  # افتراض أن اسم الصورة في العمود الأول
        image = Image.open(f"{self.img_dir}/{img_name}").convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # استخراج التصنيفات (0 أو 1) لكل مرض
        labels = torch.zeros(len(self.diseases))
        for i, disease in enumerate(self.diseases):
            if disease in self.data_frame.iloc[idx, 1]:  # افتراض أن التصنيفات في العمود الثاني
                labels[i] = 1
                
        return image, labels


def create_weighted_sampler(dataset):
    """
    إنشاء WeightedRandomSampler لمعالجة عدم توازن البيانات
    
    الطريقة:
    1. حساب تكرار كل فئة (تصنيف) في مجموعة البيانات
    2. حساب الأوزان العكسية لكل مثال استنادًا إلى تكرار الفئات
    3. إنشاء WeightedRandomSampler باستخدام هذه الأوزان
    
    ملاحظة: في حالة التصنيف متعدد التسميات، نستخدم أسلوبًا مختلفًا عن تصنيف الفئة الواحدة
    """
    # تحميل كافة التصنيفات
    all_labels = []
    for _, labels in dataset:
        all_labels.append(labels)
    
    all_labels = torch.stack(all_labels)
    
    # طريقة 1: استنادًا إلى متوسط عدد الحالات الإيجابية لكل مرض
    # وهذه مناسبة للتصنيف متعدد التسميات
    pos_count = torch.sum(all_labels, dim=0)
    neg_count = len(dataset) - pos_count
    pos_weights = neg_count / pos_count
    
    # حساب وزن لكل نموذج في مجموعة البيانات
    weights = torch.zeros(len(dataset))
    for idx, label in enumerate(all_labels):
        # الوزن هو متوسط الأوزان للفئات الإيجابية في هذا النموذج
        pos_indices = torch.where(label == 1)[0]
        if len(pos_indices) > 0:
            weights[idx] = torch.mean(pos_weights[pos_indices])
        else:
            weights[idx] = 0.1  # وزن منخفض للنماذج التي لا تحتوي على أي تصنيفات إيجابية
    
    # تطبيع الأوزان
    weights = weights / weights.sum()
    
    # إنشاء WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True
    )
    
    return sampler


def main():
    """
    مثال كامل على استخدام WeightedRandomSampler في تدريب نموذج
    """
    # تحويلات الصور
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # إنشاء مجموعات البيانات
    train_dataset = ChestXrayDataset(
        csv_file='train.csv',
        img_dir='data/images',
        transform=transform
    )
    
    # تحليل توزيع البيانات
    print("تحليل توزيع البيانات:")
    all_labels = []
    for _, labels in train_dataset:
        all_labels.append(labels)
    all_labels = torch.stack(all_labels)
    pos_count = torch.sum(all_labels, dim=0)
    
    for i, disease in enumerate(train_dataset.diseases):
        print(f"{disease}: {pos_count[i]} من {len(train_dataset)} ({pos_count[i]/len(train_dataset)*100:.2f}%)")
    
    # إنشاء الـ sampler
    sampler = create_weighted_sampler(train_dataset)
    
    # إنشاء الـ DataLoader مع الـ sampler
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=sampler,  # استخدام WeightedRandomSampler
        num_workers=4,
        pin_memory=True
    )
    
    print("\nبدء التدريب مع استخدام WeightedRandomSampler...")
    
    # هنا يمكن إضافة كود التدريب الفعلي للنموذج
    # model = YourModel()
    # criterion = torch.nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 
    # for epoch in range(num_epochs):
    #     for images, labels in train_loader:
    #         outputs = model(images)
    #         loss = criterion(outputs, labels)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()


"""
ملاحظات إضافية حول استخدام WeightedRandomSampler:

1. المشكلة:
   - في مجموعات بيانات الأشعة السينية، عادة ما تكون هناك اختلافات كبيرة في تكرار الأمراض المختلفة.
   - مثلاً، قد يكون لديك 60% من الصور تحتوي على Infiltration ولكن فقط 2% تحتوي على Hernia.
   - هذا يؤدي إلى تحيز النموذج نحو الفئات الأكثر تكراراً.

2. الحل باستخدام WeightedRandomSampler:
   - يسمح بأخذ عينات من البيانات بناءً على أوزان محددة.
   - يمكننا إعطاء أوزان أعلى للفئات النادرة وأوزان أقل للفئات الشائعة.
   - هذا يجعل تدريب النموذج أكثر توازناً.

3. استراتيجيات بديلة:
   - استخدام فقدان موزون (Weighted Loss) بدلاً من أو بالإضافة إلى العينات الموزونة.
   - استخدام تقنيات مثل SMOTE لتوليد بيانات اصطناعية للفئات النادرة.
   - تقليل عدد العينات من الفئات الأكثر تكراراً (Under-sampling).
"""

if __name__ == "__main__":
    main() 