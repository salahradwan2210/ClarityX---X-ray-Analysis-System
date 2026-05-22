import numpy as np
import pandas as pd
import imageio.v2 as imageio
import skimage.transform
import pickle
import sys, os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing

# Define the 15 classes including No Finding
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
               'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
               'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
               'Pleural_Thickening', 'Hernia', 'No Finding']

def analyze_class_distribution(meta_data):
    """تحليل توزيع الفئات في البيانات"""
    all_labels = []
    for labels in meta_data["Finding Labels"]:
        all_labels.extend(labels.split("|"))
    label_counts = Counter(all_labels)
    print("\nClass distribution:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")
    return label_counts

def balance_dataset(meta_data, max_no_finding_ratio=2.0):
    """
    موازنة البيانات مع تحديد نسبة صور No Finding
    max_no_finding_ratio: النسبة القصوى لعدد صور No Finding مقارنة بمتوسط الفئات الأخرى
    """
    # فصل البيانات إلى No Finding والفئات الأخرى
    no_finding_data = meta_data[meta_data["Finding Labels"] == "No Finding"]
    other_data = meta_data[meta_data["Finding Labels"] != "No Finding"]
    
    # حساب متوسط عدد الصور للفئات الأخرى
    label_counts = analyze_class_distribution(other_data)
    avg_other_count = np.mean([count for label, count in label_counts.items()])
    
    # تحديد العدد المطلوب من صور No Finding
    target_no_finding_count = int(avg_other_count * max_no_finding_ratio)
    
    print(f"\nBalancing dataset:")
    print(f"Average count for other classes: {avg_other_count:.0f}")
    print(f"Original No Finding count: {len(no_finding_data)}")
    print(f"Target No Finding count: {target_no_finding_count}")
    
    # أخذ عينة عشوائية من صور No Finding
    if len(no_finding_data) > target_no_finding_count:
        no_finding_data = no_finding_data.sample(n=target_no_finding_count, random_state=42)
    
    # دمج البيانات
    balanced_data = pd.concat([other_data, no_finding_data])
    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Final dataset size: {len(balanced_data)}")
    return balanced_data

def balance_dataset_weights(y):
    """Calculate weights for each class to balance the dataset"""
    label_counts = np.sum(y, axis=0)
    total_samples = len(y)
    
    # حساب الأوزان مع تجنب القسمة على صفر
    class_weights = np.zeros_like(label_counts, dtype=np.float32)
    for i in range(len(CLASS_NAMES)):
        if label_counts[i] > 0:  # تجنب القسمة على صفر
            if i == CLASS_NAMES.index('No Finding'):
                # وزن أقل لـ No Finding لأنها الفئة الأكثر شيوعاً
                class_weights[i] = total_samples / (len(CLASS_NAMES) * label_counts[i] * 2)
            else:
                class_weights[i] = total_samples / (len(CLASS_NAMES) * label_counts[i])
        else:
            class_weights[i] = 0.0  # وزن صفر للفئات غير الموجودة
    
    # تطبيع الأوزان
    if np.max(class_weights) > 0:
        class_weights = class_weights / np.max(class_weights)
    
    sample_weights = np.ones(len(y), dtype=np.float32)  # البدء بأوزان متساوية
    
    for i in range(len(y)):
        if np.sum(y[i]) == 0:  # إذا لم يكن هناك أي تشخيص
            sample_weights[i] = class_weights[-1] if class_weights[-1] > 0 else 1.0
        else:
            # حساب الوزن كمتوسط أوزان الفئات الموجودة
            active_classes = y[i][:-1] > 0  # تجاهل No Finding
            if np.any(active_classes):
                weights = class_weights[:-1][active_classes]
                sample_weights[i] = np.mean(weights[weights > 0]) if np.any(weights > 0) else 1.0
    
    # تطبيع الأوزان النهائية
    sample_weights = sample_weights / np.mean(sample_weights)
    
    return sample_weights

def get_labels(pic_id, meta_data):
    labels = meta_data.loc[meta_data["Image Index"] == pic_id, "Finding Labels"]
    label_list = labels.tolist()[0].split("|")
    # إضافة No Finding كفئة منفصلة
    if "No Finding" in label_list:
        return ["No Finding"]
    return label_list

def process_image(image_path, target_size=(512, 512)):
    """معالجة الصورة مع تحسين التباين"""
    try:
        img = imageio.imread(image_path)
        if len(img.shape) == 3:
            img = img[:,:,0]
        
        # تحسين التباين
        p2, p98 = np.percentile(img, (2, 98))
        img = np.clip(img, p2, p98)
        img = (img - p2) / (p98 - p2)
        
        # تغيير الحجم
        img_resized = skimage.transform.resize(img, target_size)
        return np.array(img_resized, dtype=np.float32)
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def find_image_path(image_name, base_folder):
    """البحث عن الصورة في جميع المجلدات"""
    for i in range(1, 13):  # من 1 إلى 12
        folder_name = f'images_{i:03d}'  # مثال: images_001
        image_path = os.path.join(base_folder, folder_name, 'images', image_name)
        if os.path.exists(image_path):
            return image_path
    return None

def process_dataset(data, image_folder_path, output_path, split, meta_data):
    """معالجة مجموعة البيانات وحفظها في دفعات"""
    print(f"\nProcessing {split} dataset...")
    
    # تحديد حجم الدفعة الأكبر
    BATCH_SIZE = 1000  # زيادة حجم الدفعة من 100 إلى 1000
    
    # تقسيم البيانات إلى دفعات
    num_batches = (len(data) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Total images: {len(data)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of batches: {num_batches}")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, len(data))
        batch_data = data.iloc[start_idx:end_idx]
        
        batch_images = []
        batch_labels = []
        
        print(f"\nProcessing batch {batch_idx + 1}/{num_batches}")
        print(f"Images {start_idx} to {end_idx}")
        
        for _, row in tqdm(batch_data.iterrows(), total=len(batch_data)):
            image_path = find_image_path(row["Image Index"], image_folder_path)
            if image_path:
                img = process_image(image_path)
                if img is not None:
                    label = get_labels(row["Image Index"], meta_data)
                    batch_images.append(img)
                    batch_labels.append(label)
        
        if batch_images:
            # تحويل البيانات
            batch_images = np.array(batch_images)
            encoder = MultiLabelBinarizer(classes=CLASS_NAMES)
            batch_labels = encoder.fit_transform(batch_labels)
            
            # حساب الأوزان للموازنة
            batch_weights = balance_dataset_weights(batch_labels)
            
            # حفظ البيانات
            print(f"Saving batch {batch_idx} with {len(batch_images)} images")
            np.save(os.path.join(output_path, f"{split}_X_batch_{batch_idx}.npy"), batch_images)
            np.save(os.path.join(output_path, f"{split}_y_batch_{batch_idx}.npy"), batch_labels)
            np.save(os.path.join(output_path, f"{split}_weights_batch_{batch_idx}.npy"), batch_weights)
            
            # حفظ معلومات المشفر للاستخدام لاحقاً
            if batch_idx == 0:
                with open(os.path.join(output_path, f"{split}_label_encoder.pkl"), "wb") as f:
                    pickle.dump(encoder, f)

def main():
    if len(sys.argv) != 4:
        print("Usage: python preprocessing.py <image_folder_path> <data_entry_csv> <output_path>")
        sys.exit(1)

    image_folder_path = sys.argv[1]
    data_entry_path = sys.argv[2]
    output_path = sys.argv[3]
    
    os.makedirs(output_path, exist_ok=True)
    print(f"Processing data...")
    print(f"Image folder: {image_folder_path}")
    print(f"Data entry file: {data_entry_path}")
    print(f"Output path: {output_path}")

    # قراءة البيانات الوصفية
    meta_data = pd.read_csv(data_entry_path)
    print(f"Total images in metadata: {len(meta_data)}")
    
    # تحليل توزيع الفئات قبل الموازنة
    print("\nClass distribution before balancing:")
    label_distribution = analyze_class_distribution(meta_data)
    
    # موازنة البيانات
    balanced_data = balance_dataset(meta_data, max_no_finding_ratio=2.0)
    
    # تحليل توزيع الفئات بعد الموازنة
    print("\nClass distribution after balancing:")
    label_distribution = analyze_class_distribution(balanced_data)
    
    # تقسيم البيانات بدون stratify
    train_data = balanced_data.sample(frac=0.8, random_state=42)
    valid_data = balanced_data.drop(train_data.index)
    
    print(f"\nTraining samples: {len(train_data)}")
    print(f"Validation samples: {len(valid_data)}")
    
    # معالجة الصور وحفظها
    process_dataset(train_data, image_folder_path, output_path, "train", meta_data)
    process_dataset(valid_data, image_folder_path, output_path, "valid", meta_data)

if __name__ == "__main__":
    multiprocessing.freeze_support()  # ضروري لـ Windows
    main()