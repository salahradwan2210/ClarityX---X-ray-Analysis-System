import numpy as np
import pandas as pd
import imageio.v2 as imageio
import skimage.transform
import pickle
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing

# Define the 15 classes including No Finding
CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia', 'No Finding'
]

def analyze_class_distribution(meta_data):
    """
    تحليل توزيع الفئات في البيانات
    
    المعلمات:
        meta_data (DataFrame): إطار البيانات الذي يحتوي على تسميات الصور
        
    الإرجاع:
        Counter: عدد تكرار كل فئة
    """
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
    
    المعلمات:
        meta_data (DataFrame): إطار البيانات الذي يحتوي على تسميات الصور
        max_no_finding_ratio (float): النسبة القصوى لعدد صور No Finding مقارنة بمتوسط الفئات الأخرى
        
    الإرجاع:
        DataFrame: إطار البيانات المتوازن
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
    
    if len(no_finding_data) > target_no_finding_count:
        # أخذ عينة عشوائية من صور No Finding
        no_finding_sample = no_finding_data.sample(target_no_finding_count, random_state=42)
        # دمج العينة مع البيانات الأخرى
        balanced_data = pd.concat([other_data, no_finding_sample])
        print(f"Reduced No Finding samples from {len(no_finding_data)} to {len(no_finding_sample)}")
    else:
        balanced_data = meta_data.copy()
        print("No balance needed, No Finding count is already below the target ratio")
    
    # خلط البيانات المتوازنة
    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return balanced_data

def balance_dataset_weights(y):
    """
    حساب أوزان الفئات لتعويض عدم التوازن في التدريب
    
    المعلمات:
        y (numpy.array): مصفوفة التسميات (one-hot encoding)
        
    الإرجاع:
        numpy.array: أوزان كل عينة
    """
    # حساب عدد العينات الإيجابية لكل فئة
    positive_counts = np.sum(y, axis=0)
    total_samples = len(y)
    
    # حساب أوزان الفئات (عكس تردد الفئة)
    class_weights = total_samples / (positive_counts * len(positive_counts))
    
    # حساب أوزان العينات بناء على الفئات الموجودة فيها
    sample_weights = np.zeros(total_samples)
    for i in range(len(y)):
        sample_weights[i] = np.sum(y[i] * class_weights) / np.sum(y[i])
    
    return sample_weights

def balance_dataset_weights_advanced(y, class_weights=None, sample_weights=None):
    """
    حساب أوزان الفئات والعينات بشكل متقدم لتعويض عدم التوازن في التدريب
    
    المعلمات:
        y (numpy.array): مصفوفة التسميات (one-hot encoding)
        class_weights (numpy.array): أوزان الفئات المخصصة (اختياري)
        sample_weights (numpy.array): أوزان العينات المخصصة (اختياري)
        
    الإرجاع:
        tuple: (أوزان الفئات، أوزان العينات)
    """
    # حساب عدد العينات الإيجابية لكل فئة
    positive_counts = np.sum(y, axis=0)
    total_samples = len(y)
    
    # حساب أوزان الفئات إذا لم يتم توفيرها
    if class_weights is None:
        # حساب أوزان الفئات باستخدام معادلة متقدمة
        class_weights = np.zeros(len(positive_counts))
        for i in range(len(positive_counts)):
            if positive_counts[i] > 0:
                # استخدام معادلة متقدمة تأخذ في الاعتبار التردد النسبي
                class_weights[i] = np.log(total_samples / positive_counts[i])
            else:
                class_weights[i] = 0
    
    # حساب أوزان العينات إذا لم يتم توفيرها
    if sample_weights is None:
        sample_weights = np.zeros(total_samples)
        for i in range(len(y)):
            # حساب وزن العينة بناء على الفئات الموجودة فيها
            positive_classes = np.where(y[i] == 1)[0]
            if len(positive_classes) > 0:
                # استخدام متوسط أوزان الفئات الموجودة
                sample_weights[i] = np.mean(class_weights[positive_classes])
            else:
                sample_weights[i] = 0
    
    # تطبيع الأوزان
    if np.sum(sample_weights) > 0:
        sample_weights = sample_weights / np.sum(sample_weights) * total_samples
    
    return class_weights, sample_weights

def get_labels(pic_id, meta_data):
    """
    الحصول على تسميات صورة معينة
    
    المعلمات:
        pic_id (str): معرف الصورة
        meta_data (DataFrame): إطار البيانات الذي يحتوي على تسميات الصور
        
    الإرجاع:
        list: قائمة بتسميات الصورة
    """
    labels = meta_data.loc[meta_data["Image Index"] == pic_id, "Finding Labels"].values[0]
    return labels.split("|")

def process_image(image_path, target_size=(224, 224)):
    """
    معالجة الصورة وتحويلها إلى الحجم المطلوب
    
    المعلمات:
        image_path (str): مسار الصورة
        target_size (tuple): الحجم المطلوب للصورة
        
    الإرجاع:
        numpy.array: الصورة المعالجة
    """
    try:
        # قراءة الصورة
        img = imageio.imread(image_path)
        
        # تحويل الصورة إلى الحجم المطلوب
        img_resized = skimage.transform.resize(img, target_size, anti_aliasing=True, preserve_range=True)
        
        # تطبيع الصورة
        img_normalized = img_resized / 255.0
        
        return img_normalized
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def find_image_path(image_name, base_folder):
    """
    البحث عن مسار الصورة في المجلدات الفرعية
    
    المعلمات:
        image_name (str): اسم الصورة
        base_folder (str): المجلد الأساسي
        
    الإرجاع:
        str: مسار الصورة الكامل
    """
    for i in range(1, 13):  # من 1 إلى 12
        folder_name = f'images_{i:03d}'  # مثال: images_001
        image_path = os.path.join(base_folder, folder_name, "images", image_name)
        if os.path.exists(image_path):
            return image_path
    return None

def process_dataset(meta_data, image_folder_path, output_folder_path, split_ratio=0.2):
    """
    معالجة مجموعة البيانات وحفظها في ملفات
    
    المعلمات:
        meta_data (DataFrame): إطار البيانات الذي يحتوي على تسميات الصور
        image_folder_path (str): مسار مجلد الصور
        output_folder_path (str): مسار المجلد لحفظ البيانات المعالجة
        split_ratio (float): نسبة بيانات الاختبار
        
    الإرجاع:
        None
    """
    # إنشاء مجلد الإخراج إذا لم يكن موجودًا
    os.makedirs(output_folder_path, exist_ok=True)
    
    # تحويل التسميات إلى one-hot encoding
    mlb = MultiLabelBinarizer(classes=CLASS_NAMES)
    pic_ids = meta_data["Image Index"].values
    labels_list = [get_labels(pic_id, meta_data) for pic_id in pic_ids]
    y = mlb.fit_transform(labels_list)
    
    # تقسيم البيانات إلى تدريب واختبار
    train_indices, test_indices = train_test_split(
        range(len(pic_ids)), test_size=split_ratio, random_state=42, stratify=np.argmax(y, axis=1)
    )
    
    # معالجة الصور وحفظها
    num_cores = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # معالجة صور التدريب
        train_futures = []
        print(f"\nProcessing {len(train_indices)} training images with {num_cores} CPU cores")
        for idx in train_indices:
            pic_id = pic_ids[idx]
            image_path = find_image_path(pic_id, image_folder_path)
            if image_path:
                train_futures.append(executor.submit(process_image, image_path))
        
        # جمع النتائج لصور التدريب
        X_train = []
        for future in tqdm(as_completed(train_futures), total=len(train_futures), desc="Training images"):
            result = future.result()
            if result is not None:
                X_train.append(result)
        X_train = np.array(X_train)
        y_train = y[train_indices]
        
        # معالجة صور الاختبار
        test_futures = []
        print(f"\nProcessing {len(test_indices)} test images with {num_cores} CPU cores")
        for idx in test_indices:
            pic_id = pic_ids[idx]
            image_path = find_image_path(pic_id, image_folder_path)
            if image_path:
                test_futures.append(executor.submit(process_image, image_path))
        
        # جمع النتائج لصور الاختبار
        X_test = []
        for future in tqdm(as_completed(test_futures), total=len(test_futures), desc="Test images"):
            result = future.result()
            if result is not None:
                X_test.append(result)
        X_test = np.array(X_test)
        y_test = y[test_indices]
    
    # حفظ البيانات المعالجة
    print(f"\nSaving processed data to {output_folder_path}")
    np.save(os.path.join(output_folder_path, "X_train.npy"), X_train)
    np.save(os.path.join(output_folder_path, "y_train.npy"), y_train)
    np.save(os.path.join(output_folder_path, "X_test.npy"), X_test)
    np.save(os.path.join(output_folder_path, "y_test.npy"), y_test)
    
    # حساب وحفظ أوزان العينات
    sample_weights = balance_dataset_weights(y_train)
    np.save(os.path.join(output_folder_path, "sample_weights.npy"), sample_weights)
    
    print(f"Processed and saved {len(X_train)} training and {len(X_test)} test images")

def main(data_path="./data/Data_Entry_2017.csv", image_folder="./data", output_folder="./preprocessed4", balanced=True):
    """
    الدالة الرئيسية لمعالجة البيانات
    
    المعلمات:
        data_path (str): مسار ملف البيانات الوصفية
        image_folder (str): مسار مجلد الصور
        output_folder (str): مسار المجلد لحفظ البيانات المعالجة
        balanced (bool): ما إذا كان يتم موازنة البيانات
        
    الإرجاع:
        None
    """
    # قراءة البيانات الوصفية
    meta_data = pd.read_csv(data_path)
    print(f"Read {len(meta_data)} entries from {data_path}")
    
    # تحليل توزيع الفئات
    analyze_class_distribution(meta_data)
    
    # موازنة البيانات إذا كان مطلوبًا
    if balanced:
        meta_data = balance_dataset(meta_data)
        analyze_class_distribution(meta_data)
    
    # معالجة البيانات وحفظها
    process_dataset(meta_data, image_folder, output_folder)
    
    print("\nPreprocessing completed!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess NIH Chest X-ray Dataset')
    parser.add_argument('--data_path', type=str, default='./data/Data_Entry_2017.csv', help='Path to Data_Entry_2017.csv')
    parser.add_argument('--image_folder', type=str, default='./data', help='Path to image folder')
    parser.add_argument('--output_folder', type=str, default='./preprocessed4', help='Path to output folder')
    parser.add_argument('--balanced', action='store_true', help='Balance the dataset')
    
    args = parser.parse_args()
    
    main(args.data_path, args.image_folder, args.output_folder, args.balanced) 