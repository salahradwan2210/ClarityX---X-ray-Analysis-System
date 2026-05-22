import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def load_data(data_path="./data/Data_Entry_2017.csv"):
    """
    تحميل بيانات Data_Entry_2017.csv
    
    المعلمات:
        data_path (str): مسار ملف البيانات
        
    الإرجاع:
        DataFrame: بيانات المرضى والصور
    """
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    print(f"Loaded {len(data)} records")
    return data

def analyze_demographics(data):
    """
    تحليل البيانات الديموغرافية
    
    المعلمات:
        data (DataFrame): بيانات المرضى والصور
    """
    # تحليل توزيع الأعمار
    plt.figure(figsize=(12, 6))
    sns.histplot(data['Patient Age'], bins=20, kde=True)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.savefig('age_distribution.png')
    
    # تحليل توزيع الجنس
    gender_counts = data['Patient Gender'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')
    plt.title('Gender Distribution')
    plt.savefig('gender_distribution.png')
    
    # تحليل توزيع وضعية الصورة
    view_counts = data['View Position'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=view_counts.index, y=view_counts.values)
    plt.title('View Position Distribution')
    plt.xlabel('View Position')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.savefig('view_position_distribution.png')
    
    # تحليل العلاقة بين العمر والأمراض
    disease_by_age = {}
    for _, row in data.iterrows():
        age = row['Patient Age']
        diseases = row['Finding Labels'].split('|')
        for disease in diseases:
            if disease not in disease_by_age:
                disease_by_age[disease] = []
            disease_by_age[disease].append(age)
    
    plt.figure(figsize=(15, 10))
    for i, (disease, ages) in enumerate(disease_by_age.items()):
        if disease == 'No Finding':
            continue
        plt.subplot(4, 4, i+1)
        sns.histplot(ages, bins=10, kde=True)
        plt.title(disease)
        plt.xlabel('Age')
    plt.tight_layout()
    plt.savefig('disease_by_age.png')
    
    # تحليل العلاقة بين الجنس والأمراض
    disease_by_gender = {}
    for _, row in data.iterrows():
        gender = row['Patient Gender']
        diseases = row['Finding Labels'].split('|')
        for disease in diseases:
            if disease not in disease_by_gender:
                disease_by_gender[disease] = {'M': 0, 'F': 0}
            disease_by_gender[disease][gender] += 1
    
    diseases = []
    male_counts = []
    female_counts = []
    for disease, counts in disease_by_gender.items():
        if disease == 'No Finding':
            continue
        diseases.append(disease)
        male_counts.append(counts['M'])
        female_counts.append(counts['F'])
    
    plt.figure(figsize=(15, 8))
    x = np.arange(len(diseases))
    width = 0.35
    plt.bar(x - width/2, male_counts, width, label='Male')
    plt.bar(x + width/2, female_counts, width, label='Female')
    plt.xticks(x, diseases, rotation=45)
    plt.xlabel('Disease')
    plt.ylabel('Count')
    plt.title('Disease Distribution by Gender')
    plt.legend()
    plt.tight_layout()
    plt.savefig('disease_by_gender.png')
    
    # تحليل العلاقة بين وضعية الصورة والأمراض
    disease_by_view = {}
    for _, row in data.iterrows():
        view = row['View Position']
        diseases = row['Finding Labels'].split('|')
        for disease in diseases:
            if disease not in disease_by_view:
                disease_by_view[disease] = Counter()
            disease_by_view[disease][view] += 1
    
    plt.figure(figsize=(15, 10))
    for i, (disease, views) in enumerate(disease_by_view.items()):
        if disease == 'No Finding' or i >= 16:
            continue
        plt.subplot(4, 4, i+1)
        view_positions = list(views.keys())
        counts = list(views.values())
        sns.barplot(x=view_positions, y=counts)
        plt.title(disease)
        plt.xlabel('View Position')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('disease_by_view.png')
    
    return {
        'age_stats': {
            'mean': data['Patient Age'].mean(),
            'median': data['Patient Age'].median(),
            'min': data['Patient Age'].min(),
            'max': data['Patient Age'].max()
        },
        'gender_stats': gender_counts.to_dict(),
        'view_stats': view_counts.to_dict()
    }

def analyze_disease_patterns(data):
    """
    تحليل أنماط الأمراض
    
    المعلمات:
        data (DataFrame): بيانات المرضى والصور
    """
    # استخراج جميع الأمراض
    all_diseases = []
    for labels in data['Finding Labels']:
        diseases = labels.split('|')
        all_diseases.extend(diseases)
    
    # حساب تكرار كل مرض
    disease_counts = Counter(all_diseases)
    
    # رسم توزيع الأمراض
    plt.figure(figsize=(15, 8))
    diseases = [d for d in disease_counts.keys() if d != 'No Finding']
    counts = [disease_counts[d] for d in diseases]
    sns.barplot(x=diseases, y=counts)
    plt.title('Disease Distribution')
    plt.xlabel('Disease')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('disease_distribution.png')
    
    # تحليل الأمراض المتعددة
    multi_disease_counts = Counter()
    for labels in data['Finding Labels']:
        diseases = labels.split('|')
        if len(diseases) > 1 and 'No Finding' not in diseases:
            multi_disease_counts[tuple(sorted(diseases))] += 1
    
    # رسم أكثر 20 مجموعة أمراض شيوعاً
    plt.figure(figsize=(15, 8))
    common_combinations = multi_disease_counts.most_common(20)
    combinations = [' + '.join(combo) for combo, _ in common_combinations]
    counts = [count for _, count in common_combinations]
    sns.barplot(x=counts, y=combinations)
    plt.title('Most Common Disease Combinations')
    plt.xlabel('Count')
    plt.ylabel('Disease Combination')
    plt.tight_layout()
    plt.savefig('disease_combinations.png')
    
    # إنشاء مصفوفة التواجد المشترك
    unique_diseases = [d for d in disease_counts.keys() if d != 'No Finding']
    cooccurrence = np.zeros((len(unique_diseases), len(unique_diseases)))
    
    for labels in data['Finding Labels']:
        diseases = labels.split('|')
        if 'No Finding' in diseases:
            continue
        for i, disease1 in enumerate(unique_diseases):
            if disease1 in diseases:
                for j, disease2 in enumerate(unique_diseases):
                    if disease2 in diseases:
                        cooccurrence[i, j] += 1
    
    # رسم مصفوفة التواجد المشترك
    plt.figure(figsize=(12, 10))
    sns.heatmap(cooccurrence, annot=True, fmt='g', xticklabels=unique_diseases, yticklabels=unique_diseases)
    plt.title('Disease Co-occurrence Matrix')
    plt.tight_layout()
    plt.savefig('disease_cooccurrence.png')
    
    return {
        'disease_counts': disease_counts,
        'multi_disease_counts': dict(multi_disease_counts.most_common(20))
    }

def analyze_patient_history(data):
    """
    تحليل تاريخ المرضى
    
    المعلمات:
        data (DataFrame): بيانات المرضى والصور
    """
    # تجميع الصور حسب المريض
    patient_images = {}
    for _, row in data.iterrows():
        patient_id = row['Patient ID']
        if patient_id not in patient_images:
            patient_images[patient_id] = []
        patient_images[patient_id].append({
            'image': row['Image Index'],
            'follow_up': row['Follow-up #'],
            'findings': row['Finding Labels']
        })
    
    # حساب عدد الصور لكل مريض
    images_per_patient = [len(images) for images in patient_images.values()]
    
    # رسم توزيع عدد الصور لكل مريض
    plt.figure(figsize=(12, 6))
    sns.histplot(images_per_patient, bins=20, kde=True)
    plt.title('Number of Images per Patient')
    plt.xlabel('Number of Images')
    plt.ylabel('Count')
    plt.savefig('images_per_patient.png')
    
    # تحليل تطور الأمراض عبر الزمن
    disease_progression = {}
    for patient_id, images in patient_images.items():
        if len(images) <= 1:
            continue
        
        # ترتيب الصور حسب رقم المتابعة
        sorted_images = sorted(images, key=lambda x: x['follow_up'])
        
        # تتبع الأمراض عبر الزمن
        for i in range(len(sorted_images) - 1):
            current_diseases = set(sorted_images[i]['findings'].split('|'))
            next_diseases = set(sorted_images[i+1]['findings'].split('|'))
            
            # الأمراض التي ظهرت حديثاً
            new_diseases = next_diseases - current_diseases
            # الأمراض التي اختفت
            resolved_diseases = current_diseases - next_diseases
            
            for disease in new_diseases:
                if disease == 'No Finding':
                    continue
                if disease not in disease_progression:
                    disease_progression[disease] = {'appeared': 0, 'resolved': 0}
                disease_progression[disease]['appeared'] += 1
            
            for disease in resolved_diseases:
                if disease == 'No Finding':
                    continue
                if disease not in disease_progression:
                    disease_progression[disease] = {'appeared': 0, 'resolved': 0}
                disease_progression[disease]['resolved'] += 1
    
    # رسم تطور الأمراض
    diseases = list(disease_progression.keys())
    appeared = [disease_progression[d]['appeared'] for d in diseases]
    resolved = [disease_progression[d]['resolved'] for d in diseases]
    
    plt.figure(figsize=(15, 8))
    x = np.arange(len(diseases))
    width = 0.35
    plt.bar(x - width/2, appeared, width, label='Appeared')
    plt.bar(x + width/2, resolved, width, label='Resolved')
    plt.xticks(x, diseases, rotation=45)
    plt.xlabel('Disease')
    plt.ylabel('Count')
    plt.title('Disease Progression Over Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig('disease_progression.png')
    
    return {
        'images_per_patient_stats': {
            'mean': np.mean(images_per_patient),
            'median': np.median(images_per_patient),
            'min': min(images_per_patient),
            'max': max(images_per_patient)
        },
        'disease_progression': disease_progression
    }

def analyze_image_properties(data):
    """
    تحليل خصائص الصور
    
    المعلمات:
        data (DataFrame): بيانات المرضى والصور
    """
    # استخراج أبعاد الصور
    widths = []
    heights = []
    for _, row in data.iterrows():
        width = row['OriginalImage[Width']
        height = row['Height]']
        widths.append(width)
        heights.append(height)
    
    # رسم توزيع أبعاد الصور
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(widths, bins=20, kde=True)
    plt.title('Image Width Distribution')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    sns.histplot(heights, bins=20, kde=True)
    plt.title('Image Height Distribution')
    plt.xlabel('Height (pixels)')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('image_dimensions.png')
    
    # استخراج تباعد البكسل
    pixel_spacing_x = []
    pixel_spacing_y = []
    for _, row in data.iterrows():
        spacing_x = row['OriginalImagePixelSpacing[x']
        spacing_y = row['y]']
        pixel_spacing_x.append(spacing_x)
        pixel_spacing_y.append(spacing_y)
    
    # رسم توزيع تباعد البكسل
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(pixel_spacing_x, bins=20, kde=True)
    plt.title('Pixel Spacing X Distribution')
    plt.xlabel('Spacing X (mm)')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    sns.histplot(pixel_spacing_y, bins=20, kde=True)
    plt.title('Pixel Spacing Y Distribution')
    plt.xlabel('Spacing Y (mm)')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('pixel_spacing.png')
    
    return {
        'width_stats': {
            'mean': np.mean(widths),
            'median': np.median(widths),
            'min': min(widths),
            'max': max(widths)
        },
        'height_stats': {
            'mean': np.mean(heights),
            'median': np.median(heights),
            'min': min(heights),
            'max': max(heights)
        },
        'spacing_x_stats': {
            'mean': np.mean(pixel_spacing_x),
            'median': np.median(pixel_spacing_x),
            'min': min(pixel_spacing_x),
            'max': max(pixel_spacing_x)
        },
        'spacing_y_stats': {
            'mean': np.mean(pixel_spacing_y),
            'median': np.median(pixel_spacing_y),
            'min': min(pixel_spacing_y),
            'max': max(pixel_spacing_y)
        }
    }

def main():
    """الدالة الرئيسية"""
    # إنشاء مجلد للمخرجات
    os.makedirs('analysis_outputs', exist_ok=True)
    
    # تحميل البيانات
    data_path = "./data/Data_Entry_2017.csv"
    data = load_data(data_path)
    
    # تحليل البيانات الديموغرافية
    print("Analyzing demographic information...")
    demographic_stats = analyze_demographics(data)
    
    # تحليل أنماط الأمراض
    print("Analyzing disease patterns...")
    disease_stats = analyze_disease_patterns(data)
    
    # تحليل تاريخ المرضى
    print("Analyzing patient history...")
    patient_stats = analyze_patient_history(data)
    
    # تحليل خصائص الصور
    print("Analyzing image properties...")
    image_stats = analyze_image_properties(data)
    
    # طباعة ملخص النتائج
    print("\nAnalysis Summary:")
    print("-" * 50)
    
    print("\nDemographic Statistics:")
    print(f"Age: Mean = {demographic_stats['age_stats']['mean']:.1f}, Median = {demographic_stats['age_stats']['median']:.1f}, Range = {demographic_stats['age_stats']['min']}-{demographic_stats['age_stats']['max']}")
    print(f"Gender: Male = {demographic_stats['gender_stats']['M']}, Female = {demographic_stats['gender_stats']['F']}")
    
    print("\nDisease Statistics:")
    print("Top 5 Most Common Diseases:")
    for disease, count in disease_stats['disease_counts'].most_common(6):
        if disease != 'No Finding':
            print(f"- {disease}: {count}")
    
    print("\nPatient Statistics:")
    print(f"Images per Patient: Mean = {patient_stats['images_per_patient_stats']['mean']:.1f}, Median = {patient_stats['images_per_patient_stats']['median']:.1f}, Range = {patient_stats['images_per_patient_stats']['min']}-{patient_stats['images_per_patient_stats']['max']}")
    
    print("\nImage Statistics:")
    print(f"Width: Mean = {image_stats['width_stats']['mean']:.1f}, Median = {image_stats['width_stats']['median']:.1f}, Range = {image_stats['width_stats']['min']}-{image_stats['width_stats']['max']}")
    print(f"Height: Mean = {image_stats['height_stats']['mean']:.1f}, Median = {image_stats['height_stats']['median']:.1f}, Range = {image_stats['height_stats']['min']}-{image_stats['height_stats']['max']}")
    
    print("\nAnalysis completed. Results saved to 'analysis_outputs' directory.")

if __name__ == "__main__":
    main() 