import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import argparse
from collections import defaultdict

# تعريف أسماء الأمراض بالعربية والإنجليزية
DISEASE_NAMES = {
    'Atelectasis': 'انخماص الرئة',
    'Cardiomegaly': 'تضخم القلب',
    'Effusion': 'انصباب',
    'Infiltration': 'ارتشاح',
    'Mass': 'كتلة',
    'Nodule': 'عقيدة',
    'Pneumonia': 'التهاب رئوي',
    'Pneumothorax': 'استرواح الصدر',
    'Consolidation': 'تصلب',
    'Edema': 'وذمة',
    'Emphysema': 'انتفاخ الرئة',
    'Fibrosis': 'تليف',
    'Pleural_Thickening': 'تثخن الجنبة',
    'Hernia': 'فتق',
    'No Finding': 'لا يوجد مرض'
}

def load_predictions(predictions_path):
    """
    تحميل التنبؤات من ملف CSV
    """
    try:
        df = pd.read_csv(predictions_path)
        print(f"تم تحميل {len(df)} صف من البيانات.")
        
        # طباعة قائمة الأعمدة للتحقق
        print(f"الأعمدة المتاحة: {df.columns.tolist()}")
        
        # تحقق من تنسيق البيانات
        prob_columns = [col for col in df.columns if col.startswith('all_probabilities.')]
        classes = [col.replace('all_probabilities.', '') for col in prob_columns]
        print(f"تم العثور على {len(classes)} فئة: {classes}")
        
        return df, classes
    except Exception as e:
        print(f"خطأ في تحميل ملف التنبؤات: {str(e)}")
        sys.exit(1)

def calculate_auroc(df, classes, threshold=0.5):
    """
    حساب قيم AUROC لكل فئة
    """
    results = {}
    class_counts = {}
    
    for cls in classes:
        probabilities = df[f'all_probabilities.{cls}'].values
        
        # استخدم العتبة لتحويل الاحتمالات إلى توقعات ثنائية (0 أو 1)
        true_labels = (probabilities >= threshold).astype(int)
        
        # حساب عدد العينات الإيجابية والسلبية
        positive_count = sum(true_labels)
        negative_count = len(true_labels) - positive_count
        class_counts[cls] = {'positive': positive_count, 'negative': negative_count}
        
        # تحقق من أن هناك عينات إيجابية وسلبية لحساب AUROC
        if positive_count > 0 and negative_count > 0:
            # حساب منحنى ROC
            fpr, tpr, _ = roc_curve(true_labels, probabilities)
            auroc_value = auc(fpr, tpr)
            results[cls] = auroc_value
        else:
            print(f"تحذير: لا يمكن حساب AUROC للفئة {cls} لأن جميع العينات إما إيجابية أو سلبية")
            results[cls] = 0.0
    
    # حساب متوسط AUROC
    average_auroc = np.mean(list(results.values()))
    
    # ترتيب الفئات من الأفضل إلى الأسوأ
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    best_class, best_auroc = sorted_results[0]
    worst_class, worst_auroc = sorted_results[-1]
    
    print(f"متوسط AUROC: {average_auroc:.4f}")
    print(f"أفضل فئة: {best_class} ({DISEASE_NAMES.get(best_class, best_class)}) مع AUROC = {best_auroc:.4f}")
    print(f"أسوأ فئة: {worst_class} ({DISEASE_NAMES.get(worst_class, worst_class)}) مع AUROC = {worst_auroc:.4f}")
    
    # طباعة تفاصيل كل فئة
    print("\nتفاصيل AUROC لكل فئة:")
    for cls, auroc_value in sorted_results:
        pos = class_counts[cls]['positive']
        neg = class_counts[cls]['negative']
        print(f"{cls} ({DISEASE_NAMES.get(cls, cls)}): AUROC = {auroc_value:.4f}, العينات الإيجابية: {pos}, السلبية: {neg}")
    
    return results, average_auroc, class_counts

def analyze_no_finding(df, threshold=0.4):
    """
    تحليل خاص لفئة 'No Finding' (لا يوجد مرض)
    """
    if 'all_probabilities.No Finding' not in df.columns:
        print("تحذير: فئة 'No Finding' غير موجودة في البيانات")
        return {}
    
    no_finding_prob = df['all_probabilities.No Finding'].values
    
    # عدد الحالات المصنفة كـ 'No Finding'
    no_finding_cases = sum(no_finding_prob >= threshold)
    total_cases = len(df)
    
    # الاحتمال المتوسط لـ 'No Finding'
    mean_probability = np.mean(no_finding_prob)
    
    # توزيع الاحتمالات
    probability_bins = np.histogram(no_finding_prob, bins=10, range=(0, 1))[0]
    
    # تداخل مع الأمراض الأخرى
    disease_overlaps = {}
    other_classes = [col.replace('all_probabilities.', '') for col in df.columns 
                    if col.startswith('all_probabilities.') and col != 'all_probabilities.No Finding']
    
    for disease in other_classes:
        disease_prob = df[f'all_probabilities.{disease}'].values
        disease_cases = sum(disease_prob >= threshold)
        
        # حالات مصنفة على أنها مرض معين و No Finding معًا
        overlap_cases = sum((disease_prob >= threshold) & (no_finding_prob >= threshold))
        
        # نسبة التداخل
        if disease_cases > 0:
            overlap_ratio = overlap_cases / disease_cases
        else:
            overlap_ratio = 0
        
        disease_overlaps[disease] = {
            'overlap_count': overlap_cases,
            'overlap_ratio': overlap_ratio,
            'disease_count': disease_cases
        }
    
    results = {
        'no_finding_count': no_finding_cases,
        'no_finding_ratio': no_finding_cases / total_cases,
        'mean_probability': mean_probability,
        'probability_distribution': probability_bins.tolist(),
        'disease_overlaps': disease_overlaps
    }
    
    # طباعة النتائج
    print("\nتحليل فئة 'No Finding':")
    print(f"عدد الحالات المصنفة كـ 'No Finding': {no_finding_cases} ({results['no_finding_ratio']:.2%} من الإجمالي)")
    print(f"متوسط احتمالية 'No Finding': {mean_probability:.4f}")
    
    print("\nالتداخل مع الأمراض الأخرى:")
    sorted_overlaps = sorted(disease_overlaps.items(), key=lambda x: x[1]['overlap_ratio'], reverse=True)
    for disease, stats in sorted_overlaps:
        print(f"{disease} ({DISEASE_NAMES.get(disease, disease)}): {stats['overlap_count']} حالة ({stats['overlap_ratio']:.2%} من حالات المرض)")
    
    return results

def plot_class_distribution(df, classes, output_path=None):
    """
    رسم توزيع الفئات
    """
    class_counts = []
    for cls in classes:
        count = sum(df[f'all_probabilities.{cls}'] >= 0.5)
        class_counts.append({'class': cls, 'count': count})
    
    distribution_df = pd.DataFrame(class_counts)
    distribution_df = distribution_df.sort_values('count', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='count', y='class', data=distribution_df)
    plt.title('توزيع الفئات المتوقعة', fontsize=14)
    plt.xlabel('عدد الحالات', fontsize=12)
    plt.ylabel('الفئة', fontsize=12)
    
    # إضافة الأسماء العربية
    labels = [f"{cls} ({DISEASE_NAMES.get(cls, cls)})" for cls in distribution_df['class']]
    plt.yticks(range(len(labels)), labels)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"تم حفظ توزيع الفئات في {output_path}")
    else:
        plt.show()
    
    plt.close()

def plot_auroc_comparison(auroc_results, output_path=None):
    """
    رسم مقارنة AUROC بين الفئات
    """
    sorted_results = sorted(auroc_results.items(), key=lambda x: x[1], reverse=True)
    classes = [item[0] for item in sorted_results]
    auroc_values = [item[1] for item in sorted_results]
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(classes, auroc_values, color='skyblue')
    
    # إضافة خط يمثل مستوى 0.5 (مستوى التخمين العشوائي)
    plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='مستوى التخمين العشوائي (0.5)')
    
    # إضافة القيم على الرسم
    for i, (cls, value) in enumerate(zip(classes, auroc_values)):
        plt.text(max(value + 0.01, 0.02), i, f'{value:.4f}', va='center')
    
    # إضافة الأسماء العربية
    labels = [f"{cls} ({DISEASE_NAMES.get(cls, cls)})" for cls in classes]
    plt.yticks(range(len(labels)), labels)
    
    plt.title('مقارنة قيم AUROC بين الفئات', fontsize=14)
    plt.xlabel('AUROC', fontsize=12)
    plt.ylabel('الفئة', fontsize=12)
    plt.xlim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"تم حفظ مقارنة AUROC في {output_path}")
    else:
        plt.show()
    
    plt.close()

def plot_confidence_distribution(df, classes, output_path=None):
    """
    رسم توزيع الثقة (الاحتمالات) لكل فئة
    """
    plt.figure(figsize=(14, 10))
    
    # إنشاء subplots لكل فئة
    ncols = 3
    nrows = (len(classes) + ncols - 1) // ncols
    
    for i, cls in enumerate(classes):
        plt.subplot(nrows, ncols, i + 1)
        
        probabilities = df[f'all_probabilities.{cls}'].values
        sns.histplot(probabilities, bins=20, kde=True)
        
        plt.title(f"{cls}\n({DISEASE_NAMES.get(cls, cls)})", fontsize=10)
        plt.xlabel('احتمالية التوقع', fontsize=8)
        plt.ylabel('العدد', fontsize=8)
        plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"تم حفظ توزيع الثقة في {output_path}")
    else:
        plt.show()
    
    plt.close()

def calculate_confusion_with_no_finding(df, threshold=0.5):
    """
    حساب الارتباك بين "No Finding" والأمراض الأخرى
    """
    if 'all_probabilities.No Finding' not in df.columns:
        print("تحذير: فئة 'No Finding' غير موجودة في البيانات")
        return {}
    
    no_finding_prob = df['all_probabilities.No Finding'].values
    no_finding_pred = no_finding_prob >= threshold
    
    other_classes = [col.replace('all_probabilities.', '') for col in df.columns 
                    if col.startswith('all_probabilities.') and col != 'all_probabilities.No Finding']
    
    confusion_metrics = {}
    
    for disease in other_classes:
        disease_prob = df[f'all_probabilities.{disease}'].values
        disease_pred = disease_prob >= threshold
        
        # حالات التناقض (No Finding وهناك مرض في نفس الوقت)
        contradiction_cases = sum(no_finding_pred & disease_pred)
        
        # نسبة التناقض من إجمالي حالات المرض
        disease_cases = sum(disease_pred)
        if disease_cases > 0:
            contradiction_ratio = contradiction_cases / disease_cases
        else:
            contradiction_ratio = 0
        
        confusion_metrics[disease] = {
            'contradiction_count': contradiction_cases,
            'contradiction_ratio': contradiction_ratio,
            'disease_count': disease_cases
        }
    
    # طباعة نتائج الارتباك
    print("\nتحليل التناقض بين 'No Finding' والأمراض:")
    sorted_confusion = sorted(confusion_metrics.items(), key=lambda x: x[1]['contradiction_ratio'], reverse=True)
    for disease, stats in sorted_confusion:
        print(f"{disease} ({DISEASE_NAMES.get(disease, disease)}): {stats['contradiction_count']} حالة متناقضة ({stats['contradiction_ratio']:.2%} من حالات المرض)")
    
    return confusion_metrics

def main():
    parser = argparse.ArgumentParser(description='تحليل نتائج AUROC وأداء النموذج')
    parser.add_argument('--predictions_path', type=str, required=True, help='مسار ملف التنبؤات CSV')
    parser.add_argument('--output_dir', type=str, default='analysis_results', help='مجلد حفظ نتائج التحليل')
    parser.add_argument('--threshold', type=float, default=0.4, help='عتبة مستخدمة في تصنيف No Finding')
    args = parser.parse_args()
    
    # إنشاء مجلد النتائج إذا لم يكن موجودًا
    os.makedirs(args.output_dir, exist_ok=True)
    
    # تحميل التنبؤات
    df, classes = load_predictions(args.predictions_path)
    
    # حساب AUROC
    auroc_results, average_auroc, class_counts = calculate_auroc(df, classes)
    
    # تحليل No Finding
    no_finding_analysis = analyze_no_finding(df, args.threshold)
    
    # حساب الارتباك بين No Finding والأمراض
    confusion_metrics = calculate_confusion_with_no_finding(df, args.threshold)
    
    # إنشاء الرسومات البيانية
    plot_class_distribution(df, classes, os.path.join(args.output_dir, 'class_distribution.png'))
    plot_auroc_comparison(auroc_results, os.path.join(args.output_dir, 'auroc_comparison.png'))
    plot_confidence_distribution(df, classes, os.path.join(args.output_dir, 'confidence_distribution.png'))
    
    # حفظ النتائج في ملف نصي
    with open(os.path.join(args.output_dir, 'analysis_results.txt'), 'w', encoding='utf-8') as f:
        f.write(f"تحليل أداء النموذج\n{'='*50}\n\n")
        f.write(f"متوسط AUROC: {average_auroc:.4f}\n\n")
        
        f.write("تفاصيل AUROC لكل فئة:\n")
        sorted_results = sorted(auroc_results.items(), key=lambda x: x[1], reverse=True)
        for cls, auroc_value in sorted_results:
            pos = class_counts[cls]['positive']
            neg = class_counts[cls]['negative']
            f.write(f"{cls} ({DISEASE_NAMES.get(cls, cls)}): AUROC = {auroc_value:.4f}, العينات الإيجابية: {pos}, السلبية: {neg}\n")
        
        if 'all_probabilities.No Finding' in df.columns:
            f.write(f"\nتحليل فئة 'No Finding':\n")
            f.write(f"عدد الحالات المصنفة كـ 'No Finding': {no_finding_analysis['no_finding_count']} ({no_finding_analysis['no_finding_ratio']:.2%} من الإجمالي)\n")
            f.write(f"متوسط احتمالية 'No Finding': {no_finding_analysis['mean_probability']:.4f}\n\n")
            
            f.write("التداخل مع الأمراض الأخرى:\n")
            sorted_overlaps = sorted(no_finding_analysis['disease_overlaps'].items(), key=lambda x: x[1]['overlap_ratio'], reverse=True)
            for disease, stats in sorted_overlaps:
                f.write(f"{disease} ({DISEASE_NAMES.get(disease, disease)}): {stats['overlap_count']} حالة ({stats['overlap_ratio']:.2%} من حالات المرض)\n")
            
            f.write("\nتحليل التناقض بين 'No Finding' والأمراض:\n")
            sorted_confusion = sorted(confusion_metrics.items(), key=lambda x: x[1]['contradiction_ratio'], reverse=True)
            for disease, stats in sorted_confusion:
                f.write(f"{disease} ({DISEASE_NAMES.get(disease, disease)}): {stats['contradiction_count']} حالة متناقضة ({stats['contradiction_ratio']:.2%} من حالات المرض)\n")
    
    print(f"\nتم حفظ نتائج التحليل الكاملة في {os.path.join(args.output_dir, 'analysis_results.txt')}")
    
if __name__ == "__main__":
    main() 