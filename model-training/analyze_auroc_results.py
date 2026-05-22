import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import argparse

# تعريف الفئات
DISEASE_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 
    'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

def load_metrics(metrics_file):
    """تحميل ملف المقاييس من التدريب وتحليله"""
    if not os.path.exists(metrics_file):
        print(f"خطأ: ملف المقاييس {metrics_file} غير موجود")
        return None
    
    try:
        metrics_df = pd.read_csv(metrics_file)
        return metrics_df
    except Exception as e:
        print(f"خطأ في قراءة ملف المقاييس: {str(e)}")
        return None

def plot_metrics_history(metrics_df, output_dir):
    """رسم تاريخ المقاييس أثناء التدريب"""
    if metrics_df is None or metrics_df.empty:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # رسم مقاييس AUROC
    plt.figure(figsize=(12, 8))
    if 'val_auroc_macro' in metrics_df.columns:
        plt.plot(metrics_df['epoch'], metrics_df['val_auroc_macro'], 'g-', label='AUROC (Macro)')
    if 'val_auroc_weighted' in metrics_df.columns:
        plt.plot(metrics_df['epoch'], metrics_df['val_auroc_weighted'], 'r-', label='AUROC (Weighted)')
    if 'val_auroc_micro' in metrics_df.columns:
        plt.plot(metrics_df['epoch'], metrics_df['val_auroc_micro'], 'b-', label='AUROC (Micro)')
    if 'train_auroc' in metrics_df.columns:
        plt.plot(metrics_df['epoch'], metrics_df['train_auroc'], 'c--', label='Train AUROC')
    
    plt.xlabel('Epoch')
    plt.ylabel('AUROC')
    plt.title('تطور AUROC أثناء التدريب')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'auroc_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # رسم الدقة والخسارة
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(metrics_df['epoch'], metrics_df['train_loss'], color=color, linestyle='--', label='Train Loss')
    if 'val_loss' in metrics_df.columns:
        ax1.plot(metrics_df['epoch'], metrics_df['val_loss'], color='darkred', label='Val Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy (%)', color=color)
    ax2.plot(metrics_df['epoch'], metrics_df['train_acc'], color=color, linestyle='--', label='Train Acc')
    if 'val_acc' in metrics_df.columns:
        ax2.plot(metrics_df['epoch'], metrics_df['val_acc'], color='darkblue', label='Val Acc')
    ax2.tick_params(axis='y', labelcolor=color)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title('تطور الدقة والخسارة أثناء التدريب')
    plt.savefig(os.path.join(output_dir, 'accuracy_loss_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

def load_predictions(predictions_file):
    """تحميل ملف التنبؤات لتحليله"""
    if not os.path.exists(predictions_file):
        print(f"خطأ: ملف التنبؤات {predictions_file} غير موجود")
        return None
    
    try:
        preds_df = pd.read_csv(predictions_file)
        return preds_df
    except Exception as e:
        print(f"خطأ في قراءة ملف التنبؤات: {str(e)}")
        return None

def analyze_predictions(preds_df, output_dir, threshold=0.5):
    """تحليل التنبؤات وحساب AUROC لكل فئة"""
    if preds_df is None or preds_df.empty:
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    # التحقق من وجود العمود المرجعي للفئات الحقيقية
    if not any(col.startswith('true_') for col in preds_df.columns):
        print("تحذير: لا توجد أعمدة للفئات الحقيقية في ملف التنبؤات. لا يمكن حساب AUROC.")
        # نقوم بتحليل التوزيعات فقط
        analyze_prediction_distributions(preds_df, output_dir, threshold)
        return None
    
    # حساب AUROC لكل فئة
    auroc_results = {}
    for cls in DISEASE_CLASSES:
        if f'true_{cls}' in preds_df.columns and cls in preds_df.columns:
            y_true = preds_df[f'true_{cls}'].values
            y_pred = preds_df[cls].values
            
            # نتجاهل الفئات التي ليس لها قيم موجبة أو سالبة
            if len(np.unique(y_true)) < 2:
                auroc_results[cls] = {'auroc': 0.5, 'support': sum(y_true)}
                continue
            
            try:
                fpr, tpr, _ = roc_curve(y_true, y_pred)
                auroc = auc(fpr, tpr)
                auroc_results[cls] = {
                    'auroc': auroc,
                    'support': sum(y_true),
                    'fpr': fpr,
                    'tpr': tpr
                }
                
                # رسم منحنى ROC لكل فئة
                plt.figure(figsize=(8, 8))
                plt.plot(fpr, tpr, 'b-', label=f'AUROC = {auroc:.4f}')
                plt.plot([0, 1], [0, 1], 'r--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'منحنى ROC لفئة {cls} (دعم = {sum(y_true)})')
                plt.legend(loc="lower right")
                plt.savefig(os.path.join(output_dir, f'roc_{cls}.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                # حساب مصفوفة الالتباس وطباعتها
                conf_matrix = confusion_matrix(y_true, y_pred > threshold)
                plt.figure(figsize=(8, 6))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['سلبي', 'إيجابي'],
                            yticklabels=['سلبي', 'إيجابي'])
                plt.title(f'مصفوفة الالتباس لفئة {cls} (عتبة = {threshold})')
                plt.ylabel('القيمة الحقيقية')
                plt.xlabel('التنبؤ')
                plt.savefig(os.path.join(output_dir, f'confusion_{cls}.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"خطأ في حساب AUROC لفئة {cls}: {str(e)}")
                auroc_results[cls] = {'auroc': 0, 'support': 0}
    
    if not auroc_results:
        return None
    
    # ترتيب النتائج بناءً على AUROC
    sorted_results = sorted(auroc_results.items(), key=lambda x: x[1]['auroc'])
    
    # طباعة النتائج وحفظها في ملف
    results_text = "*** تحليل AUROC لكل فئة ***\n\n"
    
    # الفئات ذات الأداء الأسوأ
    results_text += "الفئات ذات الأداء الأسوأ:\n"
    for cls, data in sorted_results[:3]:
        results_text += f"{cls}: {data['auroc']:.4f} (دعم = {data['support']})\n"
    
    results_text += "\nالفئات ذات الأداء الأفضل:\n"
    for cls, data in sorted_results[-3:]:
        results_text += f"{cls}: {data['auroc']:.4f} (دعم = {data['support']})\n"
    
    # حساب متوسط AUROC
    avg_auroc = np.mean([data['auroc'] for _, data in auroc_results.items()])
    weighted_avg_auroc = np.sum([data['auroc'] * data['support'] for _, data in auroc_results.items()]) / np.sum([data['support'] for _, data in auroc_results.items()])
    
    results_text += f"\nمتوسط AUROC (Macro): {avg_auroc:.4f}\n"
    results_text += f"متوسط AUROC (Weighted): {weighted_avg_auroc:.4f}\n"
    
    # حفظ النتائج في ملف
    with open(os.path.join(output_dir, 'auroc_analysis.txt'), 'w', encoding='utf-8') as f:
        f.write(results_text)
    
    print(results_text)
    
    # رسم مخطط بياني للمقارنة بين الفئات
    plt.figure(figsize=(14, 8))
    cls_names = [cls for cls, _ in sorted_results]
    auroc_values = [data['auroc'] for _, data in sorted_results]
    support_values = [data['support'] for _, data in sorted_results]
    
    # تحسين المظهر للأرقام العربية
    plt.rcParams['axes.unicode_minus'] = False
    
    bars = plt.bar(cls_names, auroc_values, color='skyblue')
    for i, (bar, support) in enumerate(zip(bars, support_values)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{support}', ha='center', va='bottom', rotation=0, fontsize=9)
    
    plt.axhline(y=0.5, color='r', linestyle='--', label='الحد الأدنى (0.5)')
    plt.axhline(y=avg_auroc, color='g', linestyle='--', label=f'المتوسط ({avg_auroc:.3f})')
    plt.axhline(y=weighted_avg_auroc, color='purple', linestyle='--', label=f'المتوسط المرجح ({weighted_avg_auroc:.3f})')
    
    plt.xlabel('الفئة')
    plt.ylabel('AUROC')
    plt.title('مقارنة AUROC بين الفئات المختلفة')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'auroc_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return auroc_results

def analyze_prediction_distributions(preds_df, output_dir, threshold=0.5):
    """تحليل توزيع التنبؤات لكل فئة"""
    if preds_df is None or preds_df.empty:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # تحليل توزيع الاحتمالات لكل فئة
    plt.figure(figsize=(14, 10))
    cols_to_plot = [col for col in preds_df.columns if col in DISEASE_CLASSES]
    
    for i, col in enumerate(cols_to_plot):
        plt.subplot(4, 4, i+1)
        sns.histplot(preds_df[col], bins=30, kde=True)
        plt.axvline(x=threshold, color='r', linestyle='--')
        plt.title(col)
        plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'prediction_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # حساب النسب المئوية للتنبؤات الإيجابية لكل فئة
    positive_counts = {}
    for col in cols_to_plot:
        positive_counts[col] = (preds_df[col] >= threshold).mean() * 100
    
    # رسم بياني للنسب المئوية
    plt.figure(figsize=(12, 8))
    plt.bar(positive_counts.keys(), positive_counts.values(), color='skyblue')
    plt.ylabel('النسبة المئوية للتنبؤات الإيجابية (%)')
    plt.title('النسبة المئوية للتنبؤات الإيجابية لكل فئة')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'positive_prediction_rates.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # حساب متوسط الاحتمالات لكل فئة
    mean_probs = {col: preds_df[col].mean() for col in cols_to_plot}
    
    # رسم بياني للمتوسطات
    plt.figure(figsize=(12, 8))
    plt.bar(mean_probs.keys(), mean_probs.values(), color='lightgreen')
    plt.ylabel('متوسط الاحتمالات')
    plt.title('متوسط احتمالات التنبؤ لكل فئة')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mean_prediction_probabilities.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='تحليل نتائج AUROC من نموذج تصنيف الأشعة السينية')
    parser.add_argument('--metrics_file', type=str, default=None, help='مسار ملف المقاييس')
    parser.add_argument('--predictions_file', type=str, default=None, help='مسار ملف التنبؤات')
    parser.add_argument('--model_dir', type=str, default=None, help='مسار مجلد النموذج (يتم البحث تلقائيًا عن الملفات)')
    parser.add_argument('--output_dir', type=str, default='auroc_analysis', help='مجلد حفظ نتائج التحليل')
    parser.add_argument('--threshold', type=float, default=0.5, help='عتبة التصنيف')
    
    args = parser.parse_args()
    
    # إنشاء مجلد الخرج إذا لم يكن موجودًا
    os.makedirs(args.output_dir, exist_ok=True)
    
    # إذا تم تحديد مجلد النموذج، نبحث تلقائيًا عن الملفات
    if args.model_dir:
        if not args.metrics_file:
            metrics_path = os.path.join(args.model_dir, 'training_metrics.csv')
            if os.path.exists(metrics_path):
                args.metrics_file = metrics_path
        
        if not args.predictions_file:
            # البحث عن ملفات التنبؤات
            pred_candidates = [
                os.path.join(args.model_dir, 'predictions.csv'),
                os.path.join(args.model_dir, '../predictions_output/predictions.csv')
            ]
            for path in pred_candidates:
                if os.path.exists(path):
                    args.predictions_file = path
                    break
    
    # تحليل المقاييس أثناء التدريب
    if args.metrics_file:
        print(f"تحليل مقاييس التدريب من: {args.metrics_file}")
        metrics_df = load_metrics(args.metrics_file)
        if metrics_df is not None:
            plot_metrics_history(metrics_df, args.output_dir)
    else:
        print("لم يتم تحديد ملف المقاييس. تخطي تحليل مقاييس التدريب.")
    
    # تحليل التنبؤات
    if args.predictions_file:
        print(f"تحليل التنبؤات من: {args.predictions_file}")
        preds_df = load_predictions(args.predictions_file)
        if preds_df is not None:
            analyze_predictions(preds_df, args.output_dir, args.threshold)
    else:
        print("لم يتم تحديد ملف التنبؤات. تخطي تحليل التنبؤات.")
    
    print(f"اكتمل التحليل. تم حفظ النتائج في: {args.output_dir}")

if __name__ == "__main__":
    main() 