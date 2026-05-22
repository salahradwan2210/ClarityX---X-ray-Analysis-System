import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
import seaborn as sns
from pathlib import Path

# List of disease classes in our dataset
DISEASE_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
    'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze model AUROC performance')
    parser.add_argument('--predictions_path', type=str, required=True, 
                       help='Path to CSV file containing predictions and true labels')
    parser.add_argument('--output_dir', type=str, default='auroc_analysis',
                       help='Directory to save analysis results')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Decision threshold for binary classification')
    return parser.parse_args()

def create_output_dir(dir_path):
    """Create output directory if it doesn't exist"""
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def load_predictions(predictions_path):
    """Load predictions from CSV file"""
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    
    df = pd.read_csv(predictions_path)
    return df

def calculate_auroc_metrics(df, threshold=0.5, verbose=True):
    """Calculate AUROC and related metrics for each class"""
    class_metrics = {}
    all_true = []
    all_pred = []
    
    # Identify prediction and ground truth columns
    pred_cols = [col for col in df.columns if col.startswith('pred_prob_')]
    true_cols = [col for col in df.columns if col.startswith('true_label_')]
    
    if not pred_cols or not true_cols:
        raise ValueError("Could not identify prediction or ground truth columns in CSV")
    
    class_names = [col.replace('pred_prob_', '') for col in pred_cols]
    
    # Calculate metrics for each class
    for i, class_name in enumerate(class_names):
        y_true = df[f'true_label_{class_name}'].values
        y_pred = df[f'pred_prob_{class_name}'].values
        
        metrics = {}
        
        # Calculate AUROC
        if len(np.unique(y_true)) > 1:  # Only if we have both positive and negative samples
            metrics['auroc'] = roc_auc_score(y_true, y_pred)
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            metrics['fpr'] = fpr
            metrics['tpr'] = tpr
            
            # Count positive and negative samples
            metrics['positive_count'] = np.sum(y_true == 1)
            metrics['negative_count'] = np.sum(y_true == 0)
            metrics['total_count'] = len(y_true)
            metrics['positive_ratio'] = metrics['positive_count'] / metrics['total_count']
            
            # Calculate binary classification metrics at threshold
            y_pred_binary = (y_pred >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()
            
            metrics['true_positives'] = tp
            metrics['false_positives'] = fp
            metrics['true_negatives'] = tn
            metrics['false_negatives'] = fn
            
            metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
            
            # Store for calculating micro-average
            all_true.extend(y_true)
            all_pred.extend(y_pred)
            
            # Add to class_metrics
            class_metrics[class_name] = metrics
            
            if verbose:
                print(f"{class_name}:")
                print(f"  AUROC: {metrics['auroc']:.4f}")
                print(f"  Positive samples: {metrics['positive_count']} ({metrics['positive_ratio']*100:.2f}%)")
                print(f"  F1 Score: {metrics['f1']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
                print()
        else:
            if verbose:
                print(f"{class_name}: Not enough unique samples for AUROC calculation")
    
    # Calculate macro and micro averages
    auroc_values = [m['auroc'] for m in class_metrics.values() if 'auroc' in m]
    metrics_summary = {
        'macro_auroc': np.mean(auroc_values),
        'class_count': len(auroc_values),
        'micro_auroc': roc_auc_score(np.array(all_true), np.array(all_pred)) if len(all_true) > 0 else 0,
        'per_class': class_metrics
    }
    
    if verbose:
        print("=" * 40)
        print(f"Macro-average AUROC: {metrics_summary['macro_auroc']:.4f}")
        print(f"Micro-average AUROC: {metrics_summary['micro_auroc']:.4f}")
    
    return metrics_summary

def find_problem_classes(metrics, threshold=0.55):
    """Identify classes with AUROC below a threshold"""
    problem_classes = {}
    
    for class_name, metrics in metrics['per_class'].items():
        if 'auroc' in metrics and metrics['auroc'] < threshold:
            problem_classes[class_name] = {
                'auroc': metrics['auroc'],
                'positive_count': metrics['positive_count'],
                'positive_ratio': metrics['positive_ratio'],
                'recall': metrics['recall']
            }
    
    # Sort by AUROC (worst first)
    problem_classes = {k: v for k, v in sorted(problem_classes.items(), key=lambda item: item[1]['auroc'])}
    
    return problem_classes

def plot_auroc_bars(metrics, output_path):
    """Plot AUROC values for each class as a bar chart"""
    # Extract class names and AUROC values
    class_names = []
    auroc_values = []
    
    for class_name, class_metrics in metrics['per_class'].items():
        if 'auroc' in class_metrics:
            class_names.append(class_name)
            auroc_values.append(class_metrics['auroc'])
    
    # Sort by AUROC
    sorted_indices = np.argsort(auroc_values)
    sorted_classes = [class_names[i] for i in sorted_indices]
    sorted_aurocs = [auroc_values[i] for i in sorted_indices]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    bars = plt.barh(sorted_classes, sorted_aurocs, color='skyblue')
    
    # Color bars red if below 0.5 (worse than random)
    for i, v in enumerate(sorted_aurocs):
        if v < 0.5:
            bars[i].set_color('red')
        elif v < 0.7:
            bars[i].set_color('orange')
        else:
            bars[i].set_color('green')
    
    # Add a vertical line at 0.5 (random guessing)
    plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Random Guessing')
    
    # Add a vertical line at the macro average
    plt.axvline(x=metrics['macro_auroc'], color='blue', linestyle='--', alpha=0.7, 
                label=f'Macro Average: {metrics["macro_auroc"]*100:.2f}%')
    
    plt.xlabel('AUROC')
    plt.title('AUROC by Disease Class')
    plt.xlim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Add value annotations
    for i, v in enumerate(sorted_aurocs):
        plt.text(v + 0.01, i, f'{v:.4f}', va='center')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_roc_curves(metrics, output_dir):
    """Plot ROC curves for each class"""
    plt.figure(figsize=(12, 10))
    
    for class_name, class_metrics in metrics['per_class'].items():
        if 'fpr' in class_metrics and 'tpr' in class_metrics:
            fpr = class_metrics['fpr']
            tpr = class_metrics['tpr']
            auroc = class_metrics['auroc']
            
            plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {auroc:.4f})')
    
    # Add the random guessing line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guessing')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves by Disease Class')
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
    plt.close()

def analyze_difficult_classes(metrics, output_dir):
    """Perform detailed analysis of difficult classes"""
    problem_classes = find_problem_classes(metrics)
    
    if not problem_classes:
        print("No problem classes identified!")
        return
    
    # Create a detailed report
    report_path = os.path.join(output_dir, 'problem_classes_report.txt')
    with open(report_path, 'w') as f:
        f.write("PROBLEM CLASSES ANALYSIS\n")
        f.write("=======================\n\n")
        
        f.write(f"Identified {len(problem_classes)} classes with AUROC < 0.55\n\n")
        
        for class_name, stats in problem_classes.items():
            f.write(f"{class_name}:\n")
            f.write(f"  AUROC: {stats['auroc']:.4f}\n")
            f.write(f"  Positive samples: {stats['positive_count']} ({stats['positive_ratio']*100:.2f}%)\n")
            f.write(f"  Recall: {stats['recall']:.4f}\n")
            
            # Add recommendations
            f.write("\n  Recommendations:\n")
            
            if stats['positive_ratio'] < 0.05:
                f.write("  - Class imbalance issue: Consider oversampling or data augmentation\n")
                f.write(f"  - Increase weight for this class by at least {min(5, 0.1/stats['positive_ratio']):.1f}x\n")
            
            if stats['auroc'] < 0.5:
                f.write("  - Performance worse than random: Consider special focus techniques\n")
                f.write("  - Increase focal loss gamma for this class\n")
                f.write("  - Review training examples for potential data issues\n")
            
            if stats['recall'] < 0.3:
                f.write("  - Very low recall: Model struggles to identify positive cases\n")
                f.write("  - Consider custom loss weighting for false negatives\n")
            
            f.write("\n")
    
    print(f"Problem classes report saved to {report_path}")
    
    # Create a visualization of problem classes
    plt.figure(figsize=(10, 6))
    classes = list(problem_classes.keys())
    aurocs = [stats['auroc'] for stats in problem_classes.values()]
    pos_ratios = [stats['positive_ratio'] * 100 for stats in problem_classes.values()]
    
    # Primary axis: AUROC
    ax1 = plt.gca()
    bars = ax1.barh(classes, aurocs, color='orange', alpha=0.7, label='AUROC')
    ax1.set_xlabel('AUROC')
    ax1.set_xlim(0, 0.6)
    
    # Add AUROC values as text
    for i, v in enumerate(aurocs):
        ax1.text(v + 0.01, i, f'{v:.3f}', va='center')
    
    # Secondary axis: Positive ratio
    ax2 = ax1.twiny()
    ax2.barh(classes, pos_ratios, color='blue', alpha=0.3, label='% Positive')
    ax2.set_xlabel('% Positive Samples')
    
    plt.title('Problem Classes Analysis')
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='orange', alpha=0.7, label='AUROC'),
        Patch(facecolor='blue', alpha=0.3, label='% Positive Samples')
    ]
    ax1.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'problem_classes.png'))
    plt.close()

def generate_improvement_suggestions(metrics, output_path):
    """Generate suggestions for improving model performance"""
    problem_classes = find_problem_classes(metrics)
    
    with open(output_path, 'w') as f:
        f.write("AUROC IMPROVEMENT SUGGESTIONS\n")
        f.write("============================\n\n")
        
        f.write("FOCAL LOSS ADJUSTMENTS\n")
        f.write("---------------------\n")
        f.write("class_specific_gamma = {\n")
        
        for class_name, stats in problem_classes.items():
            # Calculate suggested gamma based on AUROC
            if stats['auroc'] < 0.48:
                suggested_gamma = 4.0
            elif stats['auroc'] < 0.5:
                suggested_gamma = 3.5
            elif stats['auroc'] < 0.53:
                suggested_gamma = 3.0
            else:
                suggested_gamma = 2.5
                
            idx = DISEASE_CLASSES.index(class_name) if class_name in DISEASE_CLASSES else -1
            if idx >= 0:
                f.write(f"    {idx}: {suggested_gamma},  # {class_name} - current AUROC: {stats['auroc']:.4f}\n")
        
        f.write("}\n\n")
        
        f.write("SAMPLING WEIGHT ADJUSTMENTS\n")
        f.write("--------------------------\n")
        f.write("low_auroc_indices = {\n")
        
        for class_name, stats in problem_classes.items():
            # Calculate suggested weight boost based on AUROC and positive ratio
            base_boost = 2.0
            if stats['auroc'] < 0.48:
                base_boost = 4.0
            elif stats['auroc'] < 0.5:
                base_boost = 3.5
            elif stats['auroc'] < 0.53:
                base_boost = 3.0
            
            # Further adjust based on class imbalance
            if stats['positive_ratio'] < 0.01:
                base_boost *= 1.5
            elif stats['positive_ratio'] < 0.05:
                base_boost *= 1.2
                
            idx = DISEASE_CLASSES.index(class_name) if class_name in DISEASE_CLASSES else -1
            if idx >= 0:
                f.write(f"    {idx}: {base_boost},  # {class_name} - current AUROC: {stats['auroc']:.4f}, "
                       f"positive ratio: {stats['positive_ratio']*100:.2f}%\n")
        
        f.write("}\n\n")
        
        # Overall model recommendations
        f.write("OVERALL MODEL RECOMMENDATIONS\n")
        f.write("----------------------------\n")
        
        problem_count = len(problem_classes)
        total_classes = metrics['class_count']
        
        if problem_count > total_classes / 2:
            f.write("- Many classes have low AUROC. Consider fundamental model improvements:\n")
            f.write("  * Increase model capacity (larger variant or custom architecture)\n")
            f.write("  * Increase image resolution to 448 or 512\n")
            f.write("  * Use a slower learning rate (1e-6) with more patience\n")
            f.write("  * Consider curriculum learning approaches\n")
        else:
            f.write("- Focus on class-specific improvements for problematic classes\n")
            f.write("  * Use the suggested focal loss and sampling weight adjustments\n")
            f.write("  * Consider specialized data augmentation for these classes\n")
        
        very_imbalanced = any(stats['positive_ratio'] < 0.01 for stats in problem_classes.values())
        if very_imbalanced:
            f.write("\n- Severe class imbalance detected:\n")
            f.write("  * Use stronger oversampling for rare classes\n")
            f.write("  * Implement class-balanced loss functions\n")
            f.write("  * Consider data augmentation specific to rare classes\n")
        
        random_or_worse = any(stats['auroc'] <= 0.5 for stats in problem_classes.values())
        if random_or_worse:
            f.write("\n- Some classes perform at or below random guessing:\n")
            f.write("  * Review training examples for these classes\n")
            f.write("  * Consider pseudo-labeling or semi-supervised approaches\n")
            f.write("  * Try a completely different architecture for these specific classes\n")

def main():
    args = parse_args()
    output_dir = create_output_dir(args.output_dir)
    
    # Load predictions
    print(f"Loading predictions from {args.predictions_path}")
    df = load_predictions(args.predictions_path)
    
    # Calculate metrics
    print("Calculating AUROC metrics...")
    metrics = calculate_auroc_metrics(df, threshold=args.threshold)
    
    # Save metrics to file
    metrics_path = os.path.join(output_dir, 'auroc_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("AUROC METRICS SUMMARY\n")
        f.write("====================\n\n")
        f.write(f"Macro-average AUROC: {metrics['macro_auroc']:.4f}\n")
        f.write(f"Micro-average AUROC: {metrics['micro_auroc']:.4f}\n\n")
        
        f.write("Per-class AUROC:\n")
        for class_name, class_metrics in metrics['per_class'].items():
            if 'auroc' in class_metrics:
                f.write(f"{class_name}:\n")
                f.write(f"  AUROC: {class_metrics['auroc']:.4f}\n")
                f.write(f"  Positive samples: {class_metrics['positive_count']} ({class_metrics['positive_ratio']*100:.2f}%)\n")
                f.write(f"  F1 Score: {class_metrics['f1']:.4f}\n")
                f.write(f"  Precision: {class_metrics['precision']:.4f}, Recall: {class_metrics['recall']:.4f}\n\n")
    
    print(f"Metrics saved to {metrics_path}")
    
    # Create visualizations
    print("Generating visualizations...")
    plot_auroc_bars(metrics, os.path.join(output_dir, 'auroc_by_class.png'))
    plot_roc_curves(metrics, output_dir)
    
    # Analyze difficult classes
    print("Analyzing problematic classes...")
    analyze_difficult_classes(metrics, output_dir)
    
    # Generate improvement suggestions
    print("Generating improvement suggestions...")
    generate_improvement_suggestions(metrics, os.path.join(output_dir, 'improvement_suggestions.txt'))
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")

if __name__ == "__main__":
    main() 