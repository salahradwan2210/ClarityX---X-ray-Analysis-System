import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import seaborn as sns
from torch.utils.data import DataLoader
from collections import defaultdict

# Import necessary modules from fast_train_convnext.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fast_train_convnext import ConvNextModel, ChestXrayDatasetDataframe, get_transforms, Normalize

def parse_args():
    parser = argparse.ArgumentParser(description='Predict and analyze model performance')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--csv_path', type=str, default='data/Data_Entry_2017.csv', 
                        help='Path to the CSV file with image metadata')
    parser.add_argument('--output_dir', type=str, default='analysis_results', 
                        help='Directory to save analysis results')
    parser.add_argument('--predictions_file', type=str, default='predictions.csv',
                        help='Filename to save predictions to (within output_dir)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for prediction')
    parser.add_argument('--img_size', type=int, default=384, help='Image size for model input')
    parser.add_argument('--num_classes', type=int, default=14, help='Number of classes to predict')
    parser.add_argument('--no_finding_threshold', type=float, default=0.4, 
                        help='Threshold for No Finding classification')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for computation')
    parser.add_argument('--use_val_split', action='store_true', 
                        help='Use validation split for prediction instead of test split')
    
    return parser.parse_args()

def setup_model(args):
    """Load and set up the model for prediction."""
    device = torch.device(args.device)
    
    # Create model
    model = ConvNextModel(num_classes=args.num_classes)
    
    # Load model weights
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, device

def setup_dataloader(args):
    """Set up data loader for the test dataset."""
    # Get transforms for validation/testing
    _, val_transform = get_transforms(img_size=args.img_size)
    
    # Load CSV file
    df = pd.read_csv(args.csv_path)
    
    # Use validation split logic from training script to ensure consistent evaluation
    # This is a simplified version that assumes 20% of data is for validation
    # In a real implementation, you'd want to ensure this matches your training/validation split
    
    all_img_paths = df['Image Index'].values
    val_size = int(0.2 * len(all_img_paths))
    np.random.seed(42)  # Ensure reproducibility
    indices = np.random.permutation(len(all_img_paths))
    val_indices = indices[:val_size]
    val_df = df.iloc[val_indices]
    
    dataset = ChestXrayDatasetDataframe(
        val_df,
        data_dir=args.data_dir,
        transform=val_transform,
        return_paths=True
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader

def run_predictions(model, dataloader, device, class_names):
    """Run predictions on the test dataset."""
    all_labels = []
    all_probs = []
    all_image_paths = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running predictions"):
            images = batch[0].to(device)
            labels = batch[1].to(device)
            image_paths = batch[2] if len(batch) > 2 else ["unknown"] * len(labels)
            
            outputs = model(images)
            probabilities = torch.sigmoid(outputs)
            
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probabilities.cpu().numpy())
            all_image_paths.extend(image_paths)
    
    # Concatenate batch results
    y_true = np.vstack(all_labels)
    y_pred_probs = np.vstack(all_probs)
    
    # Create dataframe with results
    results_df = pd.DataFrame(y_pred_probs, columns=class_names)
    results_df['image_path'] = all_image_paths
    
    # Add ground truth to dataframe
    for i, class_name in enumerate(class_names):
        results_df[f'{class_name}_true'] = y_true[:, i]
    
    return results_df, y_true, y_pred_probs

def save_predictions(results_df, output_path):
    """Save predictions to a CSV file."""
    print(f"Saving predictions to {output_path}")
    results_df.to_csv(output_path, index=False)
    print(f"Predictions saved successfully. Shape: {results_df.shape}")
    
    # Print sample of predictions
    print("\nSample predictions (first 5 rows):")
    print(results_df.head())
    
    return output_path

def calculate_metrics(y_true, y_pred_probs, class_names):
    """Calculate performance metrics for each class."""
    metrics = {}
    
    # Calculate AUROC for each class
    for i, class_name in enumerate(class_names):
        try:
            # Skip classes with all zeros or all ones in ground truth
            if np.sum(y_true[:, i]) == 0 or np.sum(y_true[:, i]) == len(y_true):
                auroc = float('nan')
            else:
                auroc = roc_auc_score(y_true[:, i], y_pred_probs[:, i])
            metrics[class_name] = {'auroc': auroc}
        except Exception as e:
            print(f"Error calculating AUROC for {class_name}: {e}")
            metrics[class_name] = {'auroc': float('nan')}
    
    # Calculate overall metrics
    avg_auroc = np.nanmean([m['auroc'] for m in metrics.values()])
    metrics['average'] = {'auroc': avg_auroc}
    
    return metrics

def get_disease_statistics(y_true, class_names):
    """Get statistics about disease prevalence in the dataset."""
    stats = {}
    total_samples = len(y_true)
    
    for i, class_name in enumerate(class_names):
        positive_count = np.sum(y_true[:, i])
        stats[class_name] = {
            'positive_count': int(positive_count),
            'prevalence': float(positive_count / total_samples)
        }
    
    return stats

def analyze_no_finding(results_df, metrics, no_finding_idx, threshold=0.4):
    """Analyze the 'No Finding' class in relation to other diseases."""
    no_finding_class = results_df.columns[no_finding_idx]
    disease_classes = [col for i, col in enumerate(results_df.columns) 
                       if i != no_finding_idx and '_true' not in col and col != 'image_path']
    
    analysis = {
        'threshold': threshold,
        'no_finding': {
            'name': no_finding_class,
            'auroc': metrics[no_finding_class]['auroc'],
        },
        'conflicts': {}
    }
    
    # Apply threshold to get binary predictions
    results_df['no_finding_pred'] = (results_df[no_finding_class] > threshold).astype(int)
    
    # Find cases where No Finding is predicted but other diseases are also predicted
    conflicting_cases = {}
    
    for disease in disease_classes:
        results_df[f'{disease}_pred'] = (results_df[disease] > threshold).astype(int)
        
        # Find conflicts (No Finding = 1 and disease = 1)
        conflicts = results_df[(results_df['no_finding_pred'] == 1) & 
                              (results_df[f'{disease}_pred'] == 1)]
        
        if len(conflicts) > 0:
            conflict_rate = len(conflicts) / len(results_df)
            conflicting_cases[disease] = {
                'count': len(conflicts),
                'rate': conflict_rate,
                'avg_no_finding_prob': float(conflicts[no_finding_class].mean()),
                'avg_disease_prob': float(conflicts[disease].mean())
            }
    
    analysis['conflicts'] = conflicting_cases
    return analysis

def plot_class_distribution(disease_stats, output_path):
    """Plot the distribution of classes in the dataset."""
    plt.figure(figsize=(12, 8))
    
    classes = list(disease_stats.keys())
    counts = [stats['positive_count'] for stats in disease_stats.values()]
    
    # Sort by count
    sorted_indices = np.argsort(counts)[::-1]
    sorted_classes = [classes[i] for i in sorted_indices]
    sorted_counts = [counts[i] for i in sorted_indices]
    
    sns.barplot(x=sorted_counts, y=sorted_classes)
    plt.xlabel('Count')
    plt.ylabel('Disease')
    plt.title('Class Distribution in Test Set')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_auroc_comparison(metrics, output_path):
    """Plot comparison of AUROC across classes."""
    plt.figure(figsize=(12, 8))
    
    classes = [c for c in metrics.keys() if c != 'average']
    aurocs = [metrics[c]['auroc'] for c in classes]
    
    # Sort by AUROC
    sorted_indices = np.argsort(aurocs)
    sorted_classes = [classes[i] for i in sorted_indices]
    sorted_aurocs = [aurocs[i] for i in sorted_indices]
    
    colors = ['red' if a < 0.7 else 'orange' if a < 0.8 else 'green' for a in sorted_aurocs]
    
    plt.barh(sorted_classes, sorted_aurocs, color=colors)
    plt.axvline(x=metrics['average']['auroc'], color='blue', linestyle='--', 
               label=f'Average: {metrics["average"]["auroc"]:.4f}')
    
    plt.xlabel('AUROC')
    plt.ylabel('Disease')
    plt.title('AUROC by Disease Class')
    plt.xlim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_roc_curves(y_true, y_pred_probs, class_names, output_dir):
    """Plot ROC curves for each class."""
    # Plot ROC curve for each class
    for i, class_name in enumerate(class_names):
        if np.sum(y_true[:, i]) == 0 or np.sum(y_true[:, i]) == len(y_true):
            continue
            
        plt.figure(figsize=(8, 8))
        
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_probs[:, i])
        auroc = roc_auc_score(y_true[:, i], y_pred_probs[:, i])
        
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {auroc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve: {class_name}')
        plt.legend(loc="lower right")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'roc_curve_{class_name.replace(" ", "_")}.png'))
        plt.close()

def find_misclassifications(results_df, class_names, threshold=0.5):
    """Find and analyze misclassifications."""
    misclassifications = {}
    
    for class_name in class_names:
        # Skip the image_path column and *_true columns
        if class_name == 'image_path' or '_true' in class_name:
            continue
            
        true_col = f'{class_name}_true'
        if true_col not in results_df.columns:
            continue
            
        # False positives
        fp_mask = (results_df[class_name] > threshold) & (results_df[true_col] == 0)
        fp_cases = results_df[fp_mask].sort_values(by=class_name, ascending=False)
        
        # False negatives
        fn_mask = (results_df[class_name] <= threshold) & (results_df[true_col] == 1)
        fn_cases = results_df[fn_mask].sort_values(by=class_name, ascending=True)
        
        misclassifications[class_name] = {
            'false_positives': {
                'count': int(fp_mask.sum()),
                'rate': float(fp_mask.sum() / len(results_df)),
                'examples': fp_cases.head(5)['image_path'].tolist() if len(fp_cases) > 0 else []
            },
            'false_negatives': {
                'count': int(fn_mask.sum()),
                'rate': float(fn_mask.sum() / len(results_df)),
                'examples': fn_cases.head(5)['image_path'].tolist() if len(fn_cases) > 0 else []
            }
        }
    
    return misclassifications

def save_analysis_results(metrics, disease_stats, no_finding_analysis, misclassifications, output_path):
    """Save analysis results to a text file."""
    with open(output_path, 'w') as f:
        f.write("=== MODEL PERFORMANCE ANALYSIS ===\n\n")
        
        # Overall metrics
        f.write("OVERALL METRICS:\n")
        f.write(f"Average AUROC: {metrics['average']['auroc']:.4f}\n\n")
        
        # Class-specific metrics
        f.write("CLASS-SPECIFIC METRICS:\n")
        for class_name, metric in sorted(metrics.items(), key=lambda x: x[1]['auroc'] if x[0] != 'average' else 0):
            if class_name == 'average':
                continue
            f.write(f"{class_name:25} - AUROC: {metric['auroc']:.4f}, ")
            f.write(f"Prevalence: {disease_stats[class_name]['prevalence']:.4f} ")
            f.write(f"({disease_stats[class_name]['positive_count']} cases)\n")
        
        f.write("\n")
        
        # No Finding analysis
        f.write("NO FINDING ANALYSIS:\n")
        f.write(f"No Finding AUROC: {no_finding_analysis['no_finding']['auroc']:.4f}\n")
        f.write(f"Threshold used: {no_finding_analysis['threshold']}\n\n")
        
        f.write("Conflicts with other diseases:\n")
        for disease, conflict in no_finding_analysis['conflicts'].items():
            f.write(f"{disease:25} - Conflicts: {conflict['count']} ({conflict['rate']:.4f})\n")
            f.write(f"                         Avg No Finding prob: {conflict['avg_no_finding_prob']:.4f}\n")
            f.write(f"                         Avg Disease prob: {conflict['avg_disease_prob']:.4f}\n")
        
        f.write("\n")
        
        # Misclassifications
        f.write("MISCLASSIFICATION ANALYSIS:\n")
        for class_name, misclass in sorted(misclassifications.items(), 
                                          key=lambda x: x[1]['false_negatives']['rate'] + x[1]['false_positives']['rate'],
                                          reverse=True):
            f.write(f"{class_name:25}\n")
            f.write(f"  False Positives: {misclass['false_positives']['count']} ")
            f.write(f"({misclass['false_positives']['rate']:.4f})\n")
            f.write(f"  False Negatives: {misclass['false_negatives']['count']} ")
            f.write(f"({misclass['false_negatives']['rate']:.4f})\n")
        
        f.write("\n")
        
        # Best and worst performing classes
        sorted_metrics = sorted([(k, v['auroc']) for k, v in metrics.items() if k != 'average'], 
                              key=lambda x: x[1])
        
        f.write("BEST PERFORMING CLASSES:\n")
        for class_name, auroc in sorted_metrics[-3:]:
            f.write(f"{class_name:25} - AUROC: {auroc:.4f}\n")
        
        f.write("\nWORST PERFORMING CLASSES:\n")
        for class_name, auroc in sorted_metrics[:3]:
            f.write(f"{class_name:25} - AUROC: {auroc:.4f}\n")

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define class names (ensure this matches your model's classes)
    class_names = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
        'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]
    
    # No Finding class index (assuming it's the first class)
    no_finding_idx = class_names.index('No Finding') if 'No Finding' in class_names else None
    
    # Setup model and dataloader
    model, device = setup_model(args)
    dataloader = setup_dataloader(args)
    
    # Run predictions
    print("Running predictions...")
    results_df, y_true, y_pred_probs = run_predictions(model, dataloader, device, class_names)
    
    # Save predictions to CSV
    predictions_path = os.path.join(args.output_dir, args.predictions_file)
    save_predictions(results_df, predictions_path)
    
    # Calculate metrics
    print("Calculating performance metrics...")
    metrics = calculate_metrics(y_true, y_pred_probs, class_names)
    
    # Get disease statistics
    disease_stats = get_disease_statistics(y_true, class_names)
    
    # Analyze No Finding class
    no_finding_analysis = None
    if no_finding_idx is not None:
        print("Analyzing 'No Finding' class...")
        no_finding_analysis = analyze_no_finding(
            results_df, metrics, no_finding_idx, threshold=args.no_finding_threshold
        )
    
    # Find misclassifications
    print("Finding misclassifications...")
    misclassifications = find_misclassifications(results_df, class_names)
    
    # Generate visualizations
    print("Generating visualizations...")
    plot_class_distribution(
        disease_stats, 
        os.path.join(args.output_dir, 'class_distribution.png')
    )
    plot_auroc_comparison(
        metrics, 
        os.path.join(args.output_dir, 'auroc_comparison.png')
    )
    plot_roc_curves(
        y_true, 
        y_pred_probs, 
        class_names, 
        args.output_dir
    )
    
    # Save analysis results
    print("Saving analysis results...")
    save_analysis_results(
        metrics, 
        disease_stats, 
        no_finding_analysis, 
        misclassifications, 
        os.path.join(args.output_dir, 'analysis_results.txt')
    )
    
    # Print summary
    avg_auroc = metrics['average']['auroc']
    print(f"\nAnalysis complete. Average AUROC: {avg_auroc:.4f}")
    
    # Get top and bottom performing classes
    auroc_values = [(class_name, metrics[class_name]['auroc']) 
                    for class_name in class_names 
                    if not np.isnan(metrics[class_name]['auroc'])]
    
    top_classes = sorted(auroc_values, key=lambda x: x[1], reverse=True)[:3]
    bottom_classes = sorted(auroc_values, key=lambda x: x[1])[:3]
    
    print("\nTop performing classes:")
    for class_name, auroc in top_classes:
        print(f"  {class_name}: {auroc:.4f}")
    
    print("\nBottom performing classes:")
    for class_name, auroc in bottom_classes:
        print(f"  {class_name}: {auroc:.4f}")
    
    print(f"\nDetailed results saved to {args.output_dir}")
    print(f"Predictions saved to {predictions_path}")

if __name__ == "__main__":
    main() 