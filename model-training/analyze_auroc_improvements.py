import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import seaborn as sns
import glob
from pathlib import Path

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze AUROC improvements between model runs')
    parser.add_argument('--base_dir', type=str, required=True, 
                        help='Base directory containing output folders')
    parser.add_argument('--models', type=str, nargs='+', required=True,
                        help='List of model output folders to compare (e.g., output_baseline output_improved_auroc)')
    parser.add_argument('--output_dir', type=str, default='auroc_improvement_analysis',
                        help='Directory to save the analysis results')
    return parser.parse_args()

def load_metrics_files(base_dir, model_folders):
    """Load training metrics files from multiple model runs"""
    metrics_data = {}
    
    for model_folder in model_folders:
        folder_path = os.path.join(base_dir, model_folder)
        metrics_files = glob.glob(os.path.join(folder_path, "training_metrics*.csv"))
        
        if not metrics_files:
            print(f"Warning: No metrics files found for {model_folder}")
            continue
        
        # Sort by last modified time (latest first)
        metrics_files.sort(key=os.path.getmtime, reverse=True)
        latest_file = metrics_files[0]
        
        try:
            df = pd.read_csv(latest_file)
            metrics_data[model_folder] = {
                'df': df,
                'file_path': latest_file
            }
            print(f"Loaded metrics from {latest_file}")
        except Exception as e:
            print(f"Error loading {latest_file}: {e}")
    
    return metrics_data

def extract_auroc_by_class(metrics_data):
    """Extract AUROC scores by class from metrics dataframes"""
    class_names = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 
        'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]
    
    auroc_data = {}
    
    for model_name, model_metrics in metrics_data.items():
        df = model_metrics['df']
        auroc_by_class = {}
        
        # Find the best epoch (highest validation mean AUROC)
        if 'val_mean_auroc' in df.columns:
            best_epoch_idx = df['val_mean_auroc'].idxmax()
            best_epoch = df.iloc[best_epoch_idx]
        else:
            # If no validation data, use the last epoch
            best_epoch = df.iloc[-1]
        
        # Extract AUROC for each class
        for class_name in class_names:
            col_name = f'val_auroc_{class_name}'
            if col_name in best_epoch:
                auroc_by_class[class_name] = best_epoch[col_name]
        
        # Get mean AUROC
        if 'val_mean_auroc' in best_epoch:
            auroc_by_class['mean'] = best_epoch['val_mean_auroc']
        else:
            # Calculate mean from available classes
            values = [v for v in auroc_by_class.values() if not np.isnan(v)]
            auroc_by_class['mean'] = np.mean(values) if values else np.nan
        
        auroc_data[model_name] = {
            'auroc': auroc_by_class,
            'epoch': best_epoch['epoch'] if 'epoch' in best_epoch else None
        }
    
    return auroc_data

def compare_auroc_performance(auroc_data, reference_model):
    """Compare AUROC performance across models"""
    if reference_model not in auroc_data:
        print(f"Error: Reference model {reference_model} not found in data")
        reference_model = list(auroc_data.keys())[0]
        print(f"Using {reference_model} as reference instead")
    
    reference = auroc_data[reference_model]['auroc']
    
    comparison_data = {}
    
    for model_name, model_data in auroc_data.items():
        if model_name == reference_model:
            continue
        
        model_auroc = model_data['auroc']
        diff_by_class = {}
        
        # Compare each class
        for class_name, ref_auroc in reference.items():
            if class_name in model_auroc:
                current_auroc = model_auroc[class_name]
                diff = current_auroc - ref_auroc
                diff_by_class[class_name] = {
                    'reference': ref_auroc,
                    'current': current_auroc,
                    'diff': diff,
                    'percent_change': (diff / ref_auroc) * 100 if ref_auroc != 0 else float('inf')
                }
        
        comparison_data[model_name] = diff_by_class
    
    return comparison_data

def plot_auroc_comparison(auroc_data, comparison_data, output_path, reference_model):
    """Create bar chart comparing AUROC across models"""
    # Get all class names (excluding 'mean')
    all_classes = []
    for model_data in auroc_data.values():
        for class_name in model_data['auroc']:
            if class_name != 'mean' and class_name not in all_classes:
                all_classes.append(class_name)
    
    # Sort classes alphabetically
    all_classes.sort()
    
    # Extract data for plotting
    model_names = list(auroc_data.keys())
    
    # Prepare data for plotting
    plot_data = []
    
    for class_name in all_classes:
        class_row = {'Class': class_name}
        
        for model_name in model_names:
            if class_name in auroc_data[model_name]['auroc']:
                class_row[model_name] = auroc_data[model_name]['auroc'][class_name]
            else:
                class_row[model_name] = np.nan
        
        plot_data.append(class_row)
    
    # Convert to DataFrame
    plot_df = pd.DataFrame(plot_data)
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Set width of bars
    bar_width = 0.8 / len(model_names)
    
    # Set positions of bars on X axis
    r = np.arange(len(all_classes))
    
    # Create bars
    for i, model_name in enumerate(model_names):
        pos = r + i * bar_width
        bars = plt.barh(pos, plot_df[model_name], height=bar_width, 
                 label=f"{model_name} (Mean: {auroc_data[model_name]['auroc']['mean']:.4f})")
        
        # Add AUROC values on bars
        for j, bar in enumerate(bars):
            if not np.isnan(plot_df[model_name].iloc[j]):
                plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{plot_df[model_name].iloc[j]:.3f}', 
                        ha='left', va='center', fontsize=8)
    
    # Add labels and title
    plt.xlabel('AUROC Score')
    plt.title('AUROC Comparison by Class Across Models', fontsize=14)
    plt.yticks([r + bar_width * (len(model_names) - 1) / 2 for r in range(len(all_classes))], all_classes)
    plt.xlim(0, 1.0)
    plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=0.7, color='green', linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Saved AUROC comparison chart to {output_path}")

def plot_auroc_improvement(comparison_data, output_path, reference_model):
    """Create bar chart showing AUROC improvements"""
    # Prepare data for each model comparison
    for model_name, diff_data in comparison_data.items():
        # Extract class names and differences
        classes = []
        diffs = []
        percent_changes = []
        
        for class_name, values in diff_data.items():
            if class_name != 'mean':
                classes.append(class_name)
                diffs.append(values['diff'])
                percent_changes.append(values['percent_change'])
        
        # Create a DataFrame for easier sorting
        df = pd.DataFrame({
            'Class': classes,
            'AUROC_Difference': diffs,
            'Percent_Change': percent_changes
        })
        
        # Sort by AUROC difference
        df = df.sort_values('AUROC_Difference')
        
        # Set colors based on improvement
        colors = ['green' if diff > 0 else 'red' for diff in df['AUROC_Difference']]
        
        # Create the plot
        plt.figure(figsize=(14, 10))
        
        # Create bars
        bars = plt.barh(df['Class'], df['AUROC_Difference'], color=colors, alpha=0.7)
        
        # Add values to bars
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.001 if bar.get_width() >= 0 else bar.get_width() - 0.02, 
                    bar.get_y() + bar.get_height()/2, 
                    f'{df["AUROC_Difference"].iloc[i]:.3f} ({df["Percent_Change"].iloc[i]:.1f}%)', 
                    ha='left' if bar.get_width() >= 0 else 'right', 
                    va='center')
        
        # Add labels and title
        plt.xlabel('AUROC Difference')
        plt.title(f'AUROC Improvement: {model_name} vs {reference_model}', fontsize=14)
        plt.axvline(x=0, color='gray', linestyle='--')
        
        # Save the plot
        model_output_path = output_path.replace('.png', f'_{model_name}.png')
        plt.tight_layout()
        plt.savefig(model_output_path)
        plt.close()
        
        print(f"Saved AUROC improvement chart to {model_output_path}")

def generate_summary_report(auroc_data, comparison_data, output_path, reference_model):
    """Generate a detailed summary report of AUROC improvements"""
    with open(output_path, 'w') as f:
        f.write("=== AUROC Improvement Analysis ===\n\n")
        
        # Write summary for each model
        f.write("== Model Summary ==\n")
        for model_name, model_data in auroc_data.items():
            mean_auroc = model_data['auroc']['mean']
            best_epoch = model_data['epoch']
            
            f.write(f"\n{model_name}:\n")
            f.write(f"  Mean AUROC: {mean_auroc:.4f}\n")
            if best_epoch is not None:
                f.write(f"  Best Epoch: {best_epoch}\n")
            
            # Find best and worst classes
            class_aurocs = {k: v for k, v in model_data['auroc'].items() if k != 'mean'}
            if class_aurocs:
                best_class = max(class_aurocs.items(), key=lambda x: x[1])
                worst_class = min(class_aurocs.items(), key=lambda x: x[1])
                
                f.write(f"  Best Class: {best_class[0]} (AUROC = {best_class[1]:.4f})\n")
                f.write(f"  Worst Class: {worst_class[0]} (AUROC = {worst_class[1]:.4f})\n")
        
        # Write comparison details
        f.write("\n\n== AUROC Improvements ==\n")
        for model_name, diff_data in comparison_data.items():
            f.write(f"\n{model_name} vs {reference_model}:\n")
            
            # Calculate overall improvement
            mean_diff = diff_data['mean']['diff'] if 'mean' in diff_data else np.nan
            
            f.write(f"  Overall Mean AUROC Change: {mean_diff:.4f}")
            if not np.isnan(mean_diff):
                if mean_diff > 0:
                    f.write(f" (↑ Improvement of {mean_diff:.4f})\n")
                else:
                    f.write(f" (↓ Decrease of {abs(mean_diff):.4f})\n")
            else:
                f.write("\n")
            
            # Sort classes by improvement
            classes_by_improvement = sorted(
                [(c, d['diff']) for c, d in diff_data.items() if c != 'mean'],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Most improved classes
            f.write("\n  Most Improved Classes:\n")
            for class_name, diff in classes_by_improvement[:5]:
                if diff > 0:
                    reference = diff_data[class_name]['reference']
                    current = diff_data[class_name]['current']
                    percent = diff_data[class_name]['percent_change']
                    
                    f.write(f"    {class_name}: {reference:.4f} → {current:.4f} ")
                    f.write(f"(↑ +{diff:.4f}, +{percent:.1f}%)\n")
                else:
                    break
            
            # Most deteriorated classes
            deteriorated = [x for x in classes_by_improvement if x[1] < 0]
            if deteriorated:
                f.write("\n  Classes with Decreased Performance:\n")
                for class_name, diff in deteriorated:
                    reference = diff_data[class_name]['reference']
                    current = diff_data[class_name]['current']
                    percent = diff_data[class_name]['percent_change']
                    
                    f.write(f"    {class_name}: {reference:.4f} → {current:.4f} ")
                    f.write(f"(↓ {diff:.4f}, {percent:.1f}%)\n")
            else:
                f.write("\n  No classes showed decreased performance!\n")
    
    # Create a summary CSV file as well
    csv_path = output_path.replace('.txt', '.csv')
    
    # Prepare CSV data
    csv_data = []
    
    # Get all class names across all models
    all_classes = set()
    for model_data in auroc_data.values():
        all_classes.update(model_data['auroc'].keys())
    
    # Sort classes and ensure 'mean' is first
    all_classes = sorted(list(all_classes - {'mean'}))
    all_classes = ['mean'] + all_classes
    
    # Add model data rows
    for class_name in all_classes:
        row = {'Class': class_name}
        
        # Add AUROC for each model
        for model_name, model_data in auroc_data.items():
            if class_name in model_data['auroc']:
                row[f"{model_name}_AUROC"] = model_data['auroc'][class_name]
            else:
                row[f"{model_name}_AUROC"] = np.nan
        
        # Add difference for each comparison
        for model_name, diff_data in comparison_data.items():
            if class_name in diff_data:
                row[f"{model_name}_vs_{reference_model}_diff"] = diff_data[class_name]['diff']
                row[f"{model_name}_vs_{reference_model}_percent"] = diff_data[class_name]['percent_change']
            else:
                row[f"{model_name}_vs_{reference_model}_diff"] = np.nan
                row[f"{model_name}_vs_{reference_model}_percent"] = np.nan
        
        csv_data.append(row)
    
    # Write CSV file
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    
    print(f"Saved summary report to {output_path}")
    print(f"Saved summary CSV to {csv_path}")

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving results to {args.output_dir}")
    
    # Load metrics files
    print("Loading metrics files...")
    metrics_data = load_metrics_files(args.base_dir, args.models)
    
    if len(metrics_data) < 1:
        print("Error: No valid metrics files found")
        return
    
    # Extract AUROC data by class
    print("Extracting AUROC data...")
    auroc_data = extract_auroc_by_class(metrics_data)
    
    # Use first model as reference by default
    reference_model = args.models[0]
    
    # Compare AUROC performance
    print(f"Comparing AUROC performance against reference model: {reference_model}")
    comparison_data = compare_auroc_performance(auroc_data, reference_model)
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # Plot AUROC comparison
    plot_auroc_comparison(
        auroc_data, 
        comparison_data, 
        os.path.join(args.output_dir, 'auroc_comparison.png'),
        reference_model
    )
    
    # Plot AUROC improvements
    plot_auroc_improvement(
        comparison_data, 
        os.path.join(args.output_dir, 'auroc_improvement.png'),
        reference_model
    )
    
    # Generate summary report
    generate_summary_report(
        auroc_data,
        comparison_data,
        os.path.join(args.output_dir, 'improvement_summary.txt'),
        reference_model
    )
    
    print("\nAUROC improvement analysis complete!")
    print(f"Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()