import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
import signal
import sys

def find_latest_metrics(output_dir):
    """Find and read the latest metrics file"""
    metrics_files = list(Path(output_dir).glob('training_metrics*.csv'))
    if not metrics_files:
        return None
    
    latest_file = max(metrics_files, key=lambda x: x.stat().st_mtime)
    try:
        return pd.read_csv(latest_file)
    except:
        return None

def analyze_performance(df):
    """Analyze training performance"""
    if df is None or len(df) < 5:
        return True, "Not enough data"
    
    # Get the last 5 epochs
    recent_df = df.tail(5)
    
    # Calculate mean AUROC trend
    mean_auroc = recent_df['val_mean_auroc'].mean()
    min_auroc = recent_df['val_mean_auroc'].min()
    max_auroc = recent_df['val_mean_auroc'].max()
    
    # Check individual class performance
    class_columns = [col for col in df.columns if col.startswith('val_auroc_')]
    problem_classes = []
    
    for col in class_columns:
        class_name = col.replace('val_auroc_', '')
        recent_mean = recent_df[col].mean()
        if recent_mean < 0.55:  # Threshold for acceptable performance
            problem_classes.append(f"{class_name}: {recent_mean:.3f}")
    
    # Decision criteria
    continue_training = True
    message = ""
    
    # Check if performance is too low
    if mean_auroc < 0.5:
        continue_training = False
        message = f"Mean AUROC too low: {mean_auroc:.3f}"
    
    # Check if there's no improvement
    if max_auroc - min_auroc < 0.001:
        continue_training = False
        message = f"No improvement in last 5 epochs. Mean AUROC: {mean_auroc:.3f}"
    
    # Report problematic classes
    if problem_classes:
        message += f"\nProblematic classes:\n" + "\n".join(problem_classes)
    
    return continue_training, message

def main():
    output_dir = "output_improved_auroc_v2"
    check_interval = 300  # 5 minutes
    max_no_improvement_time = 7200  # 2 hours
    start_time = time.time()
    last_improvement_time = start_time
    best_auroc = 0
    
    print(f"Monitoring training in {output_dir}")
    print("Will stop training if no improvement for 2 hours or if performance is poor")
    
    while True:
        try:
            df = find_latest_metrics(output_dir)
            if df is not None:
                continue_training, message = analyze_performance(df)
                
                # Check for improvement
                if df['val_mean_auroc'].max() > best_auroc:
                    best_auroc = df['val_mean_auroc'].max()
                    last_improvement_time = time.time()
                    print(f"\nNew best AUROC: {best_auroc:.4f}")
                
                # Print status
                current_epoch = len(df)
                print(f"\nEpoch {current_epoch}")
                print(message)
                
                # Check if we should stop
                if not continue_training:
                    print("\nStopping training due to poor performance...")
                    os.system("taskkill /IM python.exe /F")
                    break
                
                # Check for no improvement timeout
                if time.time() - last_improvement_time > max_no_improvement_time:
                    print("\nNo improvement for 2 hours. Stopping training...")
                    os.system("taskkill /IM python.exe /F")
                    break
            
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(check_interval)

if __name__ == "__main__":
    main() 