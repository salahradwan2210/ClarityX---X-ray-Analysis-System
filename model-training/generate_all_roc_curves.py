import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, roc_auc_score

# Import model definitions
from models.convnext_large_model import ConvNextLargeModel
from models.resnet152_model import ResNet152Model  
from models.densenet121_model import DenseNet121Model
from models.densenet169_model import DenseNet169Model

# Define the class names
CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 
    'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# Default paths for different model checkpoints
MODEL_PATHS = {
    'convnext': 'best_model_epoch_28_auroc_0.9688.pth',  # ConvNeXt Large
    'resnet152': 'best_model.pth',                       # ResNet 152
    'densenet121': 'best_model.pth',                     # DenseNet 121
    'densenet169': 'best_model.pth'                      # DenseNet 169
}

class ChestXrayDataset(torch.utils.data.Dataset):
    """Dataset class for chest X-ray images"""
    
    def __init__(self, image_dir, df, transform=None, train=True):
        """
        Initialize the dataset
        
        Parameters:
            image_dir (str): Directory containing the images
            df (DataFrame): DataFrame with image information
            transform (callable, optional): Transform to apply to images
            train (bool): Whether this is training data
        """
        self.image_dir = image_dir
        self.df = df
        self.transform = transform
        self.train = train
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row['Image Index']
        image_path = self.find_image_path(image_name)
        
        # Load and transform image
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Create label tensor
        labels = torch.zeros(len(CLASS_NAMES))
        if 'Finding Labels' in self.df.columns:
            finding_labels = row['Finding Labels'].split('|')
            for label in finding_labels:
                if label in CLASS_NAMES:
                    labels[CLASS_NAMES.index(label)] = 1
        
        return image, labels
    
    def find_image_path(self, image_name):
        """Find the path to an image in the directory structure"""
        # Try different subdirectories where images might be stored
        potential_dirs = [
            f'data/images_{image_name[:3]}/',
            f'data/images_{image_name.split("_")[0]}/',
            'data/test_images/',
            'data/'
        ]
        
        for directory in potential_dirs:
            path = os.path.join(directory, image_name)
            if os.path.exists(path):
                return path
        
        # If not found in standard locations, do a recursive search
        for root, _, files in os.walk('data'):
            if image_name in files:
                return os.path.join(root, image_name)
        
        raise FileNotFoundError(f"Image {image_name} not found")

def load_data(args):
    """Load and prepare data"""
    print("Loading data...")
    
    # Load Data_Entry_2017.csv
    df = pd.read_csv(args.data_entry)
    
    # Load test_list.txt
    test_files = []
    with open(args.test_list, 'r') as f:
        for line in f:
            test_files.append(line.strip())
    
    # Filter dataframe to include only test files
    test_df = df[df['Image Index'].isin(test_files)]
    print(f"Test set size: {len(test_df)}")
    
    # If sample_size is specified, sample a subset
    if args.sample_size > 0 and args.sample_size < len(test_df):
        test_df = test_df.sample(args.sample_size, random_state=args.seed)
        print(f"Sampled {args.sample_size} images from test set")
    
    # Create test transforms
    test_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and loader
    test_dataset = ChestXrayDataset(args.data_dir, test_df, transform=test_transform, train=False)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return test_loader

def load_model(model_path, model_type='convnext', num_classes=14, device='cuda'):
    """Load a model from checkpoint"""
    print(f"Loading {model_type} model from {model_path}...")
    
    # Create the appropriate model
    if model_type == 'convnext':
        model = ConvNextLargeModel(num_classes=num_classes, pretrained=False)
    elif model_type == 'resnet152':
        model = ResNet152Model(num_classes=num_classes, pretrained=False)
    elif model_type == 'densenet121':
        model = DenseNet121Model(num_classes=num_classes, pretrained=False)
    elif model_type == 'densenet169':
        model = DenseNet169Model(num_classes=num_classes, pretrained=False)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model

def evaluate_model(model, data_loader, device, threshold=0.5):
    """Evaluate model performance"""
    model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs)
    
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    
    return {
        'labels': all_labels,
        'probs': all_probs
    }

def plot_roc_curves(y_true, y_pred, classes, model_name, output_dir, figsize=(15, 10)):
    """Plot ROC curves for each class"""
    plt.figure(figsize=figsize)
    
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    
    # Calculate and plot ROC curve for each class
    for i, cls_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{cls_name} (AUC = {roc_auc:.3f})')
        
        # Calculate average ROC curve
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
    
    # Add average ROC curve
    mean_tpr /= len(classes)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--', lw=2, label=f'Mean ROC (AUC = {mean_auc:.3f})')
    
    # Add baseline
    plt.plot([0, 1], [0, 1], 'r--', lw=2, label='Chance')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves - {model_name}', fontsize=16)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True)
    
    file_path = os.path.join(output_dir, f'roc_curves_{model_name}.png')
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curves saved to {file_path}")
    
    # Return the mean AUC for comparison
    return mean_auc

def calculate_and_display_metrics(results, classes, model_name, output_dir):
    """Calculate and display model metrics"""
    y_true = results['labels']
    y_pred = results['probs']
    
    # Calculate AUROC for each class
    auroc_scores = {}
    for i, cls_name in enumerate(classes):
        try:
            auroc = roc_auc_score(y_true[:, i], y_pred[:, i])
            auroc_scores[cls_name] = auroc
        except ValueError:
            auroc_scores[cls_name] = float('nan')
    
    # Calculate mean AUROC
    valid_aurocs = [score for score in auroc_scores.values() if not np.isnan(score)]
    mean_auroc = np.mean(valid_aurocs) if valid_aurocs else float('nan')
    
    # Display results
    print(f"\n--- {model_name} Results ---")
    print(f"Mean AUROC: {mean_auroc:.4f}")
    print("\nAUROC by class:")
    for class_name, auroc in sorted(auroc_scores.items(), key=lambda x: x[1], reverse=True):
        if not np.isnan(auroc):
            print(f"  {class_name}: {auroc:.4f}")
    
    # Plot ROC curves
    plot_roc_curves(y_true, y_pred, classes, model_name, output_dir)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Class': list(auroc_scores.keys()),
        'AUROC': list(auroc_scores.values())
    })
    metrics_df = metrics_df.sort_values('AUROC', ascending=False)
    metrics_csv_path = os.path.join(output_dir, f'metrics_{model_name}.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Metrics saved to {metrics_csv_path}")
    
    return mean_auroc, auroc_scores

def main():
    """Main function"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate ROC curves for multiple chest X-ray models')
    parser.add_argument('--data_dir', type=str, default='.', help='Path to data directory')
    parser.add_argument('--data_entry', type=str, default='./data/Data_Entry_2017.csv', help='Path to Data_Entry_2017.csv')
    parser.add_argument('--test_list', type=str, default='./data/test_list.txt', help='Path to test list file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading')
    parser.add_argument('--sample_size', type=int, default=500, help='Number of samples to use for evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='roc_comparison_results', help='Directory to save results')
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['convnext', 'resnet152', 'densenet121', 'densenet169'],
                        help='Models to evaluate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for evaluation')
    
    # Parse model-specific checkpoint paths
    parser.add_argument('--convnext_path', type=str, default=MODEL_PATHS['convnext'], 
                        help='Path to ConvNeXt Large checkpoint')
    parser.add_argument('--resnet152_path', type=str, default=MODEL_PATHS['resnet152'], 
                        help='Path to ResNet-152 checkpoint')
    parser.add_argument('--densenet121_path', type=str, default=MODEL_PATHS['densenet121'], 
                        help='Path to DenseNet-121 checkpoint')
    parser.add_argument('--densenet169_path', type=str, default=MODEL_PATHS['densenet169'], 
                        help='Path to DenseNet-169 checkpoint')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load data
    test_loader = load_data(args)
    
    # Update model paths dictionary from args
    model_paths = {
        'convnext': args.convnext_path,
        'resnet152': args.resnet152_path,
        'densenet121': args.densenet121_path,
        'densenet169': args.densenet169_path
    }
    
    # Dictionary to store results for comparison
    results_summary = {}
    
    # Evaluate each model
    for model_type in args.models:
        print(f"\n=== Evaluating {model_type} ===")
        
        # Load the model
        model = load_model(model_paths[model_type], model_type, num_classes=len(CLASS_NAMES), device=args.device)
        
        # Evaluate the model
        print(f"Running evaluation...")
        results = evaluate_model(model, test_loader, args.device)
        
        # Calculate and display metrics
        mean_auroc, class_aurocs = calculate_and_display_metrics(
            results, CLASS_NAMES, model_type, args.output_dir
        )
        
        # Store results for comparison
        results_summary[model_type] = {
            'mean_auroc': mean_auroc,
            'class_aurocs': class_aurocs
        }
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    # Compare model performance
    print("\n=== Model Comparison ===")
    print("Mean AUROC by Model:")
    for model_type, results in sorted(results_summary.items(), key=lambda x: x[1]['mean_auroc'], reverse=True):
        print(f"  {model_type}: {results['mean_auroc']:.4f}")
    
    # Generate comparison bar chart
    plt.figure(figsize=(12, 8))
    models = list(results_summary.keys())
    aurocs = [results_summary[model]['mean_auroc'] for model in models]
    
    # Create readable model names
    model_display_names = {
        'convnext': 'ConvNeXt Large',
        'resnet152': 'ResNet 152',
        'densenet121': 'DenseNet 121',
        'densenet169': 'DenseNet 169'
    }
    
    x_labels = [model_display_names.get(model, model) for model in models]
    
    bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = plt.bar(x_labels, aurocs, color=bar_colors)
    
    # Add values on top of bars
    for bar, auroc in zip(bars, aurocs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{auroc:.4f}", ha='center', va='bottom', fontsize=12)
    
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Mean AUROC', fontsize=14)
    plt.title('Model Performance Comparison', fontsize=16)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    comparison_path = os.path.join(args.output_dir, 'model_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison chart saved to {comparison_path}")
    print(f"All result files saved to {args.output_dir}")

if __name__ == "__main__":
    main() 