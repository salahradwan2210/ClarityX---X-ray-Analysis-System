import os
import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from PIL import Image
import torchvision.transforms as transforms
from models.convnext_large_model import IntegratedConvNextModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

class NIHChestXrayDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image of the expected size
            img = Image.new('RGB', (224, 224), color='black')
            
        if self.transform:
            img = self.transform(img)
            
        label = self.labels[idx] if self.labels is not None else torch.zeros(14)
        return img, label, img_path

def parse_args():
    parser = argparse.ArgumentParser(description='Generate predictions from trained model')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the trained model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Directory containing the dataset')
    parser.add_argument('--csv_path', type=str, required=True, 
                        help='Path to the CSV file with image labels')
    parser.add_argument('--output_dir', type=str, default='model_predictions',
                        help='Directory to save the predictions')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for inference')
    parser.add_argument('--image_size', type=int, default=384,
                        help='Image size for model input')
    parser.add_argument('--num_classes', type=int, default=14,
                        help='Number of classes')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for computation (cuda/cpu)')
    return parser.parse_args()

def prepare_data(args):
    """Prepare data for prediction."""
    print("Preparing data...")
    
    # Read CSV file
    data_df = pd.read_csv(args.csv_path)
    
    # Define class names
    class_names = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 
        'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]
    
    # Extract labels
    data_df['Finding Labels'] = data_df['Finding Labels'].str.split('|')
    
    # Initialize arrays for labels
    all_labels = np.zeros((len(data_df), len(class_names)))
    
    # Fill in the labels
    for i, findings in enumerate(data_df['Finding Labels']):
        for finding in findings:
            if finding in class_names:
                class_idx = class_names.index(finding)
                all_labels[i, class_idx] = 1
                
    # Get image paths
    image_paths = []
    valid_indices = []
    
    for i, img_path in enumerate(data_df['Image Index']):
        full_path = os.path.join(args.data_dir, 'images', img_path)
        if os.path.exists(full_path):
            image_paths.append(full_path)
            valid_indices.append(i)
        
    # Filter labels to only include valid images
    labels = all_labels[valid_indices]
    
    print(f"Found {len(image_paths)} valid images")
    return image_paths, labels, class_names

def setup_model(args):
    """Set up the model for prediction."""
    print(f"Setting up model from {args.model_path}...")
    
    # Initialize model
    model = IntegratedConvNextModel(
        num_classes=args.num_classes,
        freeze_ratio=0.0  # No freezing for inference
    )
    
    # Load model checkpoint
    checkpoint = torch.load(args.model_path, map_location=args.device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(args.device)
    model.eval()
    
    return model

def run_predictions(model, image_paths, labels, class_names, args):
    """Run predictions on the dataset."""
    print("Running predictions...")
    
    # Set up transform
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloader
    dataset = NIHChestXrayDataset(image_paths, labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Store predictions, true labels, and image paths
    all_preds = []
    all_labels = []
    all_image_paths = []
    
    with torch.no_grad():
        for images, batch_labels, batch_paths in tqdm(dataloader, desc="Predicting"):
            images = images.to(args.device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            all_preds.append(probs)
            all_labels.append(batch_labels.numpy())
            all_image_paths.extend(batch_paths)
    
    # Concatenate results
    predictions = np.vstack(all_preds)
    true_labels = np.vstack(all_labels)
    
    return predictions, true_labels, all_image_paths

def calculate_metrics(true_labels, predictions, class_names):
    """Calculate performance metrics."""
    # Calculate AUROC for each class
    auroc_scores = {}
    for i, class_name in enumerate(class_names):
        try:
            if len(np.unique(true_labels[:, i])) > 1:  # Skip if all samples have the same label
                auroc = roc_auc_score(true_labels[:, i], predictions[:, i])
                auroc_scores[class_name] = auroc
            else:
                print(f"Warning: All samples have the same label for {class_name}, skipping AUROC calculation")
                auroc_scores[class_name] = float('nan')
        except Exception as e:
            print(f"Error calculating AUROC for {class_name}: {e}")
            auroc_scores[class_name] = float('nan')
    
    # Calculate mean AUROC
    valid_aurocs = [score for score in auroc_scores.values() if not np.isnan(score)]
    mean_auroc = np.mean(valid_aurocs) if valid_aurocs else float('nan')
    
    return auroc_scores, mean_auroc

def save_predictions(predictions, true_labels, image_paths, class_names, metrics, args):
    """Save predictions and metrics to files."""
    print(f"Saving predictions to {args.output_dir}...")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract model name from model path
    model_name = os.path.basename(args.model_path).split('.')[0]
    
    # Create DataFrame for predictions
    results_df = pd.DataFrame()
    
    # Add image paths
    results_df['image_path'] = [os.path.basename(path) for path in image_paths]
    
    # Add true labels and predictions
    for i, class_name in enumerate(class_names):
        results_df[f'true_label_{class_name}'] = true_labels[:, i]
        results_df[f'pred_prob_{class_name}'] = predictions[:, i]
    
    # Save predictions to CSV
    prediction_path = os.path.join(args.output_dir, f'{model_name}_predictions.csv')
    results_df.to_csv(prediction_path, index=False)
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, f'{model_name}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'class_auroc': metrics[0],
            'mean_auroc': float(metrics[1])
        }, f, indent=4)
    
    print(f"Predictions saved to {prediction_path}")
    print(f"Metrics saved to {metrics_path}")
    
    # Print mean AUROC
    print(f"Mean AUROC: {metrics[1]:.4f}")
    
    # Print AUROC for each class
    print("AUROC by class:")
    for class_name, auroc in sorted(metrics[0].items(), key=lambda x: x[1], reverse=True):
        if not np.isnan(auroc):
            print(f"  {class_name}: {auroc:.4f}")

def main():
    args = parse_args()
    
    # Prepare data
    image_paths, labels, class_names = prepare_data(args)
    
    # Set up model
    model = setup_model(args)
    
    # Run predictions
    predictions, true_labels, image_paths = run_predictions(model, image_paths, labels, class_names, args)
    
    # Calculate metrics
    metrics = calculate_metrics(true_labels, predictions, class_names)
    
    # Save predictions and metrics
    save_predictions(predictions, true_labels, image_paths, class_names, metrics, args)

if __name__ == '__main__':
    main() 