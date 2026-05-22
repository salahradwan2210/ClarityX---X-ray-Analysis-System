"""
Fixed prediction script for chest X-ray classification based on predict_convnext.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import sys
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Add the project root to the path to import the model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.convnext_large_model import IntegratedConvNextModel

# Define disease classes directly
DISEASE_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 
    'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

# Simple image transformation classes
class Compose:
    """Composes several transforms together."""
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image):
        for transform in self.transforms:
            image = transform(image)
        return image

class ResizeImage:
    """Resize an image to given size."""
    def __init__(self, size):
        self.size = size
    def __call__(self, image):
        return image.resize((self.size, self.size), Image.BILINEAR)

class ToTensor:
    """Convert PIL Image to tensor."""
    def __call__(self, image):
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.tensor(image).permute(2, 0, 1)
        return image

class NormalizeImage:
    """Normalize image with mean and std."""
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            mean = torch.tensor(self.mean).view(3, 1, 1)
            std = torch.tensor(self.std).view(3, 1, 1)
            return (image - mean) / std
        image = np.array(image).astype(np.float32) / 255.0
        for i in range(3):
            image[:, :, i] = (image[:, :, i] - self.mean[i]) / self.std[i]
        return image

def load_metadata(csv_path):
    """Load metadata from CSV file"""
    print(f"Loading metadata from {csv_path}")
    df = pd.read_csv(csv_path)
    # Convert Image Index to lowercase for Windows compatibility
    df['Image Index'] = df['Image Index'].apply(lambda x: x.lower())
    
    # Extract demographic data
    df['Patient Age'] = df['Patient Age'].astype(int)
    df['Patient Gender'] = df['Patient Gender'].map({'M': 0, 'F': 1})
    
    # Process Finding Labels
    df['Finding Labels'] = df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
    
    return df

def create_transform(input_size):
    """Create image transformation pipeline"""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def prepare_demographic_data(metadata_row):
    """Prepare demographic data for model"""
    age = int(metadata_row['Patient Age'])
    gender = int(metadata_row['Patient Gender'])
    
    # Normalize age to [0, 1]
    age_normalized = age / 100.0
    
    return torch.tensor([age_normalized, gender], dtype=torch.float32).unsqueeze(0)

def load_model(model_path, num_classes=14, model_variant='base', input_size=256, device='cuda'):
    """Load the trained model"""
    print(f"Loading model from {model_path}")
    # Initialize model with proper parameters
    model = IntegratedConvNextModel(
        num_classes=num_classes,
        pretrained=False,
        model_variant=model_variant,
        input_size=input_size
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model

def predict_image(image_path, model, transform, metadata=None, device='cuda', fp16=False):
    """Predict diseases from a single image"""
    # Load and transform image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Prepare demographic data if available
    demo_data = None
    if metadata is not None:
        # Extract filename without extension and path
        filename = os.path.basename(image_path).lower()
        
        # Find matching row in metadata
        metadata_row = metadata[metadata['Image Index'] == filename]
        
        if not metadata_row.empty:
            demo_data = prepare_demographic_data(metadata_row.iloc[0])
            demo_data = demo_data.to(device)
    
    # Perform prediction with mixed precision if enabled
    with torch.no_grad():
        if fp16:
            with torch.cuda.amp.autocast():
                if demo_data is not None:
                    outputs = model(img_tensor, demo_data)
                else:
                    outputs = model(img_tensor)
        else:
            if demo_data is not None:
                outputs = model(img_tensor, demo_data)
            else:
                outputs = model(img_tensor)
    
    # Get probabilities using sigmoid
    probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
    
    return probabilities, img

class PredictionDataset(Dataset):
    def __init__(self, data_dir, csv_path, bbox_path=None, input_size=256, use_age_gender=True):
        """Dataset for making predictions using the trained model"""
        self.data_dir = data_dir
        self.input_size = input_size
        self.use_age_gender = use_age_gender
        
        # Define disease labels
        self.labels = DISEASE_CLASSES
        
        # Image transformations
        self.transforms = Compose([
            ResizeImage(input_size),
            NormalizeImage(),
            ToTensor()
        ])
        
        # Load and process metadata
        df = pd.read_csv(csv_path)
        
        # Extract image paths and ensure they exist
        self.image_paths = []
        self.image_names = []
        self.label_data = []
        self.gender_data = []
        self.age_data = []
        self.has_labels = False
        
        if 'Finding Labels' in df.columns:
            self.has_labels = True
            
        for _, row in df.iterrows():
            img_path = os.path.join(data_dir, row['Image Index'])
            if os.path.exists(img_path):
                self.image_paths.append(img_path)
                self.image_names.append(row['Image Index'])
                
                # Process demographic data if available
                gender = -1
                age = -1
                if 'Patient Gender' in df.columns and self.use_age_gender:
                    gender = 1 if row['Patient Gender'] == 'M' else 0
                if 'Patient Age' in df.columns and self.use_age_gender:
                    age = float(row['Patient Age'])
                
                self.gender_data.append(gender)
                self.age_data.append(age)
                
                # Process disease labels if available
                if self.has_labels:
                    labels = [0] * len(self.labels)
                    findings = row['Finding Labels'].split('|')
                    for finding in findings:
                        if finding in self.labels:
                            idx = self.labels.index(finding)
                            labels[idx] = 1
                    self.label_data.append(labels)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transforms(img)
        
        # Get demographics
        gender = self.gender_data[idx]
        age = self.age_data[idx]
        
        # Create result dictionary
        result = {
            'image': img_tensor,
            'image_path': img_path,
            'image_name': self.image_names[idx],
            'gender': gender,
            'age': age
        }
        
        # Add labels if available
        if self.has_labels:
            result['labels'] = torch.tensor(self.label_data[idx], dtype=torch.float32)
        
        return result

def visualize_predictions(probabilities, image, threshold=0.5, output_path=None, show=False):
    """Visualize prediction results with a bar chart of probabilities"""
    # Convert probabilities to binary predictions
    predictions = (probabilities >= threshold).astype(int)
    
    # Create figure with adjusted size
    plt.figure(figsize=(12, 8))
    
    # Plot the image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('X-ray Image')
    plt.axis('off')
    
    # Plot probability bars
    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(DISEASE_CLASSES))
    
    # Sort probabilities in descending order
    sorted_indices = np.argsort(-probabilities)
    sorted_probs = probabilities[sorted_indices]
    sorted_classes = [DISEASE_CLASSES[i] for i in sorted_indices]
    sorted_preds = predictions[sorted_indices]
    
    # Plot bars with color based on prediction
    bars = plt.barh(y_pos, sorted_probs, align='center')
    for i, bar in enumerate(bars):
        if sorted_preds[i] == 1:
            bar.set_color('red')
    
    plt.yticks(y_pos, sorted_classes)
    plt.xlabel('Probability')
    plt.title('Disease Probabilities')
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    plt.legend()
    plt.grid(axis='x')
    plt.tight_layout()
    
    # Save figure if output path is specified
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    # Show figure if requested
    if show:
        plt.show()
    else:
        plt.close()

def predict_and_visualize(model, dataloader, device, output_dir, threshold=0.5, save_heatmaps=False, model_variant='base', input_size=256):
    """Run prediction on all images in the dataloader and visualize results"""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    disease_labels = DISEASE_CLASSES
    print(f"Using {len(disease_labels)} disease classes: {disease_labels}")
    
    all_preds = []
    all_image_names = []
    all_labels = []
    has_labels = False
    
    # Process all batches
    for batch in tqdm(dataloader, desc="Predicting"):
        images = batch['image'].to(device)
        image_names = batch['image_name']
        all_image_names.extend(image_names)
        
        # Check if labels are available
        if 'labels' in batch:
            has_labels = True
            labels = batch['labels'].numpy()
            all_labels.append(labels)
        
        # Create demographic data tensor if available
        demographic_data = None
        if 'age' in batch and 'gender' in batch:
            age = batch['age'].unsqueeze(1) / 100.0  # Normalize age
            gender = batch['gender'].unsqueeze(1)
            demographic_data = torch.cat([age, gender], dim=1).to(device)
        
        # Run model prediction
        with torch.no_grad():
            if demographic_data is not None:
                outputs = model(images, demographic_data)
            else:
                outputs = model(images)
        
        # Convert outputs to probabilities
        probabilities = torch.sigmoid(outputs).cpu().numpy()
        all_preds.append(probabilities)
        
        # Visualize each prediction
        for i, img_name in enumerate(image_names):
            # Get original image
            img_path = batch['image_path'][i]
            orig_img = Image.open(img_path).convert('RGB')
            
            # Create visualization
            vis_path = os.path.join(output_dir, 'visualizations', f"{os.path.splitext(img_name)[0]}_prediction.png")
            visualize_predictions(probabilities[i], orig_img, threshold, vis_path)
            
            # Generate heatmap if requested
            if save_heatmaps:
                # Find top disease predictions
                top_indices = np.argsort(-probabilities[i])[:3]
                for disease_idx in top_indices:
                    if probabilities[i][disease_idx] >= threshold:
                        disease_name = disease_labels[disease_idx]
                        heatmap_path = os.path.join(output_dir, 'visualizations', 
                                    f"{os.path.splitext(img_name)[0]}_{disease_name}_heatmap.png")
                        
                        try:
                            # Skip heatmap generation to simplify this script
                            pass
                        except Exception as e:
                            print(f"Error generating heatmap for {img_name}, {disease_name}: {e}")
    
    # Concatenate all predictions
    all_preds = np.concatenate(all_preds, axis=0)
    
    # Save predictions to CSV
    results_df = pd.DataFrame({'Image': all_image_names})
    
    for i, disease in enumerate(disease_labels):
        results_df[disease] = all_preds[:, i]
    
    results_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    
    # Evaluate metrics if ground truth exists
    if has_labels:
        all_labels = np.concatenate(all_labels, axis=0)
        metrics_df = pd.DataFrame({'Disease': disease_labels})
        auroc_scores = []
        auprc_scores = []
        
        # Calculate metrics and create ROC and PR curves
        plt.figure(figsize=(15, 10))
        plt.subplot(1, 2, 1)
        for i, disease in enumerate(disease_labels):
            # Skip diseases with no positive samples
            if np.sum(all_labels[:, i]) == 0:
                auroc_scores.append(np.nan)
                auprc_scores.append(np.nan)
                continue
            
            try:
                # ROC curve
                auroc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                auroc_scores.append(auroc)
                
                # PR curve
                precision, recall, _ = precision_recall_curve(all_labels[:, i], all_preds[:, i])
                auprc = auc(recall, precision)
                auprc_scores.append(auprc)
                
                # Plot ROC curve
                plt.plot(recall, precision, label=f'{disease} (AUPRC: {auprc:.3f})')
            except Exception as e:
                print(f"Error calculating metrics for {disease}: {e}")
                auroc_scores.append(np.nan)
                auprc_scores.append(np.nan)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc='best')
        plt.grid(True)
        
        # Save metrics
        metrics_df['AUROC'] = auroc_scores
        metrics_df['AUPRC'] = auprc_scores
        metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
        
        # Save curves figure
        plt.savefig(os.path.join(output_dir, 'pr_curves.png'), bbox_inches='tight')
        plt.close()
        
        # Calculate and print average metrics
        avg_auroc = np.nanmean(auroc_scores)
        avg_auprc = np.nanmean(auprc_scores)
        print(f"Average AUROC: {avg_auroc:.4f}")
        print(f"Average AUPRC: {avg_auprc:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Make predictions with trained ConvNext model')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file with image metadata')
    parser.add_argument('--bbox_path', type=str, default=None, help='Path to CSV file with bounding box information')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='predictions', help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for predictions')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary predictions')
    parser.add_argument('--input_size', type=int, default=256, help='Input image size')
    parser.add_argument('--model_variant', type=str, default='base', choices=['base', 'large'], help='Model variant')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision for prediction')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for data loading')
    parser.add_argument('--save_heatmaps', action='store_true', help='Save heatmaps for visualizations')
    parser.add_argument('--num_classes', type=int, default=14, help='Number of classes to predict (14 or 15 including No Finding)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    dataset = PredictionDataset(
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        bbox_path=args.bbox_path,
        input_size=args.input_size,
        use_age_gender=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    print(f"Found {len(dataset)} images for prediction")
    
    # Load model with the specified number of classes
    model = IntegratedConvNextModel(
        num_classes=args.num_classes,
        pretrained=False,
        model_variant=args.model_variant,
        input_size=args.input_size
    ).to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        try:
            model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Trying to load with different state dict keys...")
            # Try with a different state dict key format
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                print("Failed to load checkpoint, no known state dict format found.")
                return
    
    model.eval()
    print(f"Loaded model from {args.model_path}")
    
    # Make predictions
    with torch.cuda.amp.autocast(enabled=args.fp16):
        predict_and_visualize(
            model=model,
            dataloader=dataloader,
            device=device,
            output_dir=args.output_dir,
            threshold=args.threshold,
            save_heatmaps=args.save_heatmaps,
            model_variant=args.model_variant,
            input_size=args.input_size
        )
    
    print(f"Predictions completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 