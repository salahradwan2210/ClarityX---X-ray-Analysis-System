"""
Script for making predictions with a trained ConvNext model for chest X-ray classification.
This fixed version doesn't depend on external data_utils module.
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from sklearn.metrics import roc_auc_score

# Define disease classes directly
DISEASE_CLASSES_14 = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
    'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

DISEASE_CLASSES_15 = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
    'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia',
    'No Finding'
]

class Compose:
    """Composes several transforms together."""
    
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, image):
        for transform in self.transforms:
            image = transform(image)
        return image


class ResizeImage:
    """Resize an image to the given size."""
    
    def __init__(self, size):
        self.size = size
        
    def __call__(self, image):
        return image.resize((self.size, self.size), Image.BILINEAR)


class ToTensor:
    """Convert a PIL Image to tensor."""
    
    def __call__(self, image):
        # Convert PIL Image to numpy array
        image = np.array(image).astype(np.float32) / 255.0
        # Convert numpy array to tensor and permute dimensions
        image = torch.tensor(image).permute(2, 0, 1)
        return image


class NormalizeImage:
    """Normalize an image with mean and std."""
    
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
        
    def __call__(self, image):
        # If image is tensor, perform normalization
        if isinstance(image, torch.Tensor):
            mean = torch.tensor(self.mean).view(3, 1, 1)
            std = torch.tensor(self.std).view(3, 1, 1)
            return (image - mean) / std
            
        # If image is PIL, convert to numpy and normalize
        image = np.array(image).astype(np.float32) / 255.0
        mean = np.array(self.mean)
        std = np.array(self.std)
        
        # Apply normalization
        for i in range(3):
            image[:, :, i] = (image[:, :, i] - self.mean[i]) / self.std[i]
            
        return image


def load_metadata(csv_path):
    """Load metadata from the CSV file."""
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded metadata with {len(df)} entries")
        return df
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None


def create_transform(input_size):
    """Create a transform pipeline for the images."""
    return Compose([
        ResizeImage(input_size),
        ToTensor(),
        NormalizeImage()
    ])


def prepare_demographic_data(metadata_row):
    """Extract and prepare demographic data from metadata."""
    if metadata_row is None:
        return None
    
    # Example: if we have age and gender data
    try:
        return None  # Currently not using demographic data
    except:
        return None


def load_model(model_path, num_classes=14, model_variant='base', input_size=256, device='cuda'):
    """Load a trained model from path."""
    try:
        # Import model module only when needed
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from models.convnext_large_model import IntegratedConvNextModel
        
        print(f"Loading model with {num_classes} classes")
        model = IntegratedConvNextModel(
            num_classes=num_classes,
            pretrained=False,
            model_variant=model_variant,
            input_size=input_size
        ).to(device)
        
        # Load the saved state
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        print(f"Model loaded from {model_path}")
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def predict_image(image_path, model, transform, metadata=None, device='cuda', fp16=False):
    """Make a prediction for a single image."""
    try:
        # Load and transform image
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            if fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(img_tensor)
            else:
                outputs = model(img_tensor)
                
        # Get probabilities
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
        
        return probs, img
        
    except Exception as e:
        print(f"Error predicting image {image_path}: {e}")
        return None, None


class PredictionDataset(Dataset):
    """Dataset for batch prediction."""
    
    def __init__(self, data_dir, csv_path, num_classes=14, input_size=256, use_test_list=None):
        """
        Initialize the prediction dataset.
        
        Args:
            data_dir: Directory containing images
            csv_path: Path to metadata CSV
            num_classes: Number of disease classes
            input_size: Size for resizing images
            use_test_list: Optional list of image files to use
        """
        self.data_dir = data_dir
        self.input_size = input_size
        self.transform = create_transform(input_size)
        self.metadata = load_metadata(csv_path)
        
        # Determine which classes to use
        self.classes = DISEASE_CLASSES_14 if num_classes == 14 else DISEASE_CLASSES_15
        self.num_classes = num_classes
        
        # Get list of image files
        if use_test_list:
            with open(use_test_list, 'r') as f:
                self.image_files = [line.strip() for line in f.readlines()]
        else:
            self.image_files = [f for f in os.listdir(data_dir) 
                               if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Found {len(self.image_files)} images for prediction")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        
        # Load image
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img)
            
            # Get metadata if available
            demo_data = None
            if self.metadata is not None:
                meta_row = self.metadata[self.metadata['Image Index'] == img_name]
                if not meta_row.empty:
                    demo_data = prepare_demographic_data(meta_row.iloc[0])
            
            return {
                'image': img_tensor,
                'image_path': img_path,
                'image_name': img_name,
                'demographic_data': demo_data
            }
        
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank tensor as fallback
            return {
                'image': torch.zeros(3, self.input_size, self.input_size),
                'image_path': img_path,
                'image_name': img_name,
                'demographic_data': None
            }


def visualize_predictions(probabilities, image, threshold=0.5, output_path=None, show=False, classes=None):
    """Visualize predictions with probability bars."""
    if classes is None:
        classes = DISEASE_CLASSES_14  # Default to 14 classes
        
    # Convert image from tensor to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
        
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Display image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Chest X-ray')
    plt.axis('off')
    
    # Display probabilities
    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(classes))
    
    # Sort probabilities
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_probs = probabilities[sorted_indices]
    sorted_classes = [classes[i] for i in sorted_indices]
    
    # Color bars based on threshold
    colors = ['green' if p >= threshold else 'blue' for p in sorted_probs]
    
    # Plot horizontal bars
    plt.barh(y_pos, sorted_probs, align='center', color=colors)
    plt.yticks(y_pos, sorted_classes)
    plt.xlabel('Probability')
    plt.title('Disease Probabilities')
    
    # Add threshold line
    plt.axvline(x=threshold, color='r', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the figure if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200)
        print(f"Visualization saved to {output_path}")
    
    # Show the figure if requested
    if show:
        plt.show()
    
    plt.close()


def predict_and_visualize(model, dataloader, device, output_dir, threshold=0.5, save_csv=True, num_classes=14):
    """Predict and visualize results for a batch of images."""
    os.makedirs(output_dir, exist_ok=True)
    classes = DISEASE_CLASSES_14 if num_classes == 14 else DISEASE_CLASSES_15
    
    all_probs = []
    all_imgs = []
    all_names = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            # Get data
            images = batch['image'].to(device)
            paths = batch['image_path']
            names = batch['image_name']
            
            # Run inference with FP16 if available
            with torch.cuda.amp.autocast():
                outputs = model(images)
            
            # Get probabilities
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            # Save results
            for i in range(len(probs)):
                img_path = paths[i]
                img_name = names[i]
                prob = probs[i]
                
                # Load image for visualization
                img = Image.open(img_path).convert('RGB')
                
                # Visualize
                vis_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_pred.png")
                visualize_predictions(prob, img, threshold, vis_path, show=False, classes=classes)
                
                # Store for CSV
                all_probs.append(prob)
                all_imgs.append(img_path)
                all_names.append(img_name)
    
    # Save results to CSV
    if save_csv:
        results = {
            'Image': all_names,
            'Path': all_imgs
        }
        
        # Add probabilities for each class
        for i, cls in enumerate(classes):
            results[f'{cls}_prob'] = [p[i] for p in all_probs]
            results[f'{cls}_pred'] = [1 if p[i] >= threshold else 0 for p in all_probs]
        
        # Create DataFrame and save
        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, 'predictions.csv')
        df.to_csv(csv_path, index=False)
        print(f"Predictions saved to {csv_path}")
        
        return df


def main():
    parser = argparse.ArgumentParser(description='Predict using trained model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with test images')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to metadata CSV')
    parser.add_argument('--output_dir', type=str, default='predictions', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for positive prediction')
    parser.add_argument('--input_size', type=int, default=320, help='Input image size')
    parser.add_argument('--model_variant', type=str, default='base', help='Model variant (base, large)')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision')
    parser.add_argument('--num_classes', type=int, default=14, help='Number of classes')
    parser.add_argument('--test_list', type=str, default=None, help='Optional list of test images')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(
        args.model_path, 
        num_classes=args.num_classes,
        model_variant=args.model_variant,
        input_size=args.input_size,
        device=device
    )
    
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Create dataset and dataloader
    dataset = PredictionDataset(
        args.data_dir,
        args.csv_path,
        num_classes=args.num_classes,
        input_size=args.input_size,
        use_test_list=args.test_list
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == 'cuda')
    )
    
    # Run prediction and visualization
    predict_and_visualize(
        model,
        dataloader,
        device,
        args.output_dir,
        args.threshold,
        save_csv=True,
        num_classes=args.num_classes
    )
    
    print("Prediction completed!")


if __name__ == '__main__':
    main() 