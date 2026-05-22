import os
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import argparse

# Import model and constants
from advanced_model import (
    AdvancedXrayModel, 
    CLASS_NAMES, 
    TRAIN_CLASSES
)

def preprocess_image(image_path, image_size=512):
    """Preprocess image for model input"""
    # Read image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Apply transforms
    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
    return image_tensor

def preprocess_metadata(age, gender, view):
    """Preprocess patient metadata"""
    # Normalize age to 0-1 range
    age = float(age) / 100.0
    
    # Gender encoding (0 for female, 1 for male)
    gender_encoded = 1.0 if gender.lower() in ['m', 'male'] else 0.0
    
    # View position encoding
    view_positions = {
        'PA': 0, 'AP': 1, 'L': 2, 'LATERAL': 2,
        'AP SUPINE': 3, 'AP_SUPINE': 3, 'SUPINE': 3
    }
    view_encoded = view_positions.get(view.upper(), 0) / 3.0  # Normalize to 0-1
    
    # Combine metadata
    metadata = torch.tensor([[age, gender_encoded, view_encoded]], dtype=torch.float32)
    return metadata

def load_model(checkpoint_path, device='cuda'):
    """Load the trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}")
    
    # Create model
    model = AdvancedXrayModel(
        num_classes=len(CLASS_NAMES),
        metadata_features=3,
        dropout_rate=0.0  # No dropout for inference
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set model to evaluation mode
    model.eval()
    model = model.to(device)
    
    # Optional: print model information
    if 'class_aurocs' in checkpoint:
        aurocs = checkpoint['class_aurocs']
        print("Model performance (AUROC):")
        for cls, score in aurocs.items():
            print(f"  {cls}: {score:.4f}")
    
    return model

def predict(model, image_tensor, metadata_tensor, threshold=0.5, device='cuda'):
    """Make prediction with the model"""
    # Move tensors to device
    image_tensor = image_tensor.to(device)
    metadata_tensor = metadata_tensor.to(device)
    
    # Get prediction
    with torch.no_grad(), torch.cuda.amp.autocast():
        _, combined_out = model(image_tensor, metadata_tensor)
        probabilities = torch.sigmoid(combined_out)[0]
    
    # Move predictions to CPU
    probabilities = probabilities.cpu().numpy()
    
    # Get predictions above threshold
    predictions = probabilities >= threshold
    
    # Create results
    results = []
    for i, cls_name in enumerate(TRAIN_CLASSES):
        results.append({
            'disease': cls_name,
            'probability': float(probabilities[i]),
            'predicted': bool(predictions[i])
        })
    
    # Sort by probability (highest first)
    results.sort(key=lambda x: x['probability'], reverse=True)
    
    return results

def display_results(image_path, results, threshold=0.5):
    """Display the image and prediction results"""
    # Read and display the image
    img = Image.open(image_path).convert('RGB')
    
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Chest X-ray')
    plt.axis('off')
    
    # Display predictions
    plt.subplot(1, 2, 2)
    
    diseases = [r['disease'] for r in results]
    probabilities = [r['probability'] for r in results]
    
    # Only show diseases above threshold with darker colors
    colors = ['darkred' if p >= threshold else 'lightgray' for p in probabilities]
    
    # Sort in descending order for display
    plt.barh(range(len(diseases)), probabilities, color=colors)
    plt.yticks(range(len(diseases)), diseases)
    plt.xlabel('Probability')
    plt.title(f'Predicted Diseases (Threshold: {threshold})')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    output_dir = "predictions"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(image_path).replace('.', '_pred.'))
    plt.savefig(output_path)
    print(f"Prediction visualization saved to {output_path}")
    
    # Show the plot
    plt.show()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test chest X-ray disease prediction model')
    parser.add_argument('--image', type=str, required=True, help='Path to the X-ray image')
    parser.add_argument('--age', type=float, required=True, help='Patient age (years)')
    parser.add_argument('--gender', type=str, required=True, choices=['M', 'F'], help='Patient gender (M/F)')
    parser.add_argument('--view', type=str, required=True, help='View position (PA, AP, LATERAL, etc.)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Prediction threshold (default: 0.5)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint (default: best model)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Select model checkpoint
    if args.checkpoint is None:
        # Find the best model in advanced_outputs directory
        output_dir = "advanced_outputs"
        best_model = None
        best_auroc = 0.0
        
        for file in os.listdir(output_dir):
            if file.startswith("best_model_auroc_"):
                try:
                    auroc = float(file.split("_")[-1].split(".")[0])
                    if auroc > best_auroc:
                        best_auroc = auroc
                        best_model = file
                except:
                    continue
        
        if best_model:
            checkpoint_path = os.path.join(output_dir, best_model)
            print(f"Using best model with AUROC {best_auroc}")
        else:
            checkpoint_path = os.path.join(output_dir, "checkpoint_latest.pth")
            print("No best model found, using latest checkpoint")
    else:
        checkpoint_path = args.checkpoint
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    # Load model
    device = args.device
    model = load_model(checkpoint_path, device)
    
    # Process image
    image_tensor = preprocess_image(args.image)
    
    # Process metadata
    metadata_tensor = preprocess_metadata(args.age, args.gender, args.view)
    
    # Make prediction
    results = predict(model, image_tensor, metadata_tensor, args.threshold, device)
    
    # Display results
    print("\nPrediction Results:")
    print("-" * 50)
    print(f"{'Disease':<20} {'Probability':<15} {'Predicted':<10}")
    print("-" * 50)
    
    for result in results:
        print(f"{result['disease']:<20} {result['probability']:.4f}{' ':<15} {'Yes' if result['predicted'] else 'No':<10}")
    
    # Show visualization
    display_results(args.image, results, args.threshold)
    
    # Summarize findings
    predicted_diseases = [r['disease'] for r in results if r['predicted']]
    if predicted_diseases:
        print(f"\nSummary: The model predicts the patient has: {', '.join(predicted_diseases)}")
    else:
        print("\nSummary: The model does not predict any diseases above the threshold")

if __name__ == "__main__":
    main() 