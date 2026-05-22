import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Define constants
DISEASE_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
    'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia',
    'No Finding'
]

class SimpleXRayDataset(Dataset):
    """Minimal dataset class for chest X-rays"""
    
    def __init__(self, data_dir, image_list, labels, transforms=None):
        self.data_dir = data_dir
        self.image_list = image_list
        self.labels = labels
        self.transforms = transforms
        self.image_names = [os.path.basename(img) for img in self.image_list]
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_folder = f"images_{int(img_name.split('_')[0]) // 1000:03d}"
        img_path = os.path.join(self.data_dir, img_folder, "images", img_name)
        
        try:
            # Try to load image
            image = Image.open(img_path).convert('RGB')
            if self.transforms:
                image = self.transforms(image)
        except Exception as e:
            # Return black image if error
            print(f"Error loading image {img_path}: {e}")
            image = torch.zeros((3, 224, 224))
        
        return {
            'image': image,
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32)
        }

def create_model(num_classes=15):
    """Create a pretrained ResNet18 model"""
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def train_loop(model, train_loader, val_loader, args):
    """Run the training loop"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Training loop
    best_auroc = 0.0
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        train_pbar = tqdm(train_loader, desc="Training")
        for batch in train_pbar:
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item()
            train_steps += 1
            train_pbar.set_postfix({'loss': loss.item()})
        
        # Calculate average training loss
        train_loss = train_loss / train_steps
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0
        all_labels = []
        all_outputs = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation")
            for batch in val_pbar:
                images = batch['image'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Update statistics
                val_loss += loss.item()
                val_steps += 1
                val_pbar.set_postfix({'loss': loss.item()})
                
                # Collect predictions and labels
                all_labels.append(labels.cpu().numpy())
                all_outputs.append(torch.sigmoid(outputs).cpu().numpy())
        
        # Calculate average validation loss
        val_loss = val_loss / val_steps
        
        # Calculate AUROC
        all_labels = np.vstack(all_labels)
        all_outputs = np.vstack(all_outputs)
        
        aurocs = []
        for i in range(len(DISEASE_CLASSES)):
            if len(np.unique(all_labels[:, i])) > 1:
                auroc = roc_auc_score(all_labels[:, i], all_outputs[:, i])
                aurocs.append(auroc)
                print(f"{DISEASE_CLASSES[i]}: {auroc:.4f}")
        
        mean_auroc = np.mean(aurocs)
        
        # Print results
        print(f"Epoch {epoch+1} results:")
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Mean AUROC: {mean_auroc:.4f}")
        
        # Save best model
        if mean_auroc > best_auroc:
            best_auroc = mean_auroc
            save_path = os.path.join(args.output_dir, f"best_model_auroc_{best_auroc:.3f}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")
    
    print("\nTraining completed!")
    print(f"Best AUROC: {best_auroc:.4f}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train a ResNet18 model on chest X-rays')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--train_val_list', type=str, required=True, help='Path to train/val list')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--use_weighted_sampler', action='store_true', help='Use weighted sampler')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(args.csv_path)
    
    with open(args.train_val_list, 'r') as f:
        image_list = [line.strip() for line in f.readlines()]
    
    # Filter image list to include only images in dataframe
    image_list = [img for img in image_list if img in df['Image Index'].values]
    
    # Balance dataset by reducing No Finding cases
    no_finding_cases = df[df['Finding Labels'] == 'No Finding']
    other_cases = df[df['Finding Labels'] != 'No Finding']
    no_finding_sample = no_finding_cases.sample(n=min(len(no_finding_cases) // 5, len(other_cases)), random_state=42)
    df = pd.concat([other_cases, no_finding_sample])
    
    # Filter image list again after balancing
    image_list = [img for img in image_list if img in df['Image Index'].values]
    
    # Prepare labels
    labels = []
    for img in image_list:
        findings = df[df['Image Index'] == img]['Finding Labels'].iloc[0].split('|')
        label = [1 if disease in findings else 0 for disease in DISEASE_CLASSES]
        labels.append(label)
    
    # Split data
    indices = list(range(len(image_list)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    # Set up transformations
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = SimpleXRayDataset(
        args.data_dir,
        [image_list[i] for i in train_idx],
        [labels[i] for i in train_idx],
        train_transforms
    )
    
    val_dataset = SimpleXRayDataset(
        args.data_dir,
        [image_list[i] for i in val_idx],
        [labels[i] for i in val_idx],
        val_transforms
    )
    
    # Create sampler for class imbalance if needed
    if args.use_weighted_sampler:
        class_counts = np.sum([labels[i] for i in train_idx], axis=0)
        weights = 1.0 / np.clip(class_counts, 5, len(train_idx))
        
        sample_weights = np.zeros(len(train_idx))
        for i, idx in enumerate(train_idx):
            for j, has_class in enumerate(labels[idx]):
                if has_class:
                    sample_weights[i] += weights[j]
        
        sampler = WeightedRandomSampler(
            torch.DoubleTensor(sample_weights),
            len(sample_weights)
        )
    else:
        sampler = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = create_model(num_classes=len(DISEASE_CLASSES))
    
    # Train model
    train_loop(model, train_loader, val_loader, args)

if __name__ == '__main__':
    main() 