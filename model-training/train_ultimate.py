import os
import sys
sys.path.append('.')  # Add current directory to path

import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split

# Import models and helpers
from models.ultimate_convnext import UltimateConvNext
from utils.bbox_utils import load_bbox_data, get_image_bboxes
from utils.preprocessing import balance_dataset_weights, balance_dataset

# Define disease classes
DISEASE_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
    'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia',
    'No Finding'  # Added No Finding as a class
]

# CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class FocalLoss(torch.nn.Module):
    """Focal Loss for better handling of class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_with_logits = torch.nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, inputs, targets):
        bce_loss = self.bce_with_logits(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def create_data_loaders(args):
    """Create data loaders with enhanced augmentation"""
    print("Starting data loading...")
    
    # Advanced image transformations for better generalization
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),  # Smaller size for memory efficiency
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((256, 256)),  # Consistent with training
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and prepare data exactly as in the original script
    # ... [Same data loading logic as in train_convnext_large.py] ...
    
    print(f"Reading image list from {args.train_val_list}")
    with open(args.train_val_list, 'r') as f:
        image_list = [line.strip() for line in f.readlines()]
    print(f"Found {len(image_list)} images in list")
    
    # Load metadata
    print(f"Reading metadata from {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    
    # Balanced sampling of No Finding cases
    no_finding_cases = df[df['Finding Labels'] == 'No Finding']
    other_cases = df[df['Finding Labels'] != 'No Finding']
    
    # Use 20% of No Finding cases
    no_finding_sample_size = int(len(no_finding_cases) * 0.2)
    no_finding_cases = no_finding_cases.sample(n=no_finding_sample_size, random_state=42)
    
    # Combine cases
    df = pd.concat([other_cases, no_finding_cases]).reset_index(drop=True)
    print(f"Total cases after sampling No Finding: {len(df)}")
    
    # Filter image list
    image_list = [img for img in image_list if img in df['Image Index'].values]
    print(f"Found {len(image_list)} valid images")
    
    # Filter metadata
    train_val_df = df[df['Image Index'].isin(image_list)]
    print(f"Found {len(train_val_df)} records in metadata")
    
    # Load bounding box data
    print(f"Reading bounding box data from {args.bbox_path}")
    bbox_data = None
    if args.bbox_path:
        try:
            bbox_data = load_bbox_data(args.bbox_path)
            if bbox_data is not None:
                print(f"Loaded {len(bbox_data)} bounding box annotations")
        except Exception as e:
            print(f"Error loading bounding box data: {e}")
    
    # Prepare labels
    labels = []
    for img_name in tqdm(image_list, desc="Preparing data"):
        img_data = train_val_df[train_val_df['Image Index'] == img_name].iloc[0]
        findings = img_data['Finding Labels'].split('|')
        label = [1 if disease in findings else 0 for disease in DISEASE_CLASSES]
        labels.append(label)
    
    # Split data with stratification
    indices = list(range(len(image_list)))
    train_idx, valid_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=[l[-1] for l in labels]
    )
    
    # Create datasets using the original IntegratedDataset from train_convnext_large.py
    from train_convnext_large import IntegratedDataset
    
    train_dataset = IntegratedDataset(
        data_dir=args.data_dir,
        image_list=[image_list[i] for i in train_idx],
        labels=[labels[i] for i in train_idx],
        transforms=train_transforms,
        bbox_data=bbox_data,
        demographic_data=train_val_df,
        cache_images=args.cache_images
    )
    
    valid_dataset = IntegratedDataset(
        data_dir=args.data_dir, 
        image_list=[image_list[i] for i in valid_idx],
        labels=[labels[i] for i in valid_idx],
        transforms=val_transforms,
        bbox_data=bbox_data,
        demographic_data=train_val_df,
        cache_images=args.cache_images
    )
    
    # Create weighted sampler to handle class imbalance
    if args.use_weighted_sampler:
        print("Creating weighted sampler for balanced training...")
        train_labels = [labels[i] for i in train_idx]
        class_counts = np.sum(train_labels, axis=0)
        class_weights = 1.0 / np.clip(class_counts, 5, len(train_idx))
        weights = np.zeros(len(train_idx))
        
        for i, idx in enumerate(train_idx):
            for j, has_class in enumerate(labels[idx]):
                if has_class:
                    weights[i] += class_weights[j]
        
        weights = torch.DoubleTensor(weights)
        sampler = WeightedRandomSampler(weights, len(weights))
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch for stability
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, valid_loader

def train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, args):
    """Train for one epoch with stability improvements"""
    model.train()
    train_loss = 0.0
    train_steps = 0
    
    # Initialize progress bar
    train_pbar = tqdm(train_loader, desc="Training")
    
    for batch_idx, batch in enumerate(train_pbar):
        # Move data to device
        images = batch['image'].to(device)
        labels = batch['labels'].to(device)
        
        # Prepare demographic data
        demographics = None
        if all(k in batch for k in ['age', 'gender', 'view']):
            demographics = {
                'age': batch['age'].to(device),
                'gender': batch['gender'].to(device),
                'view': batch['view'].to(device)
            }
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast(device_type=device.type, enabled=args.fp16):
            outputs = model(images, demographics)
            
            # Use enhanced logits if available
            logits = outputs['enhanced_logits'] if demographics is not None else outputs['class_logits']
            
            # Calculate loss
            loss = criterion(logits, labels)
            
            # Add regularization if needed
            if args.l1_reg > 0:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss = loss + args.l1_reg * l1_norm
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Apply gradient clipping to prevent exploding gradients
        if args.clip_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        
        # Update weights with gradient scaling
        scaler.step(optimizer)
        scaler.update()
        
        # Update statistics
        train_loss += loss.item()
        train_steps += 1
        
        # Update progress bar
        train_pbar.set_postfix({'loss': loss.item()})
        
        # Clear some memory
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
    
    # Calculate average loss
    return train_loss / train_steps

def validate(model, valid_loader, criterion, device):
    """Validate model with AUROC calculation"""
    model.eval()
    valid_loss = 0.0
    valid_steps = 0
    all_labels = []
    all_outputs = []
    
    # Initialize progress bar
    valid_pbar = tqdm(valid_loader, desc="Validation")
    
    with torch.no_grad():
        for batch in valid_pbar:
            # Move data to device
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)
            
            # Prepare demographic data
            demographics = None
            if all(k in batch for k in ['age', 'gender', 'view']):
                demographics = {
                    'age': batch['age'].to(device),
                    'gender': batch['gender'].to(device),
                    'view': batch['view'].to(device)
                }
            
            # Forward pass
            outputs = model(images, demographics)
            
            # Use enhanced logits if available
            logits = outputs['enhanced_logits'] if demographics is not None else outputs['class_logits']
            
            # Calculate loss
            loss = criterion(logits, labels)
            
            # Update statistics
            valid_loss += loss.item()
            valid_steps += 1
            
            # Collect predictions and labels
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(torch.sigmoid(logits).cpu().numpy())
            
            # Update progress bar
            valid_pbar.set_postfix({'loss': loss.item()})
    
    # Calculate average loss
    valid_loss = valid_loss / valid_steps
    
    # Calculate AUROC
    all_labels = np.vstack(all_labels)
    all_outputs = np.vstack(all_outputs)
    
    aurocs = []
    for i in range(len(DISEASE_CLASSES)):
        try:
            if len(np.unique(all_labels[:, i])) > 1:
                auroc = roc_auc_score(all_labels[:, i], all_outputs[:, i])
                aurocs.append(auroc)
        except Exception as e:
            print(f"Error calculating AUROC for {DISEASE_CLASSES[i]}: {e}")
    
    mean_auroc = np.mean(aurocs)
    
    # Sort diseases by performance
    disease_aurocs = [(DISEASE_CLASSES[i], auroc) for i, auroc in enumerate(aurocs)]
    disease_aurocs.sort(key=lambda x: x[1], reverse=True)
    
    return valid_loss, mean_auroc, disease_aurocs

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train UltimateConvNext model')
    
    # Basic paths
    parser.add_argument('--data_dir', type=str, required=True, help='Path to image directory')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to metadata CSV file')
    parser.add_argument('--train_val_list', type=str, required=True, help='Path to train/val image list')
    parser.add_argument('--bbox_path', type=str, default=None, help='Path to bounding box annotations')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save results')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training')
    parser.add_argument('--use_focal_loss', action='store_true', help='Use focal loss')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument('--l1_reg', type=float, default=0.0, help='L1 regularization coefficient')
    
    # Optimizer and scheduler
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'], help='Optimizer')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['plateau', 'cosine', 'none'], help='LR scheduler')
    
    # Data options
    parser.add_argument('--cache_images', action='store_true', help='Cache images in memory')
    parser.add_argument('--use_weighted_sampler', action='store_true', help='Use weighted sampler')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, valid_loader = create_data_loaders(args)
    
    # Create model
    model = UltimateConvNext(num_classes=len(DISEASE_CLASSES), pretrained=True).to(device)
    
    # Print model summary
    print(f"Created UltimateConvNext model with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Load checkpoint if exists
    start_epoch = 0
    if args.checkpoint:
        try:
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Loaded checkpoint from epoch {start_epoch}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    
    # Define loss function
    if args.use_focal_loss:
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        print("Using Focal Loss")
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
        print("Using BCE Loss")
    
    # Define optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:  # sgd
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    
    # Define scheduler
    if args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=args.patience//2, verbose=True
        )
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs, eta_min=args.learning_rate / 10
        )
    else:
        scheduler = None
    
    # Gradient scaler for mixed precision
    scaler = GradScaler()
    
    # Training loop
    best_auroc = 0.0
    patience_counter = 0
    
    print(f"\n=== Starting training for {args.num_epochs} epochs ===\n")
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # Train one epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, args)
        
        # Validate
        valid_loss, mean_auroc, disease_aurocs = validate(model, valid_loader, criterion, device)
        
        # Update learning rate scheduler
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(mean_auroc)
            else:
                scheduler.step()
        
        # Print results
        print(f"\nEpoch {epoch + 1} results:")
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {valid_loss:.4f}")
        print(f"Mean AUROC: {mean_auroc:.4f}")
        
        # Print top and bottom performing diseases
        print("\nTop performing diseases:")
        for disease, auroc in disease_aurocs[:3]:
            print(f"{disease}: {auroc:.4f}")
        
        print("Bottom performing diseases:")
        for disease, auroc in disease_aurocs[-3:]:
            print(f"{disease}: {auroc:.4f}")
        
        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.2e}")
        
        # Save best model
        if mean_auroc > best_auroc:
            best_auroc = mean_auroc
            patience_counter = 0
            
            # Save model
            checkpoint_path = os.path.join(args.output_dir, f"best_model_auroc_{best_auroc:.3f}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_auroc': best_auroc
            }, checkpoint_path)
            
            # Save latest best model
            latest_path = os.path.join(args.output_dir, "latest_best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_auroc': best_auroc
            }, latest_path)
            
            print(f"Saved best model to {checkpoint_path}")
            print("Updated latest best model")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping after {patience_counter} epochs without improvement")
            break
        
        # Save checkpoint periodically
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_auroc': best_auroc
            }, checkpoint_path)
        
        # Clear memory
        torch.cuda.empty_cache()
    
    print("\nTraining complete!")
    print(f"Best validation AUROC: {best_auroc:.4f}")

if __name__ == '__main__':
    main() 