import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import LabelEncoder
import math
import argparse
import gc
import time
import random

# Import our advanced model
from advanced_model import (
    set_seed, CLASS_NAMES, TRAIN_CLASSES, AdvancedXrayDataset, 
    mixup_data, mixup_criterion, AdvancedXrayModel, advanced_loss
)

def preprocess_metadata(df):
    """Preprocess patient metadata (age, gender, view position)"""
    df['Patient Age'] = df['Patient Age'].clip(upper=100)
    df['Patient Age'] = df['Patient Age'].fillna(df['Patient Age'].mean())
    
    gender_encoder = LabelEncoder()
    df['Patient Gender'] = gender_encoder.fit_transform(df['Patient Gender'])
    
    view_encoder = LabelEncoder()
    df['View Position'] = view_encoder.fit_transform(df['View Position'])
    
    # Normalize age to 0-1 range
    df['Patient Age'] = df['Patient Age'] / 100.0
    
    return df, gender_encoder, view_encoder

def load_image_list(list_file_path):
    """Load list of image names from a text file"""
    with open(list_file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def calculate_class_weights(df):
    """Calculate class weights based on class distribution"""
    weights = []
    for cls in TRAIN_CLASSES:
        count = len(df[df['Finding Labels'].str.contains(cls, regex=False)])
        weights.append(1.0 / (count + 1e-6))
    weights = np.array(weights)
    # Normalize weights
    weights = weights / weights.sum() * len(TRAIN_CLASSES)
    # Print class weights
    print("\nClass weights:")
    for cls, weight in zip(TRAIN_CLASSES, weights):
        count = len(df[df['Finding Labels'].str.contains(cls, regex=False)])
        print(f"{cls}: {weight:.2f} (samples: {count})")
    return torch.FloatTensor(weights)

def analyze_dataset(df):
    """Analyze class distribution in dataset"""
    print("\nClass distribution in dataset:")
    for cls in TRAIN_CLASSES:
        count = len(df[df['Finding Labels'].str.contains(cls, regex=False)])
        print(f"{cls}: {count} samples ({count/len(df)*100:.1f}%)")
    print(f"\nTotal samples: {len(df)}\n")
    return df

def plot_training_progress(metrics, save_path='training_progress.png'):
    """Plot training metrics"""
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    epochs = list(range(1, len(metrics['train_loss'])+1))
    plt.plot(epochs, metrics['train_loss'], label='Training Loss')
    plt.plot(epochs, metrics['val_loss'], label='Validation Loss')
    plt.title('Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot AUROC
    plt.subplot(2, 2, 2)
    plt.plot(epochs, metrics['mean_auroc'], label='Mean AUROC', color='blue')
    plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90% Target')
    plt.title('Mean AUROC Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('AUROC')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot learning rate
    if 'lr' in metrics:
        plt.subplot(2, 2, 3)
        plt.plot(epochs, metrics['lr'], label='Learning Rate', color='green')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_class_metrics(class_aurocs, save_path='class_metrics.png'):
    """Plot metrics for each class"""
    plt.figure(figsize=(15, 6))
    classes = list(class_aurocs.keys())
    scores = list(class_aurocs.values())
    
    # Sort by AUROC value
    sorted_idx = np.argsort(scores)
    classes = [classes[i] for i in sorted_idx]
    scores = [scores[i] for i in sorted_idx]
    
    plt.barh(classes, scores, color='skyblue')
    plt.axvline(x=0.9, color='r', linestyle='--', alpha=0.5, label='90% Target')
    plt.title('AUROC Score per Class')
    plt.xlabel('AUROC Score')
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.5, axis='x')
    
    # Add value labels
    for i, v in enumerate(scores):
        plt.text(v, i, f'{v:.3f}', va='center')
    
    plt.savefig(save_path)
    plt.close()

class WarmupCosineScheduler:
    """Cosine LR schedule with warmup"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, warmup_start_lr=1e-5, verbose=True):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.verbose = verbose
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = -1
    
    def step(self, epoch=None):
        if epoch is None:
            self.current_epoch += 1
        else:
            self.current_epoch = epoch
        
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            factor = self.current_epoch / self.warmup_epochs
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.warmup_start_lr + factor * (self.base_lrs[i] - self.warmup_start_lr)
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1 + math.cos(math.pi * progress))
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.min_lr + factor * (self.base_lrs[i] - self.min_lr)
        
        if self.verbose:
            print(f"Epoch {self.current_epoch+1}: LR set to {self.optimizer.param_groups[0]['lr']:.6f}")
    
    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

def train_advanced(args):
    """Main training function"""
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        # Optimize CUDA
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    # Set random seeds for reproducibility
    set_seed(args.seed)
    
    # Load and prepare data
    print(f"Loading data from: {args.data_path}")
    valid_images = set(load_image_list(args.list_path))
    print(f"Found {len(valid_images)} images in list file")
    
    # Load CSV data
    df = pd.read_csv(args.csv_path)
    df = df[df['Image Index'].isin(valid_images)]
    if not args.include_no_finding:
        df = df[df['Finding Labels'] != 'No Finding']
    print(f"Total usable images: {len(df)}")
    
    # Preprocess metadata
    df, gender_encoder, view_encoder = preprocess_metadata(df)
    
    # Split into train/val
    train_df = df.sample(frac=0.8, random_state=args.seed)
    valid_df = df.drop(train_df.index)
    
    # Analyze datasets
    print("\nAnalyzing training set:")
    analyze_dataset(train_df)
    print("\nAnalyzing validation set:")
    analyze_dataset(valid_df)
    
    # Define augmentations
    train_transform = A.Compose([
        A.Resize(args.image_size, args.image_size),
        A.Rotate(limit=20, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.5),
        ], p=0.5),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    valid_transform = A.Compose([
        A.Resize(args.image_size, args.image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Create datasets
    train_dataset = AdvancedXrayDataset(args.data_path, train_df, transform=train_transform, train=True)
    valid_dataset = AdvancedXrayDataset(args.data_path, valid_df, transform=valid_transform, train=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # Create model
    model = AdvancedXrayModel(
        num_classes=len(CLASS_NAMES), 
        metadata_features=3, 
        dropout_rate=args.dropout
    )
    model = model.to(device, memory_format=torch.channels_last)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'model' in n], 'lr': args.base_lr},
        {'params': [p for n, p in model.named_parameters() if 'model' not in n], 'lr': args.head_lr}
    ], weight_decay=args.weight_decay)
    
    # Initialize metrics tracking
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'mean_auroc': [],
        'lr': []
    }
    
    # Variables for loading checkpoint
    start_epoch = 0
    best_auroc = 0.0
    
    # Check if we need to load from checkpoint
    if hasattr(args, 'checkpoint_path') and os.path.exists(args.checkpoint_path) and not args.start_from_scratch:
        print(f"Loading checkpoint from {args.checkpoint_path}")
        try:
            checkpoint = torch.load(args.checkpoint_path, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Manually update optimizer learning rates if needed
            for i, param_group in enumerate(optimizer.param_groups):
                if i == 0:  # Base layers
                    param_group['lr'] = args.base_lr
                else:  # Head layers
                    param_group['lr'] = args.head_lr
            
            start_epoch = checkpoint['epoch'] + 1
            best_auroc = checkpoint.get('best_auroc', 0.0)
            if 'metrics' in checkpoint:
                metrics = checkpoint['metrics']
            
            print(f"Successfully loaded checkpoint - starting from epoch {start_epoch}")
            print(f"Previous best AUROC: {best_auroc:.4f}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch")
    else:
        print("Starting training from scratch")
    
    # Create scheduler after loading optimizer state
    scheduler = WarmupCosineScheduler(
        optimizer, 
        warmup_epochs=args.warmup_epochs, 
        total_epochs=args.epochs,
        min_lr=args.min_lr,
        warmup_start_lr=args.warmup_start_lr
    )
    
    # If resuming, update scheduler state
    if start_epoch > 0:
        for _ in range(start_epoch):
            scheduler.step()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup scaler for mixed precision training
    scaler = GradScaler()
    
    # Tracking variables for early stopping
    epochs_no_improve = 0
    
    print("\n" + "="*50)
    print(f"Starting training from epoch {start_epoch+1} for {args.epochs} epochs")
    print("="*50)
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        # Training phase
        for step, (images, labels, metadata) in enumerate(progress_bar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)
            
            # Apply mixup with probability
            use_mixup = (random.random() < args.mixup_prob) and (args.mixup_alpha > 0)
            
            # zero the parameter gradients
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                if use_mixup:
                    # Apply mixup
                    mixed_images, mixed_metadata, labels_a, labels_b, lam = mixup_data(
                        images, labels, metadata, alpha=args.mixup_alpha
                    )
                    
                    # Forward pass with mixed inputs
                    cls_out, combined_out = model(mixed_images, mixed_metadata)
                    
                    # Compute loss with mixup criterion
                    loss = lam * advanced_loss(cls_out, combined_out, labels_a, device, args.label_smoothing) + \
                           (1 - lam) * advanced_loss(cls_out, combined_out, labels_b, device, args.label_smoothing)
                else:
                    # Regular forward pass
                    cls_out, combined_out = model(images, metadata)
                    
                    # Regular loss computation
                    loss = advanced_loss(cls_out, combined_out, labels, device, args.label_smoothing)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            scaler.step(optimizer)
            scaler.update()
            
            # Update metrics
            train_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Memory cleanup
            del images, labels, metadata, cls_out, combined_out, loss
            torch.cuda.empty_cache()
        
        # Calculate average train loss
        avg_train_loss = train_loss / len(train_loader)
        metrics['train_loss'].append(avg_train_loss)
        metrics['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_targets = []
        all_outputs = []
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            for images, labels, metadata in tqdm(valid_loader, desc="Validating"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                metadata = metadata.to(device, non_blocking=True)
                
                # Forward pass
                cls_out, combined_out = model(images, metadata)
                
                # Compute validation loss
                loss = advanced_loss(cls_out, combined_out, labels, device, args.label_smoothing)
                val_loss += loss.item()
                
                # Store outputs for metrics calculation
                all_targets.append(labels.cpu())
                all_outputs.append(torch.sigmoid(combined_out).cpu())  # Use sigmoid for probability
                
                # Memory cleanup
                del images, labels, metadata, cls_out, combined_out, loss
                torch.cuda.empty_cache()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(valid_loader)
        metrics['val_loss'].append(avg_val_loss)
        
        # Concatenate all outputs and targets
        all_targets = torch.cat(all_targets, dim=0)
        all_outputs = torch.cat(all_outputs, dim=0)
        
        # Calculate AUROC for each class
        class_aurocs = {}
        for i, cls_name in enumerate(TRAIN_CLASSES):
            if len(torch.unique(all_targets[:, i])) > 1:  # Ensure we have both positive and negative samples
                auroc = roc_auc_score(all_targets[:, i].numpy(), all_outputs[:, i].numpy())
                class_aurocs[cls_name] = auroc
                print(f"{cls_name}: AUROC = {auroc:.4f}")
            else:
                print(f"Skipping {cls_name} - not enough unique labels")
                class_aurocs[cls_name] = 0.0
        
        # Calculate mean AUROC
        mean_auroc = np.mean(list(class_aurocs.values()))
        metrics['mean_auroc'].append(mean_auroc)
        
        # Print epoch results
        print(f"\nEpoch {epoch+1}/{args.epochs} - Results:")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Mean AUROC: {mean_auroc:.4f}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Update learning rate scheduler
        scheduler.step()
        
        # Plot and save metrics
        plot_training_progress(metrics, save_path=os.path.join(args.output_dir, f"training_progress_epoch_{epoch+1}.png"))
        plot_class_metrics(class_aurocs, save_path=os.path.join(args.output_dir, f"class_metrics_epoch_{epoch+1}.png"))
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, "checkpoint_latest.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.__dict__,
            'metrics': metrics,
            'best_auroc': best_auroc
        }, checkpoint_path)
        
        # Check for best model
        if mean_auroc > best_auroc:
            best_auroc = mean_auroc
            best_model_path = os.path.join(args.output_dir, f"best_model_auroc_{best_auroc:.4f}.pth")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'class_aurocs': class_aurocs,
                'mean_auroc': mean_auroc
            }, best_model_path)
            
            print(f"✅ New best model saved! Mean AUROC: {best_auroc:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"📉 No improvement for {epochs_no_improve} epochs. Best AUROC: {best_auroc:.4f}")
        
        # Early stopping check
        if epochs_no_improve >= args.patience:
            print(f"🛑 Early stopping triggered after {epoch+1} epochs.")
            break
    
    print("\n" + "="*50)
    print(f"Training completed. Best Mean AUROC: {best_auroc:.4f}")
    print("="*50)
    
    # Final evaluation on best model
    if os.path.exists(os.path.join(args.output_dir, f"best_model_auroc_{best_auroc:.4f}.pth")):
        print("\nLoading best model for final evaluation...")
        best_state_dict = torch.load(
            os.path.join(args.output_dir, f"best_model_auroc_{best_auroc:.4f}.pth"), 
            weights_only=True
        )
        model.load_state_dict(best_state_dict['model_state_dict'])
        
        # Evaluate at different thresholds
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
            print(f"\nEvaluating at threshold {threshold}:")
            evaluate_model(model, valid_loader, device, threshold=threshold)
    
    return best_auroc

def evaluate_model(model, dataloader, device, threshold=0.5):
    """Evaluate model with precision, recall and F1 at given threshold"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        for images, labels, metadata in tqdm(dataloader, desc=f"Evaluating (threshold={threshold})"):
            images = images.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)
            
            # Forward pass
            _, combined_out = model(images, metadata)
            
            # Apply threshold
            preds = (torch.sigmoid(combined_out) > threshold).float()
            
            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())
    
    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    # Calculate metrics for each class
    print(f"\nResults at threshold {threshold}:")
    print("-" * 80)
    print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUROC':<10}")
    print("-" * 80)
    
    class_metrics = {}
    for i, cls_name in enumerate(TRAIN_CLASSES):
        if np.sum(all_targets[:, i]) > 0:  # Ensure we have positive samples
            precision = precision_score(all_targets[:, i], all_preds[:, i])
            recall = recall_score(all_targets[:, i], all_preds[:, i])
            f1 = f1_score(all_targets[:, i], all_preds[:, i])
            auroc = roc_auc_score(all_targets[:, i], all_preds[:, i])
            
            class_metrics[cls_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auroc': auroc
            }
            
            print(f"{cls_name:<20} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {auroc:<10.4f}")
    
    # Calculate mean metrics
    mean_precision = np.mean([metrics['precision'] for metrics in class_metrics.values()])
    mean_recall = np.mean([metrics['recall'] for metrics in class_metrics.values()])
    mean_f1 = np.mean([metrics['f1'] for metrics in class_metrics.values()])
    mean_auroc = np.mean([metrics['auroc'] for metrics in class_metrics.values()])
    
    print("-" * 80)
    print(f"{'Mean':<20} {mean_precision:<10.4f} {mean_recall:<10.4f} {mean_f1:<10.4f} {mean_auroc:<10.4f}")
    print("-" * 80)
    
    return mean_auroc, mean_f1, class_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train advanced chest X-ray classifier")
    
    # Data paths
    parser.add_argument("--data_path", type=str, default="data", help="Path to the data directory")
    parser.add_argument("--csv_path", type=str, default="data/Data_Entry_2017.csv", help="Path to the CSV file")
    parser.add_argument("--list_path", type=str, default="data/train_val_list.txt", help="Path to train-val list")
    parser.add_argument("--output_dir", type=str, default="advanced_outputs", help="Output directory")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--image_size", type=int, default=512, help="Image size")
    parser.add_argument("--base_lr", type=float, default=1e-5, help="Base learning rate for backbone")
    parser.add_argument("--head_lr", type=float, default=5e-5, help="Learning rate for heads")
    parser.add_argument("--min_lr", type=float, default=1e-7, help="Minimum learning rate")
    parser.add_argument("--warmup_start_lr", type=float, default=1e-6, help="Starting LR for warmup")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout rate")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor")
    parser.add_argument("--mixup_alpha", type=float, default=0.5, help="Mixup alpha")
    parser.add_argument("--mixup_prob", type=float, default=0.7, help="Probability to apply mixup")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Warmup epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    
    # Other options
    parser.add_argument("--include_no_finding", action="store_true", help="Include No Finding class")
    parser.add_argument("--checkpoint_path", type=str, help="Path to checkpoint file")
    parser.add_argument("--start_from_scratch", action="store_true", help="Start training from scratch")
    
    args = parser.parse_args()
    
    # Print all arguments
    print("\nTraining with the following parameters:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    
    # Start training
    train_advanced(args) 