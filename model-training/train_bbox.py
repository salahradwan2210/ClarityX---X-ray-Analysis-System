import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import torch.nn.functional as F
from bbox_model import BBoxModel, CFG, get_transforms, calculate_iou
from torch.cuda.amp import GradScaler, autocast

def find_image_path(image_name, base_folder):
    """Find image in the nested directory structure"""
    for i in range(1, 13):
        image_path = os.path.join(base_folder, f'images_{i:03d}', 'images', image_name)
        if os.path.exists(image_path):
            return image_path
    image_path_flat = os.path.join(base_folder, 'images', image_name)
    if os.path.exists(image_path_flat):
        return image_path_flat
    return None

def custom_collate_fn(batch):
    """Custom collate function to handle variable size bounding boxes"""
    images = []
    boxes = []
    labels = []
    max_boxes = max(sample['boxes'].shape[0] for sample in batch)
    
    for sample in batch:
        images.append(sample['image'])
        
        # Pad boxes if necessary
        if sample['boxes'].shape[0] < max_boxes:
            padded_boxes = torch.zeros((max_boxes, 4))
            padded_boxes[:sample['boxes'].shape[0]] = sample['boxes']
            boxes.append(padded_boxes)
        else:
            boxes.append(sample['boxes'])
            
        labels.append(sample['labels'])
    
    # Stack all tensors
    images = torch.stack(images)
    boxes = torch.stack(boxes)
    labels = torch.stack(labels)
    
    return {
        'image': images,
        'boxes': boxes,
        'labels': labels
    }

class ChestBBoxDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.classes = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
                       'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']
    
    def __len__(self):
        return len(self.data)
    
    def load_image(self, image_path):
        """Safely load image with multiple attempts"""
        try:
            image = Image.open(image_path).convert('RGB')
            return np.array(image)
        except Exception as e:
            print(f"PIL failed to load {image_path}: {e}")
            try:
                image = cv2.imread(image_path)
                if image is not None:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"OpenCV failed to load {image_path}: {e}")
            return None
    
    def __getitem__(self, idx):
        # Get image path
        img_name = self.data.iloc[idx]['Image Index']
        img_path = find_image_path(img_name, self.img_dir)
        
        # Initialize default image and labels
        if not img_path:
            print(f"Image not found: {img_name}")
            image = np.zeros((CFG.IMG_SIZE, CFG.IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = self.load_image(img_path)
            if image is None:
                print(f"Failed to load image: {img_path}")
                image = np.zeros((CFG.IMG_SIZE, CFG.IMG_SIZE, 3), dtype=np.uint8)
        
        # Get bounding boxes and labels
        boxes = []
        labels = torch.zeros(len(self.classes))
        
        # Parse bbox annotations
        bbox_str = self.data.iloc[idx]['Bounding Boxes']
        if isinstance(bbox_str, str) and bbox_str != '':
            try:
                bbox_list = eval(bbox_str)
                for bbox_dict in bbox_list:
                    if bbox_dict['label'] in self.classes:
                        class_idx = self.classes.index(bbox_dict['label'])
                        labels[class_idx] = 1
                        # Normalize bbox coordinates
                        x = bbox_dict['x'] / image.shape[1]
                        y = bbox_dict['y'] / image.shape[0]
                        w = bbox_dict['width'] / image.shape[1]
                        h = bbox_dict['height'] / image.shape[0]
                        # Ensure coordinates are valid
                        x = np.clip(x, 0, 1)
                        y = np.clip(y, 0, 1)
                        w = np.clip(w, 0, 1-x)
                        h = np.clip(h, 0, 1-y)
                        boxes.append([x, y, w, h])
            except Exception as e:
                print(f"Error parsing bbox for {img_name}: {e}")
        
        # Ensure we always have at least one box (even if it's zero-filled)
        if len(boxes) == 0:
            boxes = [[0, 0, 0, 0]]
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        
        # Apply transforms
        if self.transform:
            try:
                transformed = self.transform(image=image)
                image = transformed['image']
            except Exception as e:
                print(f"Transform failed for {img_name}: {e}")
                image = torch.zeros((3, CFG.IMG_SIZE, CFG.IMG_SIZE))
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels
        }

def train_epoch(model, train_loader, optimizer, device, epoch, scaler):
    model.train()
    total_loss = 0.0
    bbox_losses = 0.0
    conf_losses = 0.0
    
    # Initialize loss functions
    bce_loss = nn.BCEWithLogitsLoss()
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(progress_bar):
        # Move data to GPU
        images = batch['image'].to(device, non_blocking=True)
        target_boxes = batch['boxes'].to(device, non_blocking=True)
        target_labels = batch['labels'].to(device, non_blocking=True)
        
        # Clear gradients and cache
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        
        try:
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda'):
                bbox_pred, confidence, _ = model(images)
                
                # Calculate losses
                # Confidence loss using BCEWithLogitsLoss
                conf_loss = bce_loss(confidence, target_labels)
                
                # Calculate bbox loss
                # Reshape predictions to match target shape: [batch_size, max_boxes, 4]
                B = target_boxes.size(0)  # batch size
                N = target_boxes.size(1)  # number of boxes per image
                bbox_pred = bbox_pred.view(B, -1, 4)  # [B, num_classes, 4]
                
                # Get valid box mask (boxes that are not all zeros)
                valid_mask = (target_boxes.sum(dim=-1) > 0)  # [B, N]
                
                # Initialize bbox loss
                if valid_mask.any():
                    # Calculate loss only for the first valid box per image
                    # This assumes one box per class in the ground truth
                    bbox_loss = 0
                    num_valid = 0
                    
                    for b in range(B):
                        # Get indices of valid boxes for this image
                        valid_indices = valid_mask[b].nonzero().squeeze(-1)
                        if valid_indices.numel() > 0:
                            for idx in valid_indices:
                                # Get the corresponding class prediction
                                pred_box = bbox_pred[b, idx]
                                target_box = target_boxes[b, idx]
                                
                                # Add to loss
                                bbox_loss += F.smooth_l1_loss(
                                    pred_box,
                                    target_box,
                                    reduction='mean'
                                )
                                num_valid += 1
                    
                    # Average the loss
                    if num_valid > 0:
                        bbox_loss = bbox_loss / num_valid
                    else:
                        bbox_loss = torch.tensor(0.0, device=device)
                else:
                    bbox_loss = torch.tensor(0.0, device=device)
                
                # Total loss - weight bbox_loss less if there are few valid boxes
                loss = conf_loss + 0.5 * bbox_loss
            
            # Check if loss is valid
            if not torch.isfinite(loss):
                print(f'Warning: Loss is not finite in batch {batch_idx}. Skipping batch.')
                continue
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Update metrics (detach tensors to prevent memory leaks)
            total_loss += loss.detach().item()
            bbox_losses += bbox_loss.detach().item()
            conf_losses += conf_loss.detach().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{total_loss/(batch_idx+1):.4f}',
                'bbox_loss': f'{bbox_losses/(batch_idx+1):.4f}',
                'conf_loss': f'{conf_losses/(batch_idx+1):.4f}'
            })
            
        except RuntimeError as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            torch.cuda.empty_cache()
            continue
        
        # Clear memory
        del images, target_boxes, target_labels, bbox_pred, confidence
        del loss, bbox_loss, conf_loss, valid_mask
        torch.cuda.empty_cache()
    
    # Compute average losses
    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_bbox_loss = bbox_losses / num_batches if num_batches > 0 else 0.0
    avg_conf_loss = conf_losses / num_batches if num_batches > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'bbox_loss': avg_bbox_loss,
        'conf_loss': avg_conf_loss
    }

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    bbox_losses = 0.0
    conf_losses = 0.0
    
    # Initialize loss functions
    bce_loss = nn.BCEWithLogitsLoss()
    
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for batch_idx, batch in enumerate(val_loader):
            images = batch['image'].to(device)
            target_boxes = batch['boxes'].to(device)
            target_labels = batch['labels'].to(device)
            
            try:
                # Forward pass
                bbox_pred, confidence, _ = model(images)
                
                # Confidence loss using BCEWithLogitsLoss
                conf_loss = bce_loss(confidence, target_labels)
                
                # Calculate bbox loss
                # Reshape predictions to match target shape
                B = target_boxes.size(0)
                N = target_boxes.size(1)
                bbox_pred = bbox_pred.view(B, -1, 4)
                
                # Get valid box mask
                valid_mask = (target_boxes.sum(dim=-1) > 0)
                
                # Initialize bbox loss
                if valid_mask.any():
                    # Calculate loss only for the first valid box per image
                    bbox_loss = 0
                    num_valid = 0
                    
                    for b in range(B):
                        valid_indices = valid_mask[b].nonzero().squeeze(-1)
                        if valid_indices.numel() > 0:
                            for idx in valid_indices:
                                pred_box = bbox_pred[b, idx]
                                target_box = target_boxes[b, idx]
                                bbox_loss += F.smooth_l1_loss(
                                    pred_box,
                                    target_box,
                                    reduction='mean'
                                )
                                num_valid += 1
                    
                    if num_valid > 0:
                        bbox_loss = bbox_loss / num_valid
                    else:
                        bbox_loss = torch.tensor(0.0, device=device)
                else:
                    bbox_loss = torch.tensor(0.0, device=device)
                
                # Total loss
                loss = conf_loss + 0.5 * bbox_loss
                
                # Skip batch if loss is not finite
                if not torch.isfinite(loss):
                    print(f'Warning: Validation loss is not finite in batch {batch_idx}. Skipping batch.')
                    continue
                
                # Update metrics
                total_loss += loss.item()
                bbox_losses += bbox_loss.item()
                conf_losses += conf_loss.item()
                
            except RuntimeError as e:
                print(f"Error in validation batch {batch_idx}: {str(e)}")
                continue
            
            # Clear GPU cache periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
    
    # Compute average losses
    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_bbox_loss = bbox_losses / num_batches if num_batches > 0 else 0.0
    avg_conf_loss = conf_losses / num_batches if num_batches > 0 else 0.0
    
    return {
        'val_loss': avg_loss,
        'val_bbox_loss': avg_bbox_loss,
        'val_conf_loss': avg_conf_loss
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=CFG.BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--train_csv', type=str, default='./train_bbox.csv')
    parser.add_argument('--val_csv', type=str, default='./val_bbox.csv')
    parser.add_argument('--output_dir', type=str, default='./models')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set memory optimization flags
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Create model without pretrained weights
    print("Initializing ResNet50-based model from scratch...")
    model = BBoxModel(pretrained=False)
    model = model.to(CFG.DEVICE)
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Create datasets and dataloaders
    train_transform = get_transforms(CFG.IMG_SIZE)
    val_transform = get_transforms(CFG.IMG_SIZE)
    
    print("Loading datasets...")
    train_dataset = ChestBBoxDataset(
        csv_file=args.train_csv,
        img_dir=args.data_dir,
        transform=train_transform
    )
    
    val_dataset = ChestBBoxDataset(
        csv_file=args.val_csv,
        img_dir=args.data_dir,
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=custom_collate_fn
    )
    
    # Create optimizer with weight decay and higher learning rate
    optimizer = optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.01,
        eps=1e-8
    )
    
    # Create scheduler with longer patience
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=8,
        verbose=True,
        min_lr=1e-6
    )
    
    print("Starting training...")
    # Training loop
    best_loss = float('inf')
    for epoch in range(args.epochs):
        try:
            # Train
            train_metrics = train_epoch(model, train_loader, optimizer, CFG.DEVICE, epoch, scaler)
            
            # Validate
            val_metrics = validate(model, val_loader, CFG.DEVICE)
            
            # Update learning rate
            scheduler.step(val_metrics['val_loss'])
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(
                    args.output_dir, 
                    f'checkpoint_epoch_{epoch+1}.pth'
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['val_loss'],
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
            
            # Save best model
            if val_metrics['val_loss'] < best_loss:
                best_loss = val_metrics['val_loss']
                best_model_path = os.path.join(
                    args.output_dir,
                    f'best_model_epoch_{epoch+1}_loss_{best_loss:.4f}.pth'
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_metrics['loss'],
                    'val_loss': best_loss,
                    'train_bbox_loss': train_metrics['bbox_loss'],
                    'train_conf_loss': train_metrics['conf_loss'],
                    'val_bbox_loss': val_metrics['val_bbox_loss'],
                    'val_conf_loss': val_metrics['val_conf_loss'],
                }, best_model_path)
                print(f"\nSaved new best model with loss {best_loss:.4f} to {best_model_path}")
            
            # Print metrics
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"Best Val Loss: {best_loss:.4f}")
            
            # Clear GPU cache after each epoch
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                torch.cuda.empty_cache()
                continue
            else:
                raise e

if __name__ == '__main__':
    main() 