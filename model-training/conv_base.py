import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

# Define classes
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
               'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
               'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
               'Pleural_Thickening', 'Hernia', 'No Finding']
TRAIN_CLASSES = [c for c in CLASS_NAMES if c != 'No Finding']

# Pre-resize images if not already done
def resize_images(input_dir, output_dir, size=(224, 224)):
    print("Resizing images...")
    os.makedirs(output_dir, exist_ok=True)
    for i in range(1, 13):
        folder_name = f'images_{i:03d}'
        input_folder = os.path.join(input_dir, folder_name, 'images')
        output_folder = os.path.join(output_dir, folder_name, 'images')
        os.makedirs(output_folder, exist_ok=True)
        for img_name in os.listdir(input_folder):
            img_path = os.path.join(input_folder, img_name)
            output_path = os.path.join(output_folder, img_name)
            if not os.path.exists(output_path):
                img = Image.open(img_path).convert('RGB')
                img = img.resize(size, Image.LANCZOS)
                img.save(output_path)
    print("Image resizing completed!")

def find_image_path(image_name, base_folder):
    for i in range(1, 13):
        folder_name = f'images_{i:03d}'
        image_path = os.path.join(base_folder, folder_name, 'images', image_name)
        if os.path.exists(image_path):
            return image_path
    return None

def preprocess_metadata(df):
    df['Patient Age'] = df['Patient Age'].clip(upper=100)
    df['Patient Age'] = df['Patient Age'].fillna(df['Patient Age'].mean())
    gender_encoder = LabelEncoder()
    df['Patient Gender'] = gender_encoder.fit_transform(df['Patient Gender'])
    view_encoder = LabelEncoder()
    df['View Position'] = view_encoder.fit_transform(df['View Position'])
    df['Patient Age'] = df['Patient Age'] / 100.0
    return df, gender_encoder, view_encoder

def load_bounding_boxes(bbox_path):
    bbox_df = pd.read_csv(bbox_path)
    bbox_dict = {}
    for _, row in bbox_df.iterrows():
        img_name = row['Image Index']
        disease = row['Finding Label']
        if disease in TRAIN_CLASSES:
            if img_name not in bbox_dict:
                bbox_dict[img_name] = {}
            bbox_dict[img_name][disease] = [
                row['Bbox [x'], row['y'], row['w'], row['h]']
            ]
    return bbox_dict

class ChestXrayDataset(Dataset):
    def __init__(self, image_dir, df, bbox_dict, transform=None, train=True):
        self.image_dir = image_dir
        self.df = df
        self.bbox_dict = bbox_dict
        self.transform = transform
        self.train = train
        
        self.labels = []
        self.bboxes = []
        for idx, row in self.df.iterrows():
            finding = row['Finding Labels']
            img_name = row['Image Index']
            label = np.zeros(len(TRAIN_CLASSES), dtype=np.float32)
            bbox = np.zeros((len(TRAIN_CLASSES), 4), dtype=np.float32)
            has_bbox = False
            
            for cls in finding.split('|'):
                if cls in TRAIN_CLASSES:
                    cls_idx = TRAIN_CLASSES.index(cls)
                    label[cls_idx] = 1
                    if img_name in self.bbox_dict and cls in self.bbox_dict[img_name]:
                        bbox[cls_idx] = self.bbox_dict[img_name][cls]
                        has_bbox = True
            
            self.labels.append(label)
            self.bboxes.append(bbox if has_bbox else np.zeros_like(bbox))
        
        self.labels = np.array(self.labels)
        self.bboxes = np.array(self.bboxes)
        self.metadata = self.df[['Patient Age', 'Patient Gender', 'View Position']].values.astype(np.float32)
        
        print(f"Dataset size: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['Image Index']
        img_path = find_image_path(img_name, self.image_dir)
        if img_path is None:
            raise FileNotFoundError(f"Image not found: {img_name}")
            
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.from_numpy(self.labels[idx])
        bbox = torch.from_numpy(self.bboxes[idx])
        metadata = torch.from_numpy(self.metadata[idx])
        
        return image, label, metadata, bbox

class ConvNeXtModel(nn.Module):
    def __init__(self, num_classes, model_name='convnext_large'):
        super(ConvNeXtModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model.set_grad_checkpointing(enable=True)  # Explicitly set use_reentrant=False in future PyTorch versions
        in_features = self.model.head.in_features if hasattr(self.model.head, 'in_features') else (1536 if model_name == 'convnext_large' else 1024)
        
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(in_features),
            nn.Dropout(0.6),  # Increased dropout to reduce overfitting
            nn.Linear(in_features, num_classes - 1)
        )
        
        self.localization_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(in_features),
            nn.Dropout(0.6),  # Increased dropout
            nn.Linear(in_features, (num_classes - 1) * 4)
        )
        
        self.metadata_branch = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # Increased dropout
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        self.combine_fc = nn.Linear(in_features + 128, num_classes - 1)
        
        nn.init.constant_(self.classification_head[-1].bias, 0.1)
        nn.init.constant_(self.localization_head[-1].bias, 0.0)

    def forward(self, x, metadata):
        features = self.model(x)
        cls_out = self.classification_head(features.unsqueeze(-1).unsqueeze(-1))
        bbox_out = self.localization_head(features.unsqueeze(-1).unsqueeze(-1))
        meta_features = self.metadata_branch(metadata)
        combined_features = torch.cat([features, meta_features], dim=1)
        combined_out = self.combine_fc(combined_features)
        return cls_out, bbox_out, combined_out

def custom_loss(cls_out, bbox_out, combined_out, labels, bboxes, class_weights, device, label_smoothing=0.1):
    smoothed_labels = labels * (1 - label_smoothing) + label_smoothing / 2
    cls_criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
    bbox_criterion = nn.SmoothL1Loss()
    cls_loss = cls_criterion(cls_out, smoothed_labels)
    combined_loss = cls_criterion(combined_out, smoothed_labels)
    bbox_mask = (bboxes.sum(dim=-1) > 0).float()
    bbox_loss = 0
    if bbox_mask.sum() > 0:
        bbox_loss = bbox_criterion(bbox_out * bbox_mask.unsqueeze(-1), bboxes * bbox_mask.unsqueeze(-1))
    total_loss = cls_loss + combined_loss + 0.5 * bbox_loss
    return total_loss

def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler=None, num_epochs=20, device='cuda', accum_steps=8, class_weights=None, model_name='model', fine_tune=False):
    print(f"Training {model_name} on {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'mean_auroc': [],
        'class_aurocs': defaultdict(list)
    }
    
    scaler = torch.amp.GradScaler()
    model = model.to(device, memory_format=torch.channels_last)
    best_auroc = 0.0
    patience = 4  # Reduced patience for early stopping
    epochs_no_improve = 0
    
    checkpoint_path = f'checkpoint_{model_name}.pth'
    best_model_path = f'best_model_{model_name}_auroc.pth'
    start_epoch = 0
    
    # Load checkpoint if it exists
    if os.path.exists(checkpoint_path) and not fine_tune:
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_auroc = checkpoint['best_auroc']
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
        print(f"Resuming training for {model_name} from epoch {start_epoch} with best AUROC: {best_auroc}")
    elif fine_tune and os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, weights_only=False, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_auroc = checkpoint['best_auroc']
        print(f"Fine-tuning {model_name} from best model with AUROC: {best_auroc}")
    else:
        print(f"No checkpoint found for {model_name}. Starting from scratch.")
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'{model_name} Epoch {epoch+1}/{num_epochs}')
        optimizer.zero_grad(set_to_none=True)
        
        for i, (inputs, labels, metadata, bboxes) in enumerate(progress_bar):
            inputs = inputs.to(device, non_blocking=True, memory_format=torch.channels_last)
            metadata = metadata.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            bboxes = bboxes.to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', dtype=torch.float16):
                cls_out, bbox_out, combined_out = model(inputs, metadata)
                loss = criterion(cls_out, bbox_out, combined_out, labels, bboxes, class_weights, device)
                loss = loss / accum_steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            running_loss += loss.item() * accum_steps
            progress_bar.set_postfix({'loss': f'{loss.item() * accum_steps:.4f}'})
        
        epoch_loss = running_loss / len(train_loader)
        metrics['train_loss'].append(epoch_loss)
        
        model.eval()
        val_loss = 0.0
        all_targets = []
        all_outputs = []
        
        with torch.no_grad(), torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', dtype=torch.float16):
            for inputs, labels, metadata, bboxes in valid_loader:
                inputs = inputs.to(device, non_blocking=True, memory_format=torch.channels_last)
                labels = labels.to(device, non_blocking=True)
                metadata = metadata.to(device, non_blocking=True)
                bboxes = bboxes.to(device, non_blocking=True)
                
                cls_out, bbox_out, combined_out = model(inputs, metadata)
                loss = criterion(cls_out, bbox_out, combined_out, labels, bboxes, class_weights, device)
                val_loss += loss.item()
                
                all_targets.append(labels.cpu())
                all_outputs.append(combined_out.sigmoid().cpu())
        
        val_loss = val_loss / len(valid_loader)
        metrics['val_loss'].append(val_loss)
        
        all_targets = torch.cat(all_targets, dim=0)
        all_outputs = torch.cat(all_outputs, dim=0)
        
        aurocs = {}
        for i, cls_name in enumerate(TRAIN_CLASSES):
            if len(torch.unique(all_targets[:, i])) > 1:
                auroc = roc_auc_score(all_targets[:, i], all_outputs[:, i])
                aurocs[cls_name] = auroc
                metrics['class_aurocs'][cls_name].append(auroc)
                print(f'{cls_name} AUROC: {auroc:.4f}')
        
        mean_auroc = np.mean(list(aurocs.values()))
        metrics['mean_auroc'].append(mean_auroc)
        
        print(f'\n{model_name} Epoch {epoch+1}:')
        print(f'Training Loss: {epoch_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Mean AUROC: {mean_auroc:.4f}')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_auroc': best_auroc,
            'metrics': metrics
        }, checkpoint_path)
        print(f"Saved checkpoint for {model_name} at epoch {epoch+1}")
        
        if mean_auroc > best_auroc:
            best_auroc = mean_auroc
            best_model_path = f'best_model_{model_name}_auroc_{best_auroc:.3f}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auroc': best_auroc,
                'class_aurocs': aurocs
            }, best_model_path)
            print(f'Saved new best model for {model_name} with AUROC: {best_auroc:.4f}')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if scheduler is not None:
            scheduler.step(mean_auroc)
        
        if epochs_no_improve >= patience:
            print(f"No improvement in AUROC for {patience} epochs. Stopping training for {model_name}.")
            break
    
    return best_auroc

def load_image_list(list_file_path):
    with open(list_file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def analyze_dataset(df):
    print("\nClass distribution in dataset:")
    for cls in TRAIN_CLASSES:
        count = len(df[df['Finding Labels'].str.contains(cls, regex=False)])
        print(f"{cls}: {count} samples")
    print(f"\nTotal samples: {len(df)}\n")
    return df

def balance_dataset(df):
    print("\nBalancing dataset...")
    conditions = TRAIN_CLASSES
    samples_per_condition = []
    for condition in conditions:
        count = len(df[df['Finding Labels'].str.contains(condition, regex=False)])
        if count > 0:
            samples_per_condition.append(count)
    
    target_size = int(np.mean(samples_per_condition))
    balanced_dfs = []
    
    for condition in conditions:
        condition_df = df[df['Finding Labels'].str.contains(condition, regex=False)]
        if len(condition_df) > target_size:
            condition_df = condition_df.sample(n=target_size, random_state=42)
        balanced_dfs.append(condition_df)
    
    balanced_df = pd.concat(balanced_dfs).drop_duplicates().sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("\nFinal class distribution after balancing:")
    for cls in TRAIN_CLASSES:
        count = len(balanced_df[balanced_df['Finding Labels'].str.contains(cls, regex=False)])
        print(f"{cls}: {count} samples")
    print(f"\nTotal samples after balancing: {len(balanced_df)}")
    return balanced_df

def calculate_class_weights(df):
    weights = []
    for cls in TRAIN_CLASSES:
        count = len(df[df['Finding Labels'].str.contains(cls, regex=False)])
        weights.append(1.0 / (count + 1e-6))
    weights = np.array(weights)
    weights = weights / np.min(weights)
    return torch.FloatTensor(weights)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        torch.cuda.empty_cache()
    
    base_path = 'data'
    resized_base_path = 'data_resized'
    image_dir = resized_base_path
    data_entry_path = os.path.join(base_path, 'Data_Entry_2017.csv')
    bbox_path = os.path.join(base_path, 'BBox_List_2017.csv')
    train_val_list_path = os.path.join(base_path, 'train_val_list.txt')
    
    if not os.path.exists(resized_base_path):
        resize_images(base_path, resized_base_path)
    
    print(f"Loading data from: {base_path}")
    valid_images = set(load_image_list(train_val_list_path))
    print(f"Found {len(valid_images)} images in list file")
    
    df = pd.read_csv(data_entry_path)
    df = df[df['Image Index'].isin(valid_images)]
    df = df[df['Finding Labels'] != 'No Finding']
    print(f"Total images after removing No Finding: {len(df)}")
    
    df, gender_encoder, view_encoder = preprocess_metadata(df)
    bbox_dict = load_bounding_boxes(bbox_path)
    print(f"Loaded {len(bbox_dict)} images with bounding boxes")
    
    df = balance_dataset(df)
    train_df = df.sample(frac=0.8, random_state=42)
    valid_df = df.drop(train_df.index)
    
    print("\nAnalyzing training set distribution:")
    analyze_dataset(train_df)
    print("\nAnalyzing validation set distribution:")
    analyze_dataset(valid_df)
    print(f"\nTraining samples: {len(train_df)}")
    print(f"Validation samples: {len(valid_df)}")
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ChestXrayDataset(image_dir, train_df, bbox_dict, transform=train_transform)
    valid_dataset = ChestXrayDataset(image_dir, valid_df, bbox_dict, transform=valid_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # Reduced to 1 to minimize memory usage
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    
    class_weights = calculate_class_weights(train_df)
    class_weights = class_weights.to(device)
    
    print("\nClass weights:")
    for cls, weight in zip(TRAIN_CLASSES, class_weights):
        count = len(train_df[train_df['Finding Labels'].str.contains(cls, regex=False)])
        print(f"{cls}: {weight:.2f} (samples: {count})")
    
    # Fine-tune convnext_large from the best checkpoint
    print("\nFine-tuning ConvNeXt Large from best checkpoint...")
    model_large = ConvNeXtModel(len(CLASS_NAMES), model_name='convnext_large')
    optimizer_large = torch.optim.AdamW([
        {'params': [p for n, p in model_large.named_parameters() if 'head' not in n], 'lr': 1e-5},  # Reduced LR for fine-tuning
        {'params': [p for n, p in model_large.named_parameters() if 'head' in n], 'lr': 2e-5}
    ], weight_decay=0.05)  # Increased weight decay
    
    scheduler_large = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_large,
        mode='max',
        factor=0.5,
        patience=2  # Reduced patience for fine-tuning
    )
    
    best_auroc_large = train_model(
        model_large, train_loader, valid_loader, custom_loss, optimizer_large, 
        scheduler=scheduler_large, num_epochs=5, device=device, accum_steps=8, 
        class_weights=class_weights, model_name='convnext_large', fine_tune=True
    )
    
    # Train ConvNeXt Base from scratch
    print("\nStarting training for ConvNeXt Base from scratch...")
    model_base = ConvNeXtModel(len(CLASS_NAMES), model_name='convnext_base')
    optimizer_base = torch.optim.AdamW([
        {'params': [p for n, p in model_base.named_parameters() if 'head' not in n], 'lr': 5e-5},
        {'params': [p for n, p in model_base.named_parameters() if 'head' in n], 'lr': 1e-4}
    ], weight_decay=0.05)
    
    scheduler_base = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_base,
        mode='max',
        factor=0.5,
        patience=3
    )
    
    best_auroc_base = train_model(
        model_base, train_loader, valid_loader, custom_loss, optimizer_base, 
        scheduler=scheduler_base, num_epochs=20, device=device, accum_steps=8, 
        class_weights=class_weights, model_name='convnext_base', fine_tune=False
    )

    print(f"\nBest AUROC for ConvNeXt Large: {best_auroc_large:.4f}")
    print(f"Best AUROC for ConvNeXt Base: {best_auroc_base:.4f}")