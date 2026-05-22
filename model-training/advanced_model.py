import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import gc
import random
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")

# Enable CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Set memory allocator settings
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'

# Define the 15 classes (excluding No Finding for training)
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
               'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
               'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
               'Pleural_Thickening', 'Hernia', 'No Finding']
TRAIN_CLASSES = [c for c in CLASS_NAMES if c != 'No Finding']

# Advanced dataset with aggressive augmentation
class AdvancedXrayDataset(Dataset):
    def __init__(self, image_dir, df, transform=None, train=True):
        self.image_dir = image_dir
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.train = train
        
        # Convert labels to one-hot encoding
        self.labels = []
        for idx, row in self.df.iterrows():
            finding = row['Finding Labels']
            label = np.zeros(len(TRAIN_CLASSES), dtype=np.float32)
            for cls in finding.split('|'):
                if cls in TRAIN_CLASSES:
                    cls_idx = TRAIN_CLASSES.index(cls)
                    label[cls_idx] = 1
            self.labels.append(label)
        
        self.labels = np.array(self.labels)
        self.metadata = self.df[['Patient Age', 'Patient Gender', 'View Position']].values.astype(np.float32)
        
        print(f"Dataset size: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['Image Index']
        img_path = self.find_image_path(img_name)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_name}, returning placeholder.")
            image = torch.zeros((3, 512, 512))
            label = torch.from_numpy(self.labels[idx])
            metadata = torch.from_numpy(self.metadata[idx])
            return image, label, metadata
        
        try:
            # Read image
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = clahe.apply(image)  # Apply CLAHE for better contrast
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Apply transformation
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            image = torch.zeros((3, 512, 512))
        
        label = torch.from_numpy(self.labels[idx])
        metadata = torch.from_numpy(self.metadata[idx])
        
        return image, label, metadata
    
    def find_image_path(self, image_name):
        for i in range(1, 13):
            folder_name = f'images_{i:03d}'
            image_path = os.path.join(self.image_dir, folder_name, 'images', image_name)
            if os.path.exists(image_path):
                return image_path
        # If not found, check direct path
        direct_path = os.path.join(self.image_dir, image_name)
        if os.path.exists(direct_path):
            return direct_path
        return ""

# MixUp augmentation
def mixup_data(x, y, metadata, alpha=0.4):
    """Applies Mixup augmentation to the batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_metadata = lam * metadata + (1 - lam) * metadata[index, :]
    
    y_a, y_b = y, y[index]
    
    return mixed_x, mixed_metadata, y_a, y_b, lam

def mixup_criterion(criterion, outputs, y_a, y_b, lam):
    """Custom loss function for Mixup."""
    return lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)

# Focal Loss for imbalanced data
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'sum':
            return torch.sum(F_loss)
        elif self.reduction == 'mean':
            return torch.mean(F_loss)
        else:
            return F_loss

# ASL Loss - Asymmetric Loss for imbalanced multilabel classification
class ASLLoss(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05, eps=1e-8, reduction='mean'):
        super(ASLLoss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Positive and negative parts
        pos_part = targets * torch.clamp((1.0 - inputs), min=0, max=1.0) ** self.gamma_pos
        neg_part = (1 - targets) * torch.clamp(inputs - self.clip, min=0, max=1.0) ** self.gamma_neg
        
        loss = -(pos_part * torch.log(inputs + self.eps) + neg_part * torch.log(1.0 - inputs + self.eps))
        
        if self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss

# Advanced model architecture using ConvNeXt-Large
class AdvancedXrayModel(nn.Module):
    def __init__(self, num_classes, metadata_features=3, dropout_rate=0.5):
        super(AdvancedXrayModel, self).__init__()
        
        # Use ConvNeXt-Large as the backbone
        self.model = timm.create_model('convnext_large', pretrained=True, num_classes=0)
        
        # Enable gradient checkpointing for memory efficiency
        self.model.set_grad_checkpointing(enable=True)
        
        # Get feature dimensions (ConvNeXt-Large has 1536 features)
        in_features = 1536
        
        # Classification head with larger capacity
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(in_features),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 768),
            nn.GELU(),
            nn.LayerNorm(768),
            nn.Dropout(dropout_rate/2),
            nn.Linear(768, num_classes-1)
        )
        
        # Metadata branch with larger capacity
        self.metadata_branch = nn.Sequential(
            nn.Linear(metadata_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate/2),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )
        
        # Combine features
        self.combine_fc = nn.Sequential(
            nn.Linear(in_features + 256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate/2),
            nn.Linear(512, num_classes-1)
        )
        
        # Initialize bias for better convergence
        nn.init.constant_(self.classification_head[-1].bias, -2.0)
        nn.init.constant_(self.combine_fc[-1].bias, -2.0)
    
    def forward(self, x, metadata):
        # Use channels_last memory format for better performance
        x = x.contiguous(memory_format=torch.channels_last)
        
        # Extract features
        features = self.model(x)
        
        # Process through classification head
        cls_out = self.classification_head(features.unsqueeze(-1).unsqueeze(-1))
        
        # Process metadata 
        meta_features = self.metadata_branch(metadata)
        
        # Combine visual and metadata features
        combined_features = torch.cat([features, meta_features], dim=1)
        combined_out = self.combine_fc(combined_features)
        
        return cls_out, combined_out

# Create advanced loss function
def advanced_loss(cls_out, combined_out, labels, device, label_smoothing=0.1, gamma=2.0):
    # Create both losses
    asl_loss = ASLLoss(gamma_neg=4, gamma_pos=0)
    focal_loss = FocalLoss(gamma=gamma)
    
    # Apply label smoothing
    smoothed_labels = labels * (1 - label_smoothing) + (label_smoothing / labels.shape[1])
    
    # Compute losses
    cls_loss_asl = asl_loss(torch.sigmoid(cls_out), smoothed_labels)
    cls_loss_focal = focal_loss(cls_out, smoothed_labels)
    
    combined_loss_asl = asl_loss(torch.sigmoid(combined_out), smoothed_labels)
    combined_loss_focal = focal_loss(combined_out, smoothed_labels)
    
    # Weighted combination 
    cls_loss = 0.7 * cls_loss_asl + 0.3 * cls_loss_focal
    combined_loss = 0.7 * combined_loss_asl + 0.3 * combined_loss_focal
    
    # Final loss
    total_loss = 0.4 * cls_loss + 0.6 * combined_loss
    
    return total_loss 