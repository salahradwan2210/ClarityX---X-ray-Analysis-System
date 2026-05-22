import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import timm
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from PIL import Image
import warnings
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to avoid main thread error on servers
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import math
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import cv2
    ALBUMENTATIONS_INSTALLED = True
except ImportError:
    print("Warning: Albumentations not installed. Falling back to basic Torchvision transforms.")
    ALBUMENTATIONS_INSTALLED = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
NEW_THRESHOLD = 0.45

# --- Configuration ---
class CFG:
    SEED = 42
    MODEL_NAME = 'convnext_large'
    IMG_SIZE = 512  # Increased image size for finer details
    BATCH_SIZE = 4  # Reduced batch size to accommodate larger images
    ACCUM_STEPS = 8  # Increased accumulation steps to maintain effective batch size
    NUM_WORKERS = 2
    EPOCHS = 50  # Additional epochs for fine-tuning
    LR = 5e-6  # Lower learning rate for fine-tuning
    HEAD_LR = 1e-5  # Lower head learning rate
    WEIGHT_DECAY = 0.01
    SCHEDULER = 'CosineAnnealingLR'
    T_MAX = EPOCHS
    MIN_LR = 1e-7
    WARMUP_EPOCHS = 1  # Reduced warmup for fine-tuning
    LOSS_FN = 'FocalLoss'
    FOCAL_ALPHA = 0.6
    FOCAL_GAMMA = 2.0  # Increased gamma for more focus on hard examples
    LABEL_SMOOTHING = 0.1
    PATIENCE = 20  # Increased patience for fine-tuning
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    GRAD_CLIP_NORM = 1.0
    DROPOUT_HEAD = 0.1  # Reduced dropout
    DROPOUT_META = 0.1
    BASE_PATH = 'data'
    IMAGE_DIR = BASE_PATH
    DATA_ENTRY_PATH = os.path.join(BASE_PATH, 'Data_Entry_2017.csv')
    BBOX_PATH = os.path.join(BASE_PATH, 'BBox_List_2017.csv')
    CHECKPOINT_LOAD_PATH = 'best_model_epoch_28_auroc_0.9688.pth'  # Load from your best model
    START_FROM_SCRATCH = False  # Continue training from checkpoint
    LOAD_STRICT = True  # Strict loading for fine-tuning
    MIN_AUROC = 0.8  # Higher minimum threshold
    AUROC_IMPROVE_THRESHOLD = 0.0001  # Smaller threshold for improvement detection
    CONFUSION_MATRIX_THRESHOLD = 0.5
    MIXUP_ALPHA = 0.2  # Reduced mixup for fine-tuning
    BBOX_LOSS_WEIGHT = 1.0  # Increased from 0.5 to focus more on localization

# --- Setup ---
torch.manual_seed(CFG.SEED)
np.random.seed(CFG.SEED)
if CFG.DEVICE == torch.device('cuda'):
    torch.cuda.manual_seed(CFG.SEED)
    torch.backends.cudnn.benchmark = True

# --- Class Definitions ---
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding']
TRAIN_CLASSES = [c for c in CLASS_NAMES if c != 'No Finding']
CLASSES_WITH_BBOX = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']

# --- Loss Function ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=CFG.FOCAL_ALPHA, gamma=CFG.FOCAL_GAMMA, label_smoothing=CFG.LABEL_SMOOTHING, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    def forward(self, inputs, targets):
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing / inputs.shape[1]
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduction == 'mean':
            return F_loss.mean()
        else:
            return F_loss

# --- Mixup Function ---
def mixup_data(x, y, alpha=CFG.MIXUP_ALPHA, device=CFG.DEVICE):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# --- Helper Functions ---
def find_image_path(image_name, base_folder):
    for i in range(1, 13):
        image_path = os.path.join(base_folder, f'images_{i:03d}', 'images', image_name)
        if os.path.exists(image_path):
            return image_path
    image_path_flat = os.path.join(base_folder, 'images', image_name)
    if os.path.exists(image_path_flat):
        return image_path_flat
    return None

def load_bounding_boxes(bbox_path):
    if not os.path.exists(bbox_path):
        return {}
    try:
        bbox_df = pd.read_csv(bbox_path)
        bbox_dict = {}
        for _, row in bbox_df.iterrows():
            img_name = row['Image Index']
            disease = row['Finding Label']
            if disease in CLASSES_WITH_BBOX:
                if img_name not in bbox_dict:
                    bbox_dict[img_name] = {}
                x_key = next((col for col in row.index if 'Bbox [x' in col), None)
                y_key = 'y'
                w_key = 'w'
                h_key = next((col for col in row.index if 'h]' in col), None)
                try:
                    if x_key and h_key:
                        bbox_dict[img_name][disease] = [float(row[x_key]), float(row[y_key]), float(row[w_key]), float(row[h_key])]
                except (KeyError, ValueError, TypeError):
                    pass
    except Exception as e:
        print(f"Error loading bbox file: {e}")
        return {}
    return bbox_dict

def preprocess_metadata(df):
    df = df.copy()
    df['Patient Age'] = pd.to_numeric(df['Patient Age'], errors='coerce').clip(upper=110)
    mean_age = df['Patient Age'].mean()
    df['Patient Age'] = df['Patient Age'].fillna(mean_age if not pd.isna(mean_age) else 60) / 100.0
    df['Patient Gender'] = df['Patient Gender'].astype(str).str.strip().fillna('Unknown')
    unique_genders = df['Patient Gender'].unique()
    gender_encoder = LabelEncoder().fit(unique_genders)
    df['Patient Gender'] = gender_encoder.transform(df['Patient Gender'])
    df['View Position'] = df['View Position'].astype(str).str.strip().fillna('Unknown')
    unique_views = df['View Position'].unique()
    view_encoder = LabelEncoder().fit(unique_views)
    df['View Position'] = view_encoder.transform(df['View Position'])
    return df, gender_encoder, view_encoder

# --- Dataset ---
class ChestXrayDataset(Dataset):
    def __init__(self, image_dir, df, bbox_dict, transform=None, train=True):
        self.image_dir = image_dir
        self.df = df.reset_index(drop=True)
        self.bbox_dict = bbox_dict
        self.transform = transform
        self.train = train
        self.labels = []
        self.bboxes = []
        metadata_cols = [col for col in ['Patient Age', 'Patient Gender', 'View Position'] if col in self.df.columns]
        self.metadata_list = []
        for idx, row in self.df.iterrows():
            finding = row['Finding Labels']
            img_name = row['Image Index']
            label = np.zeros(len(TRAIN_CLASSES), dtype=np.float32)
            bbox = np.zeros((len(TRAIN_CLASSES), 4), dtype=np.float32)
            for cls in finding.split('|'):
                if cls in TRAIN_CLASSES:
                    cls_idx = TRAIN_CLASSES.index(cls)
                    label[cls_idx] = 1
                    if cls in CLASSES_WITH_BBOX and img_name in self.bbox_dict and cls in self.bbox_dict[img_name]:
                        try:
                            coords = np.array(self.bbox_dict[img_name][cls]).astype(np.float32)
                            if coords.shape == (4,) and np.all(coords >= 0):
                                coords /= 1024.0
                                coords = np.clip(coords, 0.0, 1.0)
                                if coords[2] > 0 and coords[3] > 0:
                                    bbox[cls_idx] = coords
                        except Exception:
                            pass
            self.labels.append(label)
            self.bboxes.append(bbox)
            self.metadata_list.append(row[metadata_cols].values.astype(np.float32))
        self.labels = np.array(self.labels)
        self.bboxes = np.array(self.bboxes)
        self.metadata = np.array(self.metadata_list)
        self.metadata_dim = self.metadata.shape[1] if self.metadata.size > 0 else 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['Image Index']
        img_path = find_image_path(img_name, self.image_dir)
        if img_path is None:
            print(f"Warning: Cannot find image {img_name}, using dummy item.")
            return self._get_dummy_item()
        try:
            pil_image = Image.open(img_path).convert('RGB')
            try:
                image = np.array(pil_image, dtype=np.uint8)
            except Exception as e:
                print(f"Error converting PIL image to numpy array for {img_name}: {e}")
                return self._get_dummy_item()
            if self.transform:
                try:
                    if ALBUMENTATIONS_INSTALLED:
                        augmented = self.transform(image=image)
                        image = augmented['image']
                    else:
                        image = self.transform(image)
                except Exception as e:
                    print(f"Error applying transforms to {img_name}: {e}")
                    return self._get_dummy_item()
            label = torch.from_numpy(self.labels[idx])
            bbox = torch.from_numpy(self.bboxes[idx])
            metadata = torch.from_numpy(self.metadata[idx])
            if torch.isnan(image).any() or torch.isinf(image).any():
                print(f"Warning: Found NaN or Inf values in image tensor for {img_name}")
                return self._get_dummy_item()
            return image, label, metadata, bbox
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            return self._get_dummy_item()

    def _get_dummy_item(self):
        return torch.zeros((3, CFG.IMG_SIZE, CFG.IMG_SIZE)), torch.zeros(len(TRAIN_CLASSES)), torch.zeros(self.metadata_dim), torch.zeros((len(TRAIN_CLASSES), 4))

# --- Model ---
class AdvancedChestModel(nn.Module):
    def __init__(self, model_name='convnext_large', num_classes=14, metadata_features=3, pretrained=False):
        super().__init__()
        self.num_classes = num_classes
        self.train_classes = len(TRAIN_CLASSES)  # Use actual number of training classes
        self.bbox_classes = len(CLASSES_WITH_BBOX)  # Number of classes with bounding boxes
        
        # Base model
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, features_only=False)
        self.model.set_grad_checkpointing(enable=True)
        
        # Get feature dimension
        in_features = 1536  # ConvNext Large fixed dimension
        
        # Enhanced attention mechanism for better feature focus
        self.attention = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, in_features // 8),  # Wider attention module
            nn.GELU(),
            nn.Dropout(0.05),  # Light dropout in attention
            nn.Linear(in_features // 8, in_features),
            nn.Sigmoid()
        )
        
        # Enhanced localization head for better bounding box prediction
        self.localization_head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(0.05),  # Light dropout
            nn.Linear(in_features, in_features // 2),
            nn.GELU(),
            nn.LayerNorm(in_features // 2),
            nn.Dropout(0.05),  # Additional dropout layer
            nn.Linear(in_features // 2, in_features // 4),  # Additional layer
            nn.GELU(),
            nn.LayerNorm(in_features // 4),
            nn.Linear(in_features // 4, self.bbox_classes * 4)  # 4 coordinates per bbox class
        )
        
        # Metadata processing branch
        self.metadata_branch = nn.Sequential(
            nn.Linear(metadata_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        # Classification head
        self.combined_fc = nn.Sequential(
            nn.LayerNorm(in_features + 128),
            nn.Dropout(CFG.DROPOUT_HEAD),
            nn.Linear(in_features + 128, in_features // 2),  # Additional layer
            nn.GELU(),
            nn.LayerNorm(in_features // 2),
            nn.Dropout(CFG.DROPOUT_HEAD / 2),  # Reduced dropout
            nn.Linear(in_features // 2, self.train_classes)
        )

    def forward(self, x_img, x_meta):
        # Extract image features
        img_features = self.model(x_img)
        
        # Apply attention
        attention_weights = self.attention(img_features)
        attended_features = img_features * attention_weights
        
        # Bounding box prediction
        bbox_out = self.localization_head(attended_features)
        bbox_out = bbox_out.view(-1, self.bbox_classes, 4)
        bbox_out = torch.sigmoid(bbox_out)  # Normalize bbox coordinates to [0, 1]
        
        # Process metadata
        if x_meta.shape[1] != self.metadata_branch[0].in_features:
            if x_meta.shape[1] < self.metadata_branch[0].in_features:
                padding = torch.zeros(x_meta.shape[0], 
                                   self.metadata_branch[0].in_features - x_meta.shape[1], 
                                   device=x_meta.device)
                x_meta = torch.cat([x_meta, padding], dim=1)
            else:
                x_meta = x_meta[:, :self.metadata_branch[0].in_features]
        
        meta_features = self.metadata_branch(x_meta)
        
        # Combine features for classification
        combined_features = torch.cat([attended_features, meta_features], dim=1)
        cls_out = self.combined_fc(combined_features)
        
        return cls_out, bbox_out

# --- Transforms ---
def get_transforms(img_size, is_train=True):
    if ALBUMENTATIONS_INSTALLED:
        if is_train:
            return A.Compose([
                A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_AREA),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.2),  # Reduced probability
                A.CoarseDropout(max_holes=6, max_height=int(img_size*0.10), max_width=int(img_size*0.10),
                                min_holes=1, min_height=int(img_size*0.04), min_width=int(img_size*0.04),
                                fill_value=0, mask_fill_value=None, p=0.2),  # Further reduced for fine-tuning
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=10, p=0.3, border_mode=cv2.BORDER_REFLECT101),  # Smaller transformations
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
                A.GaussNoise(var_limit=(5.0, 20.0), p=0.1),  # Reduced noise
                A.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10, hue=0.05, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # Additional augmentations for localization focus
                A.GridDistortion(p=0.1, distort_limit=0.05),  # Light grid distortion
                A.OpticalDistortion(p=0.1, distort_limit=0.05, shift_limit=0.05),  # Light optical distortion
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_AREA),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    else:
        print("Using basic Torchvision transforms...")
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if is_train:
            return transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            return transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                normalize,
            ])

# --- Loss Calculation ---
def calculate_loss(cls_outputs, bbox_outputs, labels, bboxes, criterion_cls, criterion_bbox, device, mixup=False, y_a=None, y_b=None, lam=1.0):
    # Classification loss
    labels = labels.to(device, non_blocking=True)
    if mixup and y_a is not None and y_b is not None:
        cls_loss = mixup_criterion(criterion_cls, cls_outputs, y_a.to(device), y_b.to(device), lam)
    else:
        cls_loss = criterion_cls(cls_outputs, labels)
    
    # Bounding box loss with improved weighting
    bboxes = bboxes.to(device, non_blocking=True)
    bbox_mask = (bboxes.sum(dim=-1) > 0).float()
    bbox_loss = torch.tensor(0.0, device=device)
    
    if bbox_mask.sum() > 0:
        # Only compute loss for classes that have bounding boxes
        bbox_indices = [TRAIN_CLASSES.index(cls) for cls in CLASSES_WITH_BBOX]
        bbox_outputs_filtered = bbox_outputs[:, :len(CLASSES_WITH_BBOX), :]
        bboxes_filtered = bboxes[:, bbox_indices, :]
        bbox_mask_filtered = bbox_mask[:, bbox_indices]
        
        # Expand mask for all 4 coordinates
        bbox_mask_expanded = bbox_mask_filtered.unsqueeze(-1).expand_as(bbox_outputs_filtered)
        
        # L1 loss for coordinate regression
        l1_loss = F.smooth_l1_loss(
            bbox_outputs_filtered * bbox_mask_expanded,
            bboxes_filtered * bbox_mask_expanded,
            reduction='sum'
        )
        
        # IoU loss for better box prediction
        iou_loss = torch.tensor(0.0, device=device)
        valid_boxes = bbox_mask_filtered.sum()
        if valid_boxes > 0:
            pred_boxes = bbox_outputs_filtered * bbox_mask_expanded
            true_boxes = bboxes_filtered * bbox_mask_expanded
            
            # Convert to x1,y1,x2,y2 format
            pred_x1y1 = pred_boxes[..., :2]
            pred_x2y2 = pred_x1y1 + pred_boxes[..., 2:]
            true_x1y1 = true_boxes[..., :2]
            true_x2y2 = true_x1y1 + true_boxes[..., 2:]
            
            # Calculate IoU
            intersect_x1y1 = torch.max(pred_x1y1, true_x1y1)
            intersect_x2y2 = torch.min(pred_x2y2, true_x2y2)
            intersect_wh = (intersect_x2y2 - intersect_x1y1).clamp(min=0)
            intersect = intersect_wh[..., 0] * intersect_wh[..., 1]
            
            pred_area = pred_boxes[..., 2] * pred_boxes[..., 3]
            true_area = true_boxes[..., 2] * true_boxes[..., 3]
            union = pred_area + true_area - intersect
            
            iou = intersect / (union + 1e-6)
            iou_loss = (1 - iou).mean()
        
        # Combine losses with weights
        bbox_loss = (0.4 * l1_loss / (valid_boxes + 1e-6)) + (0.6 * iou_loss)  # Increased IoU loss weight
    
    # Total loss with adjusted weights
    total_loss = cls_loss + CFG.BBOX_LOSS_WEIGHT * bbox_loss  # Use configurable weight for bbox loss
    
    return total_loss, cls_loss, bbox_loss

# --- Checkpoint Loading ---
def safe_load_checkpoint(model, optimizer, scheduler, checkpoint_path, device, load_strict):
    start_epoch = 0
    best_auroc = CFG.MIN_AUROC
    metrics = defaultdict(list)
    reset_optimizer_scheduler = False
    epochs_no_improve = 0
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            if next(iter(state_dict)).startswith('module.'):
                state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
            load_result = model.load_state_dict(state_dict, strict=load_strict)
            if load_strict:
                print(f"Checkpoint loaded successfully from '{checkpoint_path}' (Strict=True).")
                if 'optimizer_state_dict' in checkpoint and optimizer is not None:
                    try:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        for state in optimizer.state.values():
                            for k, v in state.items():
                                if isinstance(v, torch.Tensor):
                                    state[k] = v.to(device)
                        print("Optimizer state loaded.")
                    except Exception as e:
                        print(f"Warning: Could not load optimizer state dict: {e}. Optimizer will be reset.")
                        reset_optimizer_scheduler = True
                if 'scheduler_state_dict' in checkpoint and scheduler is not None and not reset_optimizer_scheduler:
                    try:
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                        print("Scheduler state loaded.")
                    except Exception as e:
                        print(f"Warning: Could not load scheduler state dict: {e}. Scheduler might restart.")
            else:
                print(f"Partial weights loaded with strict=False from '{checkpoint_path}'.")
                if load_result.missing_keys:
                    print(f"Missing keys: {load_result.missing_keys}")
                if load_result.unexpected_keys:
                    print(f"Unexpected keys: {load_result.unexpected_keys}")
                print("Optimizer and Scheduler state will NOT be loaded due to strict=False. Training continues from Epoch 0 with loaded backbone.")
                reset_optimizer_scheduler = True
            start_epoch = checkpoint.get('epoch', 0) if load_strict and not reset_optimizer_scheduler else 0
            best_auroc = max(checkpoint.get('best_auroc', CFG.MIN_AUROC), CFG.MIN_AUROC)
            loaded_metrics_dict = checkpoint.get('metrics', {})
            if isinstance(loaded_metrics_dict, dict):
                metrics = defaultdict(list, loaded_metrics_dict)
            epochs_no_improve = 0
            if start_epoch > 0:
                print(f"Resuming training from epoch {start_epoch + 1}. Best AUROC so far: {best_auroc:.4f}")
            else:
                print(f"Starting training from epoch 1 (Optimizer/Scheduler reset). Previous best AUROC was: {best_auroc:.4f}")
        except Exception as e:
            print(f"Error loading checkpoint from {checkpoint_path}: {e}. Training from scratch.")
            start_epoch = 0
            best_auroc = CFG.MIN_AUROC
            metrics = defaultdict(list)
            reset_optimizer_scheduler = True
            epochs_no_improve = 0
    else:
        print("No checkpoint found. Training from scratch.")
        best_auroc = CFG.MIN_AUROC
        reset_optimizer_scheduler = True
        epochs_no_improve = 0
    return start_epoch, best_auroc, metrics, reset_optimizer_scheduler, epochs_no_improve

# --- Training and Validation Functions ---
def train_one_epoch(model, loader, optimizer, criterion_cls, criterion_bbox, device, scaler, epoch, scheduler):
    model.train()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_bbox_loss = 0.0
    progress_bar = tqdm(loader, desc=f'Epoch {epoch+1}/{CFG.EPOCHS} (Train)', leave=False)
    optimizer.zero_grad(set_to_none=True)
    num_optimizer_steps_in_epoch = math.ceil(len(loader)/CFG.ACCUM_STEPS)
    num_warmup_optimizer_steps = CFG.WARMUP_EPOCHS * num_optimizer_steps_in_epoch
    error_count = 0
    max_errors = 10
    # Freeze Backbone for first 3 epochs to preserve pre-trained weights
    if epoch < CFG.WARMUP_EPOCHS:
        for param in model.model.parameters():
            param.requires_grad = False
        print(f"Epoch {epoch+1}: Backbone frozen, training only Head.")
    else:
        for param in model.model.parameters():
            param.requires_grad = True
        print(f"Epoch {epoch+1}: Backbone unfrozen, training all layers.")
    for i, data in enumerate(progress_bar):
        try:
            if i % 50 == 0:
                torch.cuda.empty_cache()
            inputs, labels, metadata, bboxes = data
            inputs = inputs.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            bboxes = bboxes.to(device, non_blocking=True)
            # Apply Mixup for first half of training with 50% chance
            use_mixup = (epoch < CFG.EPOCHS // 2) and np.random.rand() < 0.5
            if use_mixup:
                inputs, y_a, y_b, lam = mixup_data(inputs, labels, CFG.MIXUP_ALPHA, device)
            with torch.amp.autocast(device_type=str(device), dtype=torch.float16):
                combined_out, bbox_out = model(inputs, metadata)
                if use_mixup:
                    total_loss, cls_loss, bbox_loss_val = calculate_loss(
                        combined_out, bbox_out, labels, bboxes, criterion_cls, criterion_bbox, device,
                        mixup=True, y_a=y_a, y_b=y_b, lam=lam
                    )
                else:
                    total_loss, cls_loss, bbox_loss_val = calculate_loss(
                        combined_out, bbox_out, labels, bboxes, criterion_cls, criterion_bbox, device
                    )
                loss_scaled = total_loss / CFG.ACCUM_STEPS
            scaler.scale(loss_scaled).backward()
            if (i + 1) % CFG.ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CFG.GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler and CFG.SCHEDULER == 'CosineAnnealingLR':
                    current_total_optimizer_step = (epoch * num_optimizer_steps_in_epoch) + ((i+1)//CFG.ACCUM_STEPS) - 1
                    if current_total_optimizer_step < num_warmup_optimizer_steps:
                        lr_scale = float(current_total_optimizer_step + 1) / float(max(1, num_warmup_optimizer_steps))
                        for pg_idx, pg in enumerate(optimizer.param_groups):
                            base_lr = CFG.HEAD_LR if pg_idx > 0 else CFG.LR
                            pg['lr'] = base_lr * lr_scale
                    elif current_total_optimizer_step == num_warmup_optimizer_steps:
                        for pg_idx, pg in enumerate(optimizer.param_groups):
                            base_lr = CFG.HEAD_LR if pg_idx > 0 else CFG.LR
                            pg['lr'] = base_lr
                        scheduler.step()
                    else:
                        scheduler.step()
            running_loss += total_loss.item()
            running_cls_loss += cls_loss.item()
            running_bbox_loss += bbox_loss_val.item()
            progress_bar.set_postfix({'loss': f'{total_loss.item():.4f}', 'cls_l': f'{cls_loss.item():.4f}', 'box_l': f'{bbox_loss_val.item():.4f}', 'lr': f"{optimizer.param_groups[0]['lr']:.1e}"})
        except Exception as e:
            error_count += 1
            print(f"\nError during training step {i}: {e}")
            if error_count >= max_errors:
                print(f"\nToo many errors ({error_count}). Stopping training.")
                break
            torch.cuda.empty_cache()
            continue
    avg_loss = running_loss / max(1, len(progress_bar) - error_count)
    avg_cls_loss = running_cls_loss / max(1, len(progress_bar) - error_count)
    avg_bbox_loss = running_bbox_loss / max(1, len(progress_bar) - error_count)
    return avg_loss, avg_cls_loss, avg_bbox_loss

def validate_one_epoch(model, loader, criterion_cls, criterion_bbox, device, epoch):
    model.eval()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_bbox_loss = 0.0
    all_targets = []
    all_outputs = []
    no_finding_correct = 0
    no_finding_total = 0
    best_no_finding_threshold = 0.25
    best_no_finding_accuracy = 0.0
    threshold_range = np.arange(0.1, 0.5, 0.05)
    confusion_matrices = {}
    progress_bar = tqdm(loader, desc=f'Epoch {epoch+1}/{CFG.EPOCHS} (Valid)', leave=False)
    with torch.no_grad():
        for inputs, labels, metadata, bboxes in progress_bar:
            inputs = inputs.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=str(device), dtype=torch.float16):
                combined_out, bbox_out = model(inputs, metadata)
                total_loss, cls_loss, bbox_loss_val = calculate_loss(combined_out, bbox_out, labels, bboxes, criterion_cls, criterion_bbox, device)
            running_loss += total_loss.item()
            running_cls_loss += cls_loss.item()
            running_bbox_loss += bbox_loss_val.item()
            outputs_sigmoid = combined_out.sigmoid().cpu()
            all_outputs.append(outputs_sigmoid)
            all_targets.append(labels.cpu())
            progress_bar.set_postfix({'val_loss': f'{total_loss.item():.4f}'})
        all_outputs = torch.cat(all_outputs, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        for i, cls_name in enumerate(TRAIN_CLASSES):
            true_labels = all_targets[:, i]
            pred_labels = (all_outputs[:, i] >= CFG.CONFUSION_MATRIX_THRESHOLD).astype(int)
            try:
                cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
                confusion_matrices[cls_name] = cm
            except Exception as e:
                print(f"Error computing confusion matrix for {cls_name}: {e}")
                confusion_matrices[cls_name] = np.zeros((2, 2), dtype=int)
        is_no_finding = np.all(all_targets == 0, axis=1).astype(int)
        max_probs = np.max(all_outputs, axis=1)
        for threshold in threshold_range:
            no_finding_correct = 0
            no_finding_total = 0
            pred_no_finding = (max_probs < threshold).astype(int)
            try:
                cm_no_finding = confusion_matrix(is_no_finding, pred_no_finding, labels=[0, 1])
                no_finding_total = is_no_finding.sum()
                no_finding_correct = cm_no_finding[1, 1]
                accuracy = no_finding_correct / max(no_finding_total, 1) * 100
                if accuracy > best_no_finding_accuracy:
                    best_no_finding_accuracy = accuracy
                    best_no_finding_threshold = threshold
                    confusion_matrices['No Finding'] = cm_no_finding
            except Exception as e:
                print(f"Error computing confusion matrix for No Finding: {e}")
        aurocs = {}
        valid_auroc_scores = []
        for i, cls_name in enumerate(TRAIN_CLASSES):
            try:
                if len(np.unique(all_targets[:, i])) > 1:
                    score = roc_auc_score(all_targets[:, i], all_outputs[:, i])
                    aurocs[cls_name] = score
                    valid_auroc_scores.append(score)
                else:
                    aurocs[cls_name] = np.nan
            except ValueError:
                aurocs[cls_name] = np.nan
        mean_auroc = np.nanmean(valid_auroc_scores) if valid_auroc_scores else 0.0
        avg_loss = running_loss / len(loader)
        avg_cls_loss = running_cls_loss / len(loader)
        avg_bbox_loss = running_bbox_loss / len(loader)
        print(f"  No Finding Accuracy: {best_no_finding_accuracy:.2f}% ({no_finding_correct}/{no_finding_total}) at threshold {best_no_finding_threshold:.2f}")
    return avg_loss, avg_cls_loss, avg_bbox_loss, mean_auroc, aurocs, all_targets, all_outputs, best_no_finding_accuracy, best_no_finding_threshold, confusion_matrices

# --- Plotting Functions ---
def plot_training_progress(metrics, num_epochs, save_path='training_progress.png'):
    epochs = list(range(1, len(metrics['train_loss']) + 1))
    if not epochs:
        return
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, metrics['train_loss'], label='Train Total Loss', marker='.')
    plt.plot(epochs, metrics['val_loss'], label='Valid Total Loss', marker='.')
    plt.plot(epochs, metrics['train_cls_loss'], label='Train Class Loss', linestyle='--')
    plt.plot(epochs, metrics['val_cls_loss'], label='Valid Class Loss', linestyle='--')
    plt.title('Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 3, 2)
    plt.plot(epochs, metrics['mean_auroc'], label='Mean AUROC', color='blue', marker='.')
    plt.axhline(y=CFG.MIN_AUROC, color='red', linestyle='--', label=f'Min AUROC ({CFG.MIN_AUROC:.4f})')
    plt.title('Mean Validation AUROC Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('AUROC')
    plt.legend()
    plt.grid(True)
    y_min = min(metrics.get('mean_auroc', [0])) - 0.05
    y_max = max(metrics.get('mean_auroc', [1])) + 0.05
    plt.ylim(max(0, y_min), min(1.05, y_max))
    plt.subplot(1, 3, 3)
    plt.plot(epochs, metrics['train_bbox_loss'], label='Train Bbox Loss', marker='.', linestyle=':')
    plt.plot(epochs, metrics['val_bbox_loss'], label='Valid Bbox Loss', marker='.', linestyle=':')
    plt.title('Bounding Box Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_class_metrics(class_aurocs, save_path='class_aurocs.png'):
    if not class_aurocs:
        return
    plt.figure(figsize=(15, 7))
    classes = list(class_aurocs.keys())
    scores = [s if not np.isnan(s) else 0 for s in class_aurocs.values()]
    colors = ['red' if np.isnan(class_aurocs.get(cls, np.nan)) else 'blue' for cls in classes]
    plt.bar(classes, scores, color=colors)
    plt.xticks(rotation=60, ha='right')
    plt.title('Validation AUROC Score per Class (NaN means not calculable)')
    plt.xlabel('Classes')
    plt.ylabel('AUROC Score')
    plt.ylim(0, 1)
    for i, v in enumerate(scores):
        display_val = f'{v:.3f}' if not np.isnan(class_aurocs.get(classes[i], np.nan)) else 'N/A'
        plt.text(i, v + 0.01, display_val, ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curves(targets, outputs, class_names, epoch, mean_auc, save_path='roc_curves.png'):
    plt.figure(figsize=(12, 10))
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance (AUC = 0.500)')
    valid_classes_plotted = 0
    for i, name in enumerate(class_names):
        y_true = targets[:, i]
        y_score = outputs[:, i]
        if len(np.unique(y_true)) < 2:
            continue
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            auc_score = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc_score:.3f})')
            valid_classes_plotted += 1
        except Exception as e:
            print(f"Could not plot ROC for class '{name}': {e}")
    if valid_classes_plotted == 0:
        print("No valid ROC curves to plot.")
        plt.close()
        return
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - Epoch {epoch+1} (Mean AUC: {mean_auc:.4f})')
    plt.legend(loc="lower right", fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrices(confusion_matrices, class_names, epoch, save_path='confusion_matrices'):
    os.makedirs(save_path, exist_ok=True)
    for cls_name in class_names:
        cm = confusion_matrices.get(cls_name, np.zeros((2, 2), dtype=int))
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix for {cls_name} - Epoch {epoch+1}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'confusion_matrix_{cls_name}epoch{epoch+1}.png'))
        plt.close()

def save_emergency_checkpoint(model, optimizer, scheduler, metrics, epoch, loss, auroc):
    try:
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': dict(metrics) if metrics else {},
            'last_loss': loss,
            'best_auroc': auroc
        }
        emergency_path = f'emergency_checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint_data, emergency_path)
        print(f"\nSaved emergency checkpoint to {emergency_path}")
        return True
    except Exception as e:
        print(f"\nFailed to save emergency checkpoint: {e}")
        return False

# --- Main Execution ---
if __name__ == '__main__':
    print(f"--- Starting Training Run ---")
    print(f"Device: {CFG.DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0)}" if CFG.DEVICE == torch.device('cuda') else "CPU")
    print(f"Config: {vars(CFG)}")
    print("Loading data entry...")
    df_main = pd.read_csv(CFG.DATA_ENTRY_PATH)
    df = df_main.copy()
    df_findings = df[df['Finding Labels'] != 'No Finding'].copy()
    df_nofinding = df[df['Finding Labels'] == 'No Finding'].copy()
    nofinding_sample_fraction = 0.35
    print(f"Using No Finding sample fraction: {nofinding_sample_fraction}")
    df_nofinding_sampled = df_nofinding.sample(frac=nofinding_sample_fraction, random_state=CFG.SEED)
    df_combined = pd.concat([df_findings, df_nofinding_sampled], ignore_index=True)
    df_processed = df_combined.sample(frac=1, random_state=CFG.SEED).reset_index(drop=True)
    print(f"Dataset size: {len(df_processed)} (after sampling)")
    print("Preprocessing metadata...")
    df_processed, _, _ = preprocess_metadata(df_processed)
    print("Metadata OK.")
    print("Splitting data...")
    train_df, valid_df = train_test_split(df_processed, test_size=0.2, random_state=CFG.SEED)
    print(f"Train: {len(train_df)}, Valid: {len(valid_df)}")
    print("Loading bboxes...")
    bbox_dict = load_bounding_boxes(CFG.BBOX_PATH)
    print(f"Loaded {len(bbox_dict)} images with bboxes.")
    train_transform = get_transforms(CFG.IMG_SIZE, is_train=True)
    valid_transform = get_transforms(CFG.IMG_SIZE, is_train=False)
    print("Creating Datasets...")
    train_dataset = ChestXrayDataset(CFG.IMAGE_DIR, train_df, bbox_dict, transform=train_transform, train=True)
    valid_dataset = ChestXrayDataset(CFG.IMAGE_DIR, valid_df, bbox_dict, transform=valid_transform, train=False)
    print("Datasets OK.")
    print("Calculating sample weights...")
    labels_np = train_dataset.labels
    pos_counts = np.maximum(labels_np.sum(axis=0), 1e-6)
    class_weights_sampler = 1.0 / pos_counts
    no_finding_samples = np.all(labels_np == 0, axis=1)
    sample_weights = np.maximum(np.max(labels_np * class_weights_sampler, axis=1), 1e-6)
    sample_weights[no_finding_samples] *= 2.0  # Reduced to 2x to balance focus on diseases
    print(f"Applied boosted weight (2.0x) to {np.sum(no_finding_samples)} 'No Finding' samples")
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    print("Sampler OK.")
    print("Creating DataLoaders...")
    train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, sampler=sampler, num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True, persistent_workers=(CFG.NUM_WORKERS > 0))
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.BATCH_SIZE * 2, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=False, persistent_workers=(CFG.NUM_WORKERS > 0))
    print("DataLoaders OK.")
    print(f"Initializing model: {CFG.MODEL_NAME}...")
    model = AdvancedChestModel(CFG.MODEL_NAME, num_classes=len(CLASS_NAMES), metadata_features=train_dataset.metadata_dim, pretrained=False)
    model.to(CFG.DEVICE)
    print("Model OK.")
    print("Setting up optimizer...")
    backbone_params = [p for n, p in model.model.named_parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and 'model.' not in n]
    optimizer = torch.optim.AdamW([{'params': backbone_params, 'lr': CFG.LR, 'name': 'backbone'}, {'params': head_params, 'lr': CFG.HEAD_LR, 'name': 'heads'}], weight_decay=CFG.WEIGHT_DECAY)
    print("Optimizer OK.")
    print(f"Setting up scheduler: {CFG.SCHEDULER}...")
    num_train_optimizer_steps = math.ceil(len(train_loader) / CFG.ACCUM_STEPS) * CFG.EPOCHS
    num_warmup_optimizer_steps = math.ceil(len(train_loader) / CFG.ACCUM_STEPS) * CFG.WARMUP_EPOCHS
    scheduler = None
    if CFG.SCHEDULER == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_train_optimizer_steps - num_warmup_optimizer_steps, eta_min=CFG.MIN_LR)
    elif CFG.SCHEDULER == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=2, verbose=True, min_lr=CFG.MIN_LR)
    if scheduler:
        print(f"Scheduler OK. Type: {CFG.SCHEDULER}")
    else:
        print("Scheduler Disabled.")
    print(f"Setting up loss functions...")
    criterion_cls = FocalLoss(label_smoothing=CFG.LABEL_SMOOTHING).to(CFG.DEVICE) if CFG.LOSS_FN == 'FocalLoss' else nn.BCEWithLogitsLoss().to(CFG.DEVICE)
    criterion_bbox = nn.SmoothL1Loss(reduction='mean').to(CFG.DEVICE)
    print("Loss functions OK.")
    start_epoch = 0
    best_auroc = 0.0
    metrics = defaultdict(list)
    if not CFG.START_FROM_SCRATCH:
        start_epoch, best_auroc, metrics, reset_optimizer_scheduler, epochs_no_improve = safe_load_checkpoint(
            model, optimizer, scheduler, CFG.CHECKPOINT_LOAD_PATH, CFG.DEVICE, CFG.LOAD_STRICT
        )
        if not isinstance(metrics, defaultdict):
            metrics = defaultdict(list, metrics if metrics else {})
        if reset_optimizer_scheduler:
            print("Resetting optimizer and scheduler states...")
            optimizer = torch.optim.AdamW([{'params': backbone_params, 'lr': CFG.LR, 'name': 'backbone'}, {'params': head_params, 'lr': CFG.HEAD_LR, 'name': 'heads'}], weight_decay=CFG.WEIGHT_DECAY)
            if CFG.SCHEDULER == 'CosineAnnealingLR':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_train_optimizer_steps - num_warmup_optimizer_steps, eta_min=CFG.MIN_LR)
            elif CFG.SCHEDULER == 'ReduceLROnPlateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=2, verbose=True, min_lr=CFG.MIN_LR)
            else:
                scheduler = None
            start_epoch = 0
            epochs_no_improve = 0
            metrics = defaultdict(list)
            best_auroc = CFG.MIN_AUROC
            print("Starting training from Epoch 1 due to optimizer/scheduler reset.")
    else:
        print("Starting training from scratch as requested.")
        best_auroc = CFG.MIN_AUROC
        epochs_no_improve = 0
    scaler = torch.amp.GradScaler()
    print(f"\n--- Starting/Resuming Training from Epoch {start_epoch + 1} ---")
    import time
    start_time = time.time()
    for epoch in range(start_epoch, CFG.EPOCHS):
        epoch_start_time = time.time()
        train_loss, train_cls_loss, train_bbox_loss = train_one_epoch(model, train_loader, optimizer, criterion_cls, criterion_bbox, CFG.DEVICE, scaler, epoch, scheduler)
        val_loss, val_cls_loss, val_bbox_loss, mean_auroc, class_aurocs, all_targets, all_outputs, no_finding_accuracy, no_finding_threshold, confusion_matrices = validate_one_epoch(model, valid_loader, criterion_cls, criterion_bbox, CFG.DEVICE, epoch)
        metrics['train_loss'].append(train_loss)
        metrics['train_cls_loss'].append(train_cls_loss)
        metrics['train_bbox_loss'].append(train_bbox_loss)
        metrics['val_loss'].append(val_loss)
        metrics['val_cls_loss'].append(val_cls_loss)
        metrics['val_bbox_loss'].append(val_bbox_loss)
        metrics['mean_auroc'].append(mean_auroc)
        metrics['no_finding_accuracy'].append(no_finding_accuracy)
        current_class_aurocs = {}
        for cls, score in class_aurocs.items():
            current_class_aurocs[cls] = score
            metrics[f'auroc_{cls}'].append(score if not np.isnan(score) else 0.0)
        epoch_duration = time.time() - epoch_start_time
        print(f"\nEpoch {epoch+1}/{CFG.EPOCHS} Summary (Duration: {epoch_duration:.2f}s):")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e} (Backbone), {optimizer.param_groups[1]['lr']:.2e} (Heads)")
        print(f"  Train Loss: {train_loss:.4f} (Cls: {train_cls_loss:.4f}, Box: {train_bbox_loss:.4f})")
        print(f"  Valid Loss: {val_loss:.4f} (Cls: {val_cls_loss:.4f}, Box: {val_bbox_loss:.4f})")
        print(f"  Mean Valid AUROC: {mean_auroc:.4f} (Best: {best_auroc:.4f})")
        print(f"  No Finding Accuracy: {no_finding_accuracy:.2f}% at threshold {no_finding_threshold:.2f}")
        if (epoch + 1) % 2 == 0:
            save_emergency_checkpoint(model, optimizer, scheduler, metrics, epoch, val_loss, mean_auroc)
        if scheduler and CFG.SCHEDULER == 'ReduceLROnPlateau':
            scheduler.step(mean_auroc)
        resume_checkpoint_path = CFG.CHECKPOINT_LOAD_PATH
        try:
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_auroc': best_auroc,
                'metrics': dict(metrics)
            }
            torch.save(checkpoint_data, resume_checkpoint_path)
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            save_emergency_checkpoint(model, optimizer, scheduler, metrics, epoch, val_loss, mean_auroc)
        if mean_auroc > 0.88:
            if best_auroc > CFG.MIN_AUROC and metrics['mean_auroc'] and len(metrics['mean_auroc']) > 1:
                prev_aurocs = metrics['mean_auroc'][:-1]
                if prev_aurocs:
                    prev_best_idx = np.argmax(prev_aurocs)
                    prev_best_auroc_val = prev_aurocs[prev_best_idx]
                    old_best_path = f'best_model_epoch_{prev_best_idx+1}auroc{prev_best_auroc_val:.4f}.pth'
                    if os.path.exists(old_best_path):
                        try:
                            os.remove(old_best_path)
                            print(f"Deleted old best model: {old_best_path}")
                        except OSError as e:
                            print(f"Error deleting old model {old_best_path}: {e}")
            best_auroc = max(best_auroc, mean_auroc)
            best_model_path = f'best_model_epoch_{epoch+1}auroc{mean_auroc:.4f}.pth'
            torch.save(model.state_dict(), best_model_path)
            print(f"*** Saved new model with AUROC: {mean_auroc:.4f} at {best_model_path} ***")
            epochs_no_improve = 0
            print("Generating plots for best epoch...")
            plot_training_progress(metrics, CFG.EPOCHS, save_path=f'training_progress_best_epoch_{epoch+1}.png')
            plot_class_metrics(current_class_aurocs, save_path=f'class_aurocs_best_epoch_{epoch+1}.png')
            plot_roc_curves(all_targets, all_outputs, TRAIN_CLASSES, epoch, mean_auroc, save_path=f'roc_curves_best_epoch_{epoch+1}.png')
            plot_confusion_matrices(confusion_matrices, TRAIN_CLASSES + ['No Finding'], epoch, save_path=f'confusion_matrices_best_epoch_{epoch+1}')
        else:
            epochs_no_improve += 1
            print(f"Validation AUROC did not improve significantly for {epochs_no_improve} epoch(s). Patience: {epochs_no_improve}/{CFG.PATIENCE}")
        if epochs_no_improve >= CFG.PATIENCE or mean_auroc < CFG.MIN_AUROC:
            print(f"\nEarly stopping triggered after {CFG.PATIENCE} epochs without improvement or AUROC dropped below {CFG.MIN_AUROC:.4f}.")
            save_emergency_checkpoint(model, optimizer, scheduler, metrics, epoch, val_loss, best_auroc)
            break
    total_training_time = time.time() - start_time
    print(f"\n--- Training finished in {total_training_time/60:.2f} minutes. Best validation AUROC achieved: {best_auroc:.4f} ---")
    print("\n--- Updating diagnose_image.py with optimal threshold ---")
    with open('update_diagnose_threshold.py', 'w') as f:
        f.write(f"""
# Script to update diagnose_image.py threshold
import os
import re

# Optimal threshold from training
NEW_THRESHOLD = {no_finding_threshold:.2f}

def update_diagnose_script():
    if not os.path.exists('diagnose_image.py'):
        print("diagnose_image.py not found")
        return False
    with open('diagnose_image.py', 'r') as file:
        content = file.read()
    updated_content = re.sub(
        r'THRESHOLD\\s*=\\s*0\\.\\d+',
        f'THRESHOLD = {NEW_THRESHOLD}',
        content
    )
    with open('diagnose_image.py', 'w') as file:
        file.write(updated_content)
    print(f"Updated diagnose_image.py threshold to {NEW_THRESHOLD}")
    return True

if __name__ == "__main__":
    update_diagnose_script()
""")
    print("Created threshold update script. Run 'python update_diagnose_threshold.py' to apply the optimal threshold.")