import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms
from torchcam.methods import GradCAMpp
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import shap
import base64
import sys
import seaborn as sns
import random
from torchcam.methods import SmoothGradCAMpp, LayerCAM
from torchcam.utils import overlay_mask
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import re
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
class CFG:
    SEED = 42
    MODEL_NAME = 'convnextv2_large.fcmae_ft_in22k_in1k'
    IMG_SIZE = 320
    BATCH_SIZE = 2
    ACCUM_STEPS = 32
    NUM_WORKERS = 2
    EPOCHS = 60
    LR = 3e-6
    HEAD_LR = 1e-5
    WEIGHT_DECAY = 0.07
    SCHEDULER = 'CosineAnnealingLR'
    T_MAX = EPOCHS
    MIN_LR = 1e-8
    WARMUP_EPOCHS = 7
    LOSS_FN = 'FocalLoss'
    FOCAL_ALPHA = 0.5
    FOCAL_GAMMA = 2.0
    LABEL_SMOOTHING = 0.05
    PATIENCE = 12
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DROPOUT_HEAD = 0.25
    DROPOUT_META = 0.2
    BASE_PATH = 'data'
    IMAGE_DIR = BASE_PATH
    DATA_ENTRY_PATH = os.path.join(BASE_PATH, 'Data_Entry_2017.csv')
    BBOX_PATH = os.path.join(BASE_PATH, 'BBox_List_2017.csv')
    CHECKPOINT_PATH = 'best_model_convnext_v2_large.pth'
    MIN_AUROC = 0.80
    MIXUP_ALPHA = 0.2
    HEATMAP_THRESHOLD = 0.5
    OUTPUT_DIR = 'output'
    ROC_PATH = os.path.join(OUTPUT_DIR, 'roc_curve.png')
    CM_PATH = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    HEATMAP_EXAMPLE_PATH = os.path.join(OUTPUT_DIR, 'heatmap_example.png')
    LEARNING_CURVE_PATH = os.path.join(OUTPUT_DIR, 'learning_curve_epoch_{epoch}.png')
    SAVE_CHECKPOINTS = True  # Flag to indicate if we should save multiple checkpoints
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')  # Directory to save multiple checkpoints

# Create checkpoint directory if enabled
if CFG.SAVE_CHECKPOINTS:
    os.makedirs(CFG.CHECKPOINT_DIR, exist_ok=True)

os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

torch.manual_seed(CFG.SEED)
np.random.seed(CFG.SEED)
if CFG.DEVICE.type == 'cuda':
    torch.cuda.manual_seed(CFG.SEED)

CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 
               'Hernia', 'No Finding']
TRAIN_CLASSES = [c for c in CLASS_NAMES if c != 'No Finding']
CLASSES_WITH_BBOX = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
                     'Pneumonia', 'Pneumothorax']

class FocalLoss(nn.Module):
    def __init__(self, alpha=CFG.FOCAL_ALPHA, gamma=CFG.FOCAL_GAMMA, label_smoothing=CFG.LABEL_SMOOTHING):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    def forward(self, inputs, targets):
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing / inputs.shape[1]
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

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

def find_image_path(image_name, base_folder):
    for i in range(1, 13):
        image_path = os.path.join(base_folder, f'images_{i:03d}', 'images', image_name)
        if os.path.exists(image_path):
            return image_path
    image_path_flat = os.path.join(base_folder, 'images', image_name)
    if os.path.exists(image_path_flat):
        return image_path_flat
    return None

class ChestXrayDataset(Dataset):
    def __init__(self, df, bbox_df, image_dir, transform=None):
        self.df = df
        self.bbox_df = bbox_df
        self.image_dir = image_dir
        self.transform = transform
        self.labels = self._prepare_labels()
        self.metadata = self._prepare_metadata()
        self.bboxes = self._prepare_bboxes()

    def _prepare_labels(self):
        labels = []
        for _, row in self.df.iterrows():
            findings = row['Finding Labels'].split('|')
            label = np.zeros(len(TRAIN_CLASSES), dtype=np.float32)
            for finding in findings:
                if finding in TRAIN_CLASSES:
                    label[TRAIN_CLASSES.index(finding)] = 1.0
            labels.append(label)
        return np.array(labels)

    def _prepare_metadata(self):
        metadata = []
        for _, row in self.df.iterrows():
            age = min(row['Patient Age'], 100) / 100.0
            gender = 1.0 if row['Patient Gender'] == 'M' else 0.0
            view = 1.0 if row['View Position'] == 'PA' else 0.0
            metadata.append([age, gender, view])
        return np.array(metadata, dtype=np.float32)

    def _prepare_bboxes(self):
        bboxes = []
        for _, row in self.df.iterrows():
            img_bboxes = self.bbox_df[self.bbox_df['Image Index'] == row['Image Index']]
            bbox = np.zeros((len(CLASSES_WITH_BBOX), 4), dtype=np.float32)
            for _, bbox_row in img_bboxes.iterrows():
                disease = bbox_row['Finding Label']
                if disease in CLASSES_WITH_BBOX:
                    idx = CLASSES_WITH_BBOX.index(disease)
                    x = bbox_row['Bbox [x'] / 1024.0
                    y = bbox_row['y'] / 1024.0
                    w = bbox_row['w'] / 1024.0
                    h = bbox_row['h]'] / 1024.0
                    bbox[idx] = [x, y, w, h]
            bboxes.append(bbox)
        return np.array(bboxes)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_name = self.df.iloc[idx]['Image Index']
        img_path = find_image_path(image_name, self.image_dir)
        if img_path is None:
            raise FileNotFoundError(f"Image {image_name} not found in {self.image_dir}")
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        
        if self.transform:
            img = self.transform(image=img)['image']
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        metadata = torch.tensor(self.metadata[idx], dtype=torch.float32)
        bbox = torch.tensor(self.bboxes[idx], dtype=torch.float32)
        
        return img, label, metadata, bbox

class AdvancedChestModel(nn.Module):
    def __init__(self, num_classes=len(TRAIN_CLASSES), metadata_features=3):
        super().__init__()
        # Create model without pretrained weights to avoid SSL download issues
        self.backbone = timm.create_model(CFG.MODEL_NAME, pretrained=False, num_classes=0)
        self.feature_dim = self.backbone(torch.randn(1, 3, CFG.IMG_SIZE, CFG.IMG_SIZE)).shape[1]
        
        self.attention = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim // 16),
            nn.GELU(),
            nn.Linear(self.feature_dim // 16, self.feature_dim),
            nn.Sigmoid()
        )
        
        self.metadata_branch = nn.Sequential(
            nn.Linear(metadata_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(CFG.DROPOUT_META),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        self.combined_fc = nn.Sequential(
            nn.LayerNorm(self.feature_dim + 128),
            nn.Dropout(CFG.DROPOUT_HEAD),
            nn.Linear(self.feature_dim + 128, num_classes)
        )
        
        self.localization_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, len(CLASSES_WITH_BBOX) * 4)
        )

    def forward(self, x, metadata):
        with torch.amp.autocast(device_type=CFG.DEVICE.type):
            features = self.backbone(x)
            attention_weights = self.attention(features)
            features = features * attention_weights
            metadata_features = self.metadata_branch(metadata)
            combined_features = torch.cat([features, metadata_features], dim=1)
            cls_out = self.combined_fc(combined_features)
            bbox_out = self.localization_head(features)
            bbox_out = bbox_out.view(-1, len(CLASSES_WITH_BBOX), 4)
            return cls_out, bbox_out, features

def generate_heatmap(model, image, class_idx, cam_extractor, device, original_size):
    model.eval()
    image = image.unsqueeze(0).to(device)
    try:
        with torch.no_grad():
            cls_out, _, features = model(image, torch.zeros(1, 3).to(device))
            # Modified to handle the case where cam_extractor is dummy or fails
            try:
                if hasattr(cam_extractor, '__call__'):
                    cam = cam_extractor(class_idx, cls_out)
                else:
                    return None
                    
                if cam is None or len(cam) == 0 or cam[0] is None:
                    print("CAM extraction returned None")
                    return None
                    
                heatmap = cam[0].cpu().numpy()
                heatmap = cv2.resize(heatmap, original_size)
                heatmap = np.clip(heatmap, 0, 1)
                heatmap = (heatmap * 255).astype(np.uint8)
                return heatmap
            except Exception as e:
                print(f"Error in CAM extraction: {str(e)}")
                return None
    except Exception as e:
        print(f"Error in generate_heatmap: {str(e)}")
        return None

def explain_with_shap(model, image, metadata, device, class_idx):
    model.eval()
    def model_predict(inputs):
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        metadata_batch = metadata.repeat(inputs.shape[0], 1).to(device)
        with torch.no_grad():
            cls_out, _, _ = model(inputs, metadata_batch)
            return torch.sigmoid(cls_out).cpu().numpy()
    
    image = image.unsqueeze(0).to(device)
    background = image.repeat(100, 1, 1, 1)
    explainer = shap.KernelExplainer(model_predict, background.cpu().numpy())
    shap_values = explainer.shap_values(image.cpu().numpy(), nsamples=50)
    
    return shap_values[class_idx]

def generate_text_explanation(shap_values, metadata, class_name):
    explanation = f"The model predicted {class_name} because: "
    if shap_values.max() > 0.1:
        explanation += "a specific pattern in the image (e.g., lung collapse). "
    if metadata[0] > 40:
        explanation += f"and the patient's age ({metadata[0]} years) increased the likelihood."
    return explanation

def plot_roc_curve(y_true, y_pred, class_names, save_path=CFG.ROC_PATH):
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        if np.sum(y_true[:, i]) == 0 or np.sum(y_true[:, i]) == len(y_true[:, i]):
            continue
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=CFG.CM_PATH):
    plt.figure(figsize=(15, 10))
    for i, class_name in enumerate(class_names):
        if np.sum(y_true[:, i]) == 0 or np.sum(y_true[:, i]) == len(y_true[:, i]):
            continue
        y_pred_binary = (y_pred[:, i] > 0.5).astype(int)
        cm = confusion_matrix(y_true[:, i], y_pred_binary)
        plt.subplot(4, 4, i+1)
        plt.imshow(cm, cmap='Blues')
        plt.title(f'{class_name}')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def get_transforms(train=True):
    if train:
        return A.Compose([
            A.Resize(height=CFG.IMG_SIZE, width=CFG.IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(p=0.2),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(height=CFG.IMG_SIZE, width=CFG.IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

def train_model(resume_from_checkpoint=True, specific_checkpoint_to_load=None):
    """
    Main training loop for the chest X-ray model.
    Handles model initialization, data loading, training, validation,
    checkpointing, and plotting of learning curves.
    Can resume from a general checkpoint or a specific model-state checkpoint.
    """
    # Ensure output directories exist
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    os.makedirs(CFG.CHECKPOINT_DIR, exist_ok=True)

    print(f"Using device: {CFG.DEVICE}")
    print(f"Output directory: {CFG.OUTPUT_DIR}")
    df = pd.read_csv(CFG.DATA_ENTRY_PATH)
    bbox_df = pd.read_csv(CFG.BBOX_PATH)
    df_findings = df[df['Finding Labels'] != 'No Finding']
    df_nofinding = df[df['Finding Labels'] == 'No Finding'].sample(frac=0.35, random_state=CFG.SEED)
    df_combined = pd.concat([df_findings, df_nofinding], ignore_index=True).sample(frac=1, random_state=CFG.SEED)
    train_df, valid_df = train_test_split(df_combined, test_size=0.2, random_state=CFG.SEED)

    train_dataset = ChestXrayDataset(train_df, bbox_df, CFG.IMAGE_DIR, get_transforms(train=True))
    valid_dataset = ChestXrayDataset(valid_df, bbox_df, CFG.IMAGE_DIR, get_transforms(train=False))
    train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)

    # Initialize model (without pretrained weights to avoid SSL download)
    model = AdvancedChestModel().to(CFG.DEVICE)
    
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': CFG.LR},
        {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': CFG.HEAD_LR}
    ], weight_decay=CFG.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS, eta_min=CFG.MIN_LR)
    criterion_cls = FocalLoss().to(CFG.DEVICE)
    criterion_bbox = nn.SmoothL1Loss().to(CFG.DEVICE)
    scaler = torch.amp.GradScaler()

    # Variables for training state - defaults
    start_epoch = 0
    best_auroc = CFG.MIN_AUROC
    patience_counter = 0
    train_losses = []
    valid_losses = []
    valid_aurocs = []
    
    # Path for the general full-state checkpoint
    general_checkpoint_path = os.path.join(CFG.OUTPUT_DIR, 'training_checkpoint.pth')
    
    checkpoint_loaded_successfully = False

    # First, try loading from the specific checkpoint (our priority)
    if specific_checkpoint_to_load and os.path.exists(specific_checkpoint_to_load):
        print(f"Attempting to load MODEL STATE from specific checkpoint: {specific_checkpoint_to_load}")
        try:
            # This type of checkpoint is expected to only contain model_state_dict
            model.load_state_dict(torch.load(specific_checkpoint_to_load, map_location=CFG.DEVICE))
            print("Model state loaded successfully from specific checkpoint.")
            checkpoint_loaded_successfully = True

            # Try to parse epoch and best_auroc from filename
            fname = os.path.basename(specific_checkpoint_to_load)
            match = re.search(r"model_epoch_(\d+)_auroc_([0-9.]+)\.pth", fname)
            if match:
                parsed_epoch_num = int(match.group(1))  # 1-indexed completed epoch
                start_epoch = parsed_epoch_num  # Next epoch to run is this number (0-indexed in loop)
                best_auroc = float(match.group(2))
                print(f"Resuming from epoch {start_epoch} with best AUROC: {best_auroc:.4f} (from filename).")
                print("Optimizer, scheduler, and history will start fresh.")
                # Re-initialize optimizer and scheduler
                optimizer = torch.optim.AdamW([
                    {'params': model.backbone.parameters(), 'lr': CFG.LR},
                    {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': CFG.HEAD_LR}
                ], weight_decay=CFG.WEIGHT_DECAY)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS, eta_min=CFG.MIN_LR)
                patience_counter = 0
                train_losses, valid_losses, valid_aurocs = [], [], []
            else:
                print(f"Could not parse epoch/AUROC from filename '{fname}'. Using loaded model weights with default start_epoch=0 and fresh optimizer/scheduler.")
                # Model is loaded, other params are default. Optimizer/scheduler are already fresh from above.
        except Exception as e:
            print(f"Failed to load specific checkpoint {specific_checkpoint_to_load}: {str(e)}")
            checkpoint_loaded_successfully = False # Ensure it's false if specific load fails
            # Reset states to default if specific load failed partway
            start_epoch = 0; best_auroc = CFG.MIN_AUROC; patience_counter = 0; train_losses = []; valid_losses = []; valid_aurocs = []
            # Re-initialize model to be safe
            model = AdvancedChestModel().to(CFG.DEVICE)
            optimizer = torch.optim.AdamW([
                {'params': model.backbone.parameters(), 'lr': CFG.LR},
                {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': CFG.HEAD_LR}
            ], weight_decay=CFG.WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS, eta_min=CFG.MIN_LR)

    # If specific checkpoint was not loaded (or not provided), try the general one
    if not checkpoint_loaded_successfully and resume_from_checkpoint and os.path.exists(general_checkpoint_path):
        print(f"Loading general (full training state) checkpoint from {general_checkpoint_path}")
        try:
            full_checkpoint_data = torch.load(general_checkpoint_path, map_location=CFG.DEVICE)
            print(f"General checkpoint loaded successfully. Keys: {list(full_checkpoint_data.keys())}")
            model.load_state_dict(full_checkpoint_data['model_state_dict'])
            print("Model state loaded successfully from general checkpoint.")
            optimizer.load_state_dict(full_checkpoint_data['optimizer_state_dict'])
            print("Optimizer state loaded successfully from general checkpoint.")
            scheduler.load_state_dict(full_checkpoint_data['scheduler_state_dict'])
            print("Scheduler state loaded successfully from general checkpoint.")
            start_epoch = full_checkpoint_data['epoch'] + 1
            best_auroc = full_checkpoint_data['best_auroc']
            patience_counter = full_checkpoint_data['patience_counter']
            train_losses = full_checkpoint_data['train_losses']
            valid_losses = full_checkpoint_data['valid_losses']
            valid_aurocs = full_checkpoint_data['valid_aurocs']
            print(f"Resuming from epoch {start_epoch} with best AUROC: {best_auroc:.4f}, patience: {patience_counter}")
            print(f"Training history loaded: {len(train_losses)} epochs of data from general checkpoint.")
            
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(CFG.DEVICE)
            checkpoint_loaded_successfully = True
        except Exception as e:
            print(f"Failed to load general checkpoint: {str(e)}")
            print("Error details:", e.__class__.__name__)
            import traceback
            traceback.print_exc()
            print("Starting training from scratch as general checkpoint load failed.")
            # Ensure defaults if general load fails after specific was not attempted or failed
            start_epoch = 0; best_auroc = CFG.MIN_AUROC; patience_counter = 0; train_losses = []; valid_losses = []; valid_aurocs = []
            # Re-initialize model, optimizer, scheduler if general checkpoint load fails
            model = AdvancedChestModel().to(CFG.DEVICE)
            optimizer = torch.optim.AdamW([
                {'params': model.backbone.parameters(), 'lr': CFG.LR},
                {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': CFG.HEAD_LR}
            ], weight_decay=CFG.WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS, eta_min=CFG.MIN_LR)
            
    if not checkpoint_loaded_successfully:
        print("Starting training from scratch (no valid checkpoint loaded or resume_from_checkpoint is False).")
        # Ensure model, optimizer, scheduler are freshly initialized (already done by definitions above or in except blocks)
        # and start_epoch, best_auroc, etc., are at their defaults.

    for epoch in range(start_epoch, CFG.EPOCHS):
        try:
            if epoch < CFG.WARMUP_EPOCHS:
                for param in model.backbone.parameters():
                    param.requires_grad = False
            else:
                for param in model.backbone.parameters():
                    param.requires_grad = True

            model.train()
            train_loss = 0
            for i, (images, labels, metadata, bboxes) in enumerate(tqdm(train_loader)):
                # Clear cache between batches to reduce memory pressure
                if CFG.DEVICE.type == 'cuda':
                    torch.cuda.empty_cache()
                    
                images, labels, metadata, bboxes = images.to(CFG.DEVICE), labels.to(CFG.DEVICE), metadata.to(CFG.DEVICE), bboxes.to(CFG.DEVICE)

                if np.random.rand() < 0.5:
                    images, labels_a, labels_b, lam = mixup_data(images, labels)
                    with torch.amp.autocast(device_type=CFG.DEVICE.type):
                        cls_out, bbox_out, _ = model(images, metadata)
                        cls_loss = mixup_criterion(criterion_cls, cls_out, labels_a, labels_b, lam)
                        bbox_loss = criterion_bbox(bbox_out, bboxes)
                        loss = cls_loss + 0.05 * bbox_loss
                else:
                    with torch.amp.autocast(device_type=CFG.DEVICE.type):
                        cls_out, bbox_out, _ = model(images, metadata)
                        cls_loss = criterion_cls(cls_out, labels)
                        bbox_loss = criterion_bbox(bbox_out, bboxes)
                        loss = cls_loss + 0.05 * bbox_loss

                scaler.scale(loss / CFG.ACCUM_STEPS).backward()
                if (i + 1) % CFG.ACCUM_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                # Use safe way to accumulate loss
                with torch.no_grad():
                    train_loss += loss.item() * images.size(0)

            train_loss /= len(train_loader.dataset)
            train_losses.append(train_loss)  # Store training loss

            # Clear memory before validation
            if CFG.DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

            model.eval()
            valid_preds = []
            valid_labels = []
            valid_loss = 0
            with torch.no_grad():
                for images, labels, metadata, bboxes in tqdm(valid_loader):
                    images, labels, metadata, bboxes = images.to(CFG.DEVICE), labels.to(CFG.DEVICE), metadata.to(CFG.DEVICE), bboxes.to(CFG.DEVICE)
                    with torch.amp.autocast(device_type=CFG.DEVICE.type):
                        cls_out, bbox_out, _ = model(images, metadata)
                        cls_loss = criterion_cls(cls_out, labels)
                        bbox_loss = criterion_bbox(bbox_out, bboxes)
                        loss = cls_loss + 0.05 * bbox_loss
                    valid_loss += loss.item() * images.size(0)
                    valid_preds.append(torch.sigmoid(cls_out).cpu().numpy())
                    valid_labels.append(labels.cpu().numpy())

            valid_loss /= len(valid_loader.dataset)
            valid_losses.append(valid_loss)  # Store validation loss
            
            valid_preds_concat = np.concatenate(valid_preds)
            valid_labels_concat = np.concatenate(valid_labels)

            auroc_scores = []
            for i in range(len(TRAIN_CLASSES)):
                if np.sum(valid_labels_concat[:, i]) > 0 and np.sum(valid_labels_concat[:, i]) < len(valid_labels_concat[:, i]):
                    auroc = roc_auc_score(valid_labels_concat[:, i], valid_preds_concat[:, i])
                    auroc_scores.append(auroc)
            
            valid_auroc = np.mean(auroc_scores)
            valid_aurocs.append(valid_auroc)  # Store validation AUROC

            print(f"Epoch {epoch+1}/{CFG.EPOCHS} - Train Loss: {train_loss:.4f} - Valid Loss: {valid_loss:.4f} - Valid AUROC: {valid_auroc:.4f}")

            # Save model checkpoint if improved
            if valid_auroc > best_auroc:
                best_auroc = valid_auroc
                torch.save(model.state_dict(), CFG.CHECKPOINT_PATH)
                print(f"Saved model with AUROC: {best_auroc:.4f}")
                patience_counter = 0
                
                # Save additional checkpoint with epoch number if enabled
                if CFG.SAVE_CHECKPOINTS:
                    epoch_checkpoint_path = os.path.join(CFG.CHECKPOINT_DIR, f'model_epoch_{epoch+1}_auroc_{valid_auroc:.4f}.pth')
                    torch.save(model.state_dict(), epoch_checkpoint_path)
                    print(f"Also saved checkpoint at: {epoch_checkpoint_path}")
            else:
                patience_counter += 1
                
            # Periodically save model (every 10 epochs) regardless of performance
            if CFG.SAVE_CHECKPOINTS and (epoch + 1) % 10 == 0:
                periodic_checkpoint_path = os.path.join(CFG.CHECKPOINT_DIR, f'model_periodic_epoch_{epoch+1}.pth')
                torch.save(model.state_dict(), periodic_checkpoint_path)
                print(f"Saved periodic checkpoint at epoch {epoch+1}")

            # Save complete training state for resuming later if needed
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_auroc': best_auroc,
                'patience_counter': patience_counter,
                'train_losses': train_losses,
                'valid_losses': valid_losses,
                'valid_aurocs': valid_aurocs,
            }, general_checkpoint_path)

            if patience_counter >= CFG.PATIENCE:
                print("Early stopping triggered")
                break

            scheduler.step()
            
            # Plot learning curves after every 20 epochs or at the end of training
            if (epoch + 1) % 20 == 0 or epoch == CFG.EPOCHS - 1 or patience_counter >= CFG.PATIENCE:
                plt.figure(figsize=(15, 5))
                
                # Plot training and validation loss
                plt.subplot(1, 2, 1)
                plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
                plt.plot(range(1, len(valid_losses) + 1), valid_losses, label='Valid Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.title('Training and Validation Loss')
                plt.legend()
                plt.grid(True)
                
                # Plot validation AUROC
                plt.subplot(1, 2, 2)
                plt.plot(range(1, len(valid_aurocs) + 1), valid_aurocs, label='Valid AUROC', color='green')
                plt.xlabel('Epochs')
                plt.ylabel('AUROC')
                plt.title('Validation AUROC')
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                # Save the learning curve plot
                learning_curve_save_path = CFG.LEARNING_CURVE_PATH.format(epoch=epoch+1)
                plt.savefig(learning_curve_save_path)
                print(f"Learning curve plot saved to {learning_curve_save_path}")
                plt.close()
                
        except RuntimeError as e:
            print(f"Error during training: {str(e)}")
            print("Saving current state and exiting...")
            
            # Save checkpoint even on error
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_auroc': best_auroc,
                'patience_counter': patience_counter,
                'train_losses': train_losses,
                'valid_losses': valid_losses,
                'valid_aurocs': valid_aurocs,
            }, general_checkpoint_path)
            
            # Before re-raising the error, make sure plots are saved
            if len(train_losses) > 0:
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 2, 1)
                plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
                plt.plot(range(1, len(valid_losses) + 1), valid_losses, label='Valid Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.title('Training and Validation Loss')
                plt.legend()
                plt.grid(True)
                
                if len(valid_aurocs) > 0:
                    plt.subplot(1, 2, 2)
                    plt.plot(range(1, len(valid_aurocs) + 1), valid_aurocs, label='Valid AUROC', color='green')
                    plt.xlabel('Epochs')
                    plt.ylabel('AUROC')
                    plt.title('Validation AUROC')
                    plt.legend()
                    plt.grid(True)
                
                plt.tight_layout()
                emergency_path = os.path.join(CFG.OUTPUT_DIR, f'learning_curves_error_epoch_{epoch+1}.png')
                plt.savefig(emergency_path)
                plt.close()
                print(f"Emergency curve saved to {emergency_path}")
            
            # Don't re-raise to allow loading model for inference
            # raise e
            break

    # Final plots after training completes or stops
    if len(train_losses) > 0:
        plt.figure(figsize=(15, 5))
        
        # Plot training and validation loss
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
        plt.plot(range(1, len(valid_losses) + 1), valid_losses, label='Valid Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot validation AUROC
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(valid_aurocs) + 1), valid_aurocs, label='Valid AUROC', color='green')
        plt.xlabel('Epochs')
        plt.ylabel('AUROC')
        plt.title('Validation AUROC')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        final_curves_path = os.path.join(CFG.OUTPUT_DIR, 'final_learning_curves.png')
        plt.savefig(final_curves_path)
        plt.close()

    # ROC curve and confusion matrix for the last valid evaluation
    if len(valid_preds) > 0:
        valid_preds_concat = np.concatenate(valid_preds)
        valid_labels_concat = np.concatenate(valid_labels)
        plot_roc_curve(valid_labels_concat, valid_preds_concat, TRAIN_CLASSES)
        plot_confusion_matrix(valid_labels_concat, valid_preds_concat, TRAIN_CLASSES)

    # Load the best model for inference
    model.load_state_dict(torch.load(CFG.CHECKPOINT_PATH, map_location=CFG.DEVICE))
    
    # Fix GradCAMpp initialization
    try:
        # This approach is safer than hardcoding a layer name
        # Find a suitable layer from the backbone for visualization
        target_layer = None
        for name, module in model.backbone.named_modules():
            # Look for the last convolutional layer or a suitable feature extraction layer
            if isinstance(module, nn.Conv2d) or 'norm' in name:
                target_layer = module
                print(f"Found suitable target layer for GradCAM: {name}")
                break
                
        if target_layer is None:
            print("Could not find a suitable layer for GradCAM, using a fallback approach")
            # Fallback: try to use the last stage of the ConvNeXt model
            target_layer = model.backbone.stages[-1]
            
        cam_extractor = GradCAMpp(model, target_layer=target_layer)
    except Exception as e:
        print(f"Error initializing GradCAMpp: {str(e)}")
        print("Using dummy CAM extractor as fallback")
        
        # Create a dummy CAM extractor that returns None
        class DummyCAMExtractor:
            def __call__(self, *args, **kwargs):
                return [None]
                
        cam_extractor = DummyCAMExtractor()
        
    return model, cam_extractor, valid_labels_concat if len(valid_preds) > 0 else None, valid_preds_concat if len(valid_preds) > 0 else None

if __name__ == '__main__':
    # Allow skipping resume with a command line flag
    skip_resume = '--skip-resume' in sys.argv
    user_specific_checkpoint = os.path.join(CFG.CHECKPOINT_DIR, 'model_epoch_19_auroc_0.8320.pth') # User's specified checkpoint

    # Check if we should reduce the batch size to avoid CUDA errors
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        print(f"GPU memory: {gpu_mem:.2f} GB")
        if gpu_mem < 8.0:  # Less than 8GB of VRAM
            print("Low GPU memory detected, reducing batch size and image size")
            CFG.BATCH_SIZE = 1
            CFG.ACCUM_STEPS = 64
    
    # Try to run with the checkpoint-loading approach
    try:
        model, cam_extractor, valid_labels, valid_preds = train_model(
            resume_from_checkpoint=not skip_resume,
            specific_checkpoint_to_load=user_specific_checkpoint # Pass the specific checkpoint path
        )
        
    except Exception as e:
        # Print the error but continue to fallback approach
        print(f"Error during model training: {str(e)}")
        print("Attempting fallback approach with direct model loading...")
        
        # Fallback: try to directly load the model for inference only
        try:
            model = AdvancedChestModel().to(CFG.DEVICE)
            model.load_state_dict(torch.load(user_specific_checkpoint, map_location=CFG.DEVICE))
            print(f"Successfully loaded model from {user_specific_checkpoint}")
            print("Model is now ready for inference. Training was skipped.")
            
            # Handle CAM extractor creation for the loaded model
            try:
                target_layer = None
                for name, module in model.backbone.named_modules():
                    if isinstance(module, nn.Conv2d) or 'norm' in name:
                        target_layer = module
                        print(f"Found suitable target layer for GradCAM: {name}")
                        break
                
                if target_layer is None:
                    target_layer = model.backbone.stages[-1]
                
                from pytorch_grad_cam import GradCAMPlusPlus
                cam_extractor = GradCAMPlusPlus(model, target_layer=target_layer)
                valid_labels, valid_preds = None, None
                
            except Exception as cam_error:
                print(f"Error creating GradCAM: {cam_error}")
                # Create a dummy CAM extractor
                class DummyCAMExtractor:
                    def __call__(self, *args, **kwargs):
                        return [None]
                cam_extractor = DummyCAMExtractor()
                valid_labels, valid_preds = None, None
            
        except Exception as load_error:
            print(f"Could not load model: {load_error}")
            print("Cannot proceed with inference. Exiting.")
            sys.exit(1)

    # Skip visualization if we don't have validation data
    if valid_labels is None or valid_preds is None:
        print("Training was interrupted. Loading checkpoint for inference only.")
        try:
            model.load_state_dict(torch.load(CFG.CHECKPOINT_PATH, map_location=CFG.DEVICE))
            print(f"Loaded best model from {CFG.CHECKPOINT_PATH}")
        except Exception as e:
            print(f"Could not load model: {str(e)}")
            exit(1)

    # Clear GPU memory before visualization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    transform = get_transforms(train=False)
    
    try:
        df = pd.read_csv(CFG.DATA_ENTRY_PATH)
        img_path = find_image_path(df.iloc[0]['Image Index'], CFG.IMAGE_DIR)
        if img_path is None:
            print("Could not find sample image path")
        else:
            img = Image.open(img_path).convert('RGB')
            original_size = img.size
            img_np = np.array(img)
            img_transformed = transform(image=img_np)['image']

            class_idx = TRAIN_CLASSES.index('Pneumothorax')
            
            # Generate heatmap
            try:
                heatmap = generate_heatmap(model, img_transformed, class_idx, cam_extractor, CFG.DEVICE, original_size)
                if heatmap is not None:
                    plt.imshow(img, cmap='gray')
                    plt.imshow(heatmap, cmap='jet', alpha=0.5)
                    plt.savefig(CFG.HEATMAP_EXAMPLE_PATH)
                    plt.close()
                    print(f"Saved heatmap to {CFG.HEATMAP_EXAMPLE_PATH}")
            except Exception as e:
                print(f"Error generating heatmap: {str(e)}")
            
            # Generate SHAP explanation
            try:
                metadata = torch.tensor([50.0, 1.0, 0.0]).to(CFG.DEVICE)
                shap_values = explain_with_shap(model, img_transformed, metadata, CFG.DEVICE, class_idx)
                print(f"SHAP values for Pneumothorax: {shap_values}")

                explanation = generate_text_explanation(shap_values, metadata.cpu().numpy(), 'Pneumothorax')
                print(f"Explanation: {explanation}")
            except Exception as e:
                print(f"Error generating SHAP explanation: {str(e)}")
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        
    print("Done!")
