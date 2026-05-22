import os
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from tqdm import tqdm
import timm
import random
from pathlib import Path

# Constants
DATA_PATH = "/kaggle/input/data"  # Kaggle path
TRAIN_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 
    'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]
CLASS_NAMES = ['No Finding'] + TRAIN_CLASSES

# Focal Loss implementation for handling class imbalance
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
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# Custom weighted loss that gives more importance to "No Finding"
class WeightedMultiLabelLoss(nn.Module):
    def __init__(self, pos_weight=None, no_finding_idx=0, no_finding_weight=2.0):
        super(WeightedMultiLabelLoss, self).__init__()
        self.pos_weight = pos_weight
        self.no_finding_idx = no_finding_idx
        self.no_finding_weight = no_finding_weight
        
    def forward(self, y_pred, y_true):
        # Standard BCE loss
        loss = F.binary_cross_entropy_with_logits(y_pred, y_true, pos_weight=self.pos_weight)
        
        # Add extra weight to "No Finding" errors
        no_finding_true = y_true[:, self.no_finding_idx]
        no_finding_pred = y_pred[:, self.no_finding_idx]
        no_finding_loss = F.binary_cross_entropy_with_logits(no_finding_pred, no_finding_true)
        
        # Total loss with extra weight on "No Finding"
        total_loss = loss + (self.no_finding_weight - 1) * no_finding_loss
        return total_loss

# Dataset class with better preprocessing and balancing
class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, use_metadata=True, 
                 balance_classes=True, max_no_finding=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.use_metadata = use_metadata
        
        # Process labels
        self.df['labels'] = self.df['Finding Labels'].apply(self._parse_labels)
        
        # Balance classes if requested
        if balance_classes:
            self._balance_classes(max_no_finding)
        
        # Prepare one-hot encoded labels
        self.labels = []
        for _, row in self.df.iterrows():
            label = np.zeros(len(CLASS_NAMES))
            for cls in row['labels']:
                if cls in CLASS_NAMES:
                    label[CLASS_NAMES.index(cls)] = 1
            self.labels.append(label)
        
        self.labels = np.array(self.labels)
    
    def _parse_labels(self, label_string):
        labels = [l.strip() for l in label_string.split('|')]
        return labels
    
    def _balance_classes(self, max_no_finding=None):
        # Separate "No Finding" from other diseases
        no_finding_df = self.df[self.df['Finding Labels'] == 'No Finding']
        disease_df = self.df[self.df['Finding Labels'] != 'No Finding']
        
        # Limit "No Finding" samples if requested
        if max_no_finding and len(no_finding_df) > max_no_finding:
            no_finding_df = no_finding_df.sample(max_no_finding, random_state=42)
        
        # Combine back
        self.df = pd.concat([disease_df, no_finding_df])
        
        # Shuffle
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['Image Index'])
        
        # Read and process image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            # Handle missing image
            image = np.zeros((512, 512), dtype=np.uint8)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Process metadata if using it
        if self.use_metadata:
            # Extract patient metadata (age, gender, view position)
            age = float(row.get('Patient Age', 50)) / 100.0  # Normalize to 0-1
            
            # Gender: 0 for female, 1 for male
            gender = row.get('Patient Gender', 'M')
            gender_encoded = 1.0 if gender == 'M' else 0.0
            
            # View position
            view = row.get('View Position', 'PA')
            view_positions = {
                'PA': 0, 'AP': 1, 'L': 2, 'LATERAL': 2,
                'AP SUPINE': 3, 'AP_SUPINE': 3, 'SUPINE': 3
            }
            view_encoded = view_positions.get(view, 0) / 3.0  # Normalize to 0-1
            
            metadata = torch.tensor([age, gender_encoded, view_encoded], dtype=torch.float32)
            return image, torch.FloatTensor(self.labels[idx]), metadata
        else:
            return image, torch.FloatTensor(self.labels[idx])

# Improved model with better handling for "No Finding"
class ImprovedXrayModel(nn.Module):
    def __init__(self, num_classes=15, backbone='densenet121', pretrained=True, dropout_rate=0.2, metadata_features=3):
        super(ImprovedXrayModel, self).__init__()
        self.use_metadata = metadata_features > 0
        
        # Initialize backbone model
        if backbone == 'densenet121':
            self.backbone = timm.create_model('densenet121', pretrained=pretrained)
            self.feature_dim = 1024
        elif backbone == 'resnet50':
            self.backbone = timm.create_model('resnet50', pretrained=pretrained)
            self.feature_dim = 2048
        elif backbone == 'efficientnet_b3':
            self.backbone = timm.create_model('efficientnet_b3', pretrained=pretrained)
            self.feature_dim = 1536
        elif backbone == 'convnext_base':
            self.backbone = timm.create_model('convnext_base', pretrained=pretrained)
            self.feature_dim = 1024
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Modify the backbone for our task
        if 'densenet' in backbone:
            self.backbone.classifier = nn.Identity()
        elif 'resnet' in backbone:
            self.backbone.fc = nn.Identity()
        elif 'efficientnet' in backbone:
            self.backbone.classifier = nn.Identity()
        elif 'convnext' in backbone:
            self.backbone.head.fc = nn.Identity()
        
        # Metadata processing branch
        if self.use_metadata:
            self.metadata_fc1 = nn.Linear(metadata_features, 64)
            self.metadata_fc2 = nn.Linear(64, 128)
            self.metadata_bn1 = nn.BatchNorm1d(64)
            self.metadata_bn2 = nn.BatchNorm1d(128)
            
            # Combined features
            self.combined_dim = self.feature_dim + 128
            self.combined_fc = nn.Linear(self.combined_dim, 512)
            self.combined_bn = nn.BatchNorm1d(512)
        else:
            self.combined_dim = self.feature_dim
            self.combined_fc = nn.Linear(self.combined_dim, 512)
            self.combined_bn = nn.BatchNorm1d(512)
        
        # Output layers
        self.fc = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Special binary classifier just for "No Finding" vs "Finding"
        self.no_finding_classifier = nn.Linear(512, 1)
    
    def forward(self, x, metadata=None):
        # Get image features from backbone
        img_features = self.backbone(x)
        
        # Process metadata if available
        if self.use_metadata and metadata is not None:
            meta_features = F.relu(self.metadata_bn1(self.metadata_fc1(metadata)))
            meta_features = F.relu(self.metadata_bn2(self.metadata_fc2(meta_features)))
            
            # Combine features
            combined = torch.cat((img_features, meta_features), dim=1)
        else:
            combined = img_features
        
        # Final processing
        features = F.relu(self.combined_bn(self.combined_fc(combined)))
        features = self.dropout(features)
        
        # Main output for all classes
        output = self.fc(features)
        
        # Special output just for "No Finding" detection
        no_finding_output = self.no_finding_classifier(features)
        
        return no_finding_output, output, features

# Model ensemble class
class ModelEnsemble(nn.Module):
    def __init__(self, models, weights=None):
        super(ModelEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        
        # Equal weights by default
        if weights is None:
            self.weights = torch.ones(len(models)) / len(models)
        else:
            self.weights = torch.tensor(weights) / sum(weights)
    
    def forward(self, x, metadata=None):
        no_finding_outputs = []
        outputs = []
        features_list = []
        
        for i, model in enumerate(self.models):
            no_finding, output, features = model(x, metadata)
            
            no_finding_outputs.append(no_finding * self.weights[i])
            outputs.append(output * self.weights[i])
            features_list.append(features * self.weights[i])
        
        # Sum the weighted outputs
        no_finding_output = torch.stack(no_finding_outputs).sum(dim=0)
        output = torch.stack(outputs).sum(dim=0)
        features = torch.stack(features_list).sum(dim=0)
        
        return no_finding_output, output, features

# Improved prediction function with better handling of "No Finding"
def predict_with_verification(model, image_tensor, metadata_tensor=None, 
                             threshold=0.5, no_finding_threshold=0.6, 
                             device='cuda'):
    """Make prediction with improved No Finding verification"""
    # Move tensors to device
    image_tensor = image_tensor.to(device)
    if metadata_tensor is not None:
        metadata_tensor = metadata_tensor.to(device)
    
    # Get prediction
    with torch.no_grad(), torch.cuda.amp.autocast():
        no_finding_out, class_out, _ = model(image_tensor, metadata_tensor)
        
        # Get probability for No Finding (special classifier)
        no_finding_prob = torch.sigmoid(no_finding_out)[0]
        
        # Get probabilities for all classes
        class_probs = torch.sigmoid(class_out)[0]
    
    # Move predictions to CPU
    no_finding_prob = no_finding_prob.cpu().item()
    class_probs = class_probs.cpu().numpy()
    
    # Create mask for standard prediction
    predictions = class_probs >= threshold
    
    # Get maximum probability of any disease
    max_disease_prob = np.max(class_probs[1:])  # Skip "No Finding" at index 0
    
    # Verification logic for "No Finding"
    # If special No Finding classifier is confident OR 
    # if all disease probabilities are low, override to "No Finding"
    if no_finding_prob > no_finding_threshold or max_disease_prob < (1 - no_finding_threshold):
        # Override to mark as "No Finding"
        predictions = np.zeros_like(predictions)
        predictions[0] = True  # Set "No Finding" to True
        
        # Also boost the probability
        class_probs[0] = max(class_probs[0], no_finding_prob)
        
        # Reduce other probabilities
        for i in range(1, len(class_probs)):
            class_probs[i] *= (1 - no_finding_prob)
    
    # Create results
    results = []
    for i, cls_name in enumerate(CLASS_NAMES):
        results.append({
            'disease': cls_name,
            'probability': float(class_probs[i]),
            'predicted': bool(predictions[i])
        })
    
    # Sort by probability (highest first)
    results.sort(key=lambda x: x['probability'], reverse=True)
    
    return results

# Improved image preprocessing with more augmentation
def preprocess_image(image_path, image_size=512, augment=False):
    """Preprocess image for model input with optional augmentation"""
    # Read image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Apply transforms
    if augment:
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.RandomGamma(),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4)),
            ], p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=3),
                A.MedianBlur(blur_limit=3),
            ], p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
    return image_tensor

# Same function as in test_model.py
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

# Improved model loading with support for ensemble
def load_model(checkpoint_paths, device='cuda', ensemble=False):
    """Load one or more models from checkpoints"""
    if isinstance(checkpoint_paths, str):
        checkpoint_paths = [checkpoint_paths]
    
    models = []
    aurocs = {}
    
    for checkpoint_path in checkpoint_paths:
        print(f"Loading model from {checkpoint_path}")
        
        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Get model parameters
            params = checkpoint.get('model_params', {})
            backbone = params.get('backbone', 'densenet121')
            num_classes = params.get('num_classes', len(CLASS_NAMES))
            dropout_rate = params.get('dropout_rate', 0.2)
            metadata_features = params.get('metadata_features', 3)
            
            # Create model
            model = ImprovedXrayModel(
                num_classes=num_classes,
                backbone=backbone,
                pretrained=False,
                dropout_rate=dropout_rate,
                metadata_features=metadata_features
            )
            
            # Load weights
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Try to load directly if model_state_dict is not available
                model.load_state_dict(checkpoint)
            
            # Set model to evaluation mode
            model.eval()
            model = model.to(device)
            models.append(model)
            
            # Get performance metrics if available
            if 'class_aurocs' in checkpoint:
                model_aurocs = checkpoint['class_aurocs']
                print("Model performance (AUROC):")
                for cls, score in model_aurocs.items():
                    print(f"  {cls}: {score:.4f}")
                    aurocs[cls] = aurocs.get(cls, 0) + score
            
        except Exception as e:
            print(f"Error loading model from {checkpoint_path}: {e}")
    
    # Create ensemble if multiple models were loaded successfully
    if len(models) > 1 and ensemble:
        # Use AUROC as weights if available
        if aurocs:
            weights = [sum(model_aurocs.values()) for model_aurocs in [checkpoint.get('class_aurocs', {}) for checkpoint in checkpoints]]
        else:
            weights = None
        
        model = ModelEnsemble(models, weights)
        return model
    elif len(models) > 0:
        return models[0]  # Return single model
    else:
        raise ValueError("No models were loaded successfully")

# Improved visualization
def display_results(image_path, results, threshold=0.5, save_dir="predictions"):
    """Display the image and prediction results with improved visualization"""
    # Read and display the image
    img = Image.open(image_path).convert('RGB')
    
    plt.figure(figsize=(16, 8))
    
    # Image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Chest X-ray')
    plt.axis('off')
    
    # Display predictions
    plt.subplot(1, 2, 2)
    
    # Split into No Finding and diseases
    no_finding = next((r for r in results if r['disease'] == 'No Finding'), None)
    diseases = [r for r in results if r['disease'] != 'No Finding']
    
    # Create combined result list with No Finding at the top if present
    plot_results = [no_finding] if no_finding else []
    plot_results.extend(diseases)
    
    diseases = [r['disease'] for r in plot_results]
    probabilities = [r['probability'] for r in plot_results]
    
    # Coloring: "No Finding" in blue if predicted, diseases in red if predicted
    colors = []
    for r in plot_results:
        if r['disease'] == 'No Finding' and r['predicted']:
            colors.append('darkblue')
        elif r['disease'] == 'No Finding' and not r['predicted']:
            colors.append('lightblue')
        elif r['predicted']:
            colors.append('darkred')
        else:
            colors.append('lightgrey')
    
    # Plot horizontal bars
    bars = plt.barh(range(len(diseases)), probabilities, color=colors)
    plt.yticks(range(len(diseases)), diseases)
    plt.xlabel('Probability')
    plt.title(f'Predicted Conditions (Threshold: {threshold})')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.axvline(x=threshold, color='red', linestyle='--', alpha=0.5)
    
    # Add text labels showing the exact probability values
    for i, (p, disease) in enumerate(zip(probabilities, diseases)):
        plt.text(max(p + 0.02, 0.1), i, f'{p:.2f}', 
                va='center', fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, os.path.basename(image_path).replace('.', '_pred.'))
    plt.savefig(output_path)
    print(f"Prediction visualization saved to {output_path}")
    
    # Show the plot
    plt.show()

# Main function for training
def train_model(args):
    print("Setting up training...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Set up data augmentation for training
    train_transform = A.Compose([
        A.Resize(args.image_size, args.image_size),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.RandomGamma(),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4)),
        ], p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=3),
            A.MedianBlur(blur_limit=3),
        ], p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(args.image_size, args.image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Create datasets
    print("Loading datasets...")
    csv_path = os.path.join(args.data_path, 'Data_Entry_2017.csv')
    image_dir = os.path.join(args.data_path)
    
    # Load training data with class balancing
    train_dataset = ChestXrayDataset(
        csv_file=csv_path,
        image_dir=image_dir,
        transform=train_transform,
        use_metadata=args.use_metadata,
        balance_classes=args.balance_classes,
        max_no_finding=args.max_no_finding
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    print(f"Creating {args.backbone} model...")
    model = ImprovedXrayModel(
        num_classes=len(CLASS_NAMES),
        backbone=args.backbone,
        pretrained=True,
        dropout_rate=args.dropout_rate,
        metadata_features=3 if args.use_metadata else 0
    )
    
    # Move model to device
    model = model.to(device)
    
    # Set up optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Set up learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=args.lr_factor,
        patience=args.patience,
        verbose=True
    )
    
    # Set up loss function
    if args.focal_loss:
        criterion = FocalLoss(gamma=args.focal_gamma)
    elif args.weighted_loss:
        # Compute class weights based on frequency
        pos_weight = torch.ones(len(CLASS_NAMES), device=device)
        counts = np.sum(train_dataset.labels, axis=0)
        total = len(train_dataset)
        for i in range(len(CLASS_NAMES)):
            pos_weight[i] = (total - counts[i]) / max(counts[i], 1)
        criterion = WeightedMultiLabelLoss(pos_weight=pos_weight, no_finding_weight=args.no_finding_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Training loop
    print("Starting training...")
    best_auroc = 0.0
    
    # Create scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        train_steps = 0
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for batch in pbar:
            if args.use_metadata:
                images, labels, metadata = batch
                images, labels, metadata = images.to(device), labels.to(device), metadata.to(device)
            else:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                metadata = None
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    no_finding_out, outputs, _ = model(images, metadata)
                    
                    # No finding binary loss
                    no_finding_loss = F.binary_cross_entropy_with_logits(
                        no_finding_out.squeeze(), 
                        labels[:, 0]  # First label is "No Finding"
                    )
                    
                    # Main classification loss
                    main_loss = criterion(outputs, labels)
                    
                    # Combined loss
                    loss = main_loss + args.no_finding_weight * no_finding_loss
                
                # Backward pass with mixed precision
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training
                no_finding_out, outputs, _ = model(images, metadata)
                
                # No finding binary loss
                no_finding_loss = F.binary_cross_entropy_with_logits(
                    no_finding_out.squeeze(), 
                    labels[:, 0]  # First label is "No Finding"
                )
                
                # Main classification loss
                main_loss = criterion(outputs, labels)
                
                # Combined loss
                loss = main_loss + args.no_finding_weight * no_finding_loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
            
            # Update statistics
            train_loss += loss.item()
            train_steps += 1
            pbar.set_postfix({'loss': loss.item()})
        
        train_loss /= train_steps
        print(f"Training Loss: {train_loss:.4f}")
        
        # Save checkpoint every epoch
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_params': {
                'backbone': args.backbone,
                'num_classes': len(CLASS_NAMES),
                'dropout_rate': args.dropout_rate,
                'metadata_features': 3 if args.use_metadata else 0
            }
        }
        
        torch.save(checkpoint, os.path.join(args.save_dir, f"checkpoint_latest.pth"))
        print(f"Saved latest checkpoint to {os.path.join(args.save_dir, 'checkpoint_latest.pth')}")
    
    print("Training completed!")
    return model

# Main function for testing/inference
def test_model(args):
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load model (single or ensemble)
    if args.ensemble:
        # Use multiple checkpoints
        checkpoint_paths = args.checkpoint.split(',')
        model = load_model(checkpoint_paths, device, ensemble=True)
    else:
        # Use single checkpoint
        model = load_model(args.checkpoint, device)
    
    # Process image
    image_tensor = preprocess_image(args.image, image_size=args.image_size)
    
    # Process metadata
    metadata_tensor = preprocess_metadata(args.age, args.gender, args.view)
    
    # Make prediction with improved verification
    results = predict_with_verification(
        model, 
        image_tensor, 
        metadata_tensor, 
        threshold=args.threshold,
        no_finding_threshold=args.no_finding_threshold,
        device=device
    )
    
    # Display results
    print("\nPrediction Results:")
    print("-" * 60)
    print(f"{'Disease':<20} {'Probability':<15} {'Predicted':<10}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['disease']:<20} {result['probability']:.4f}{' ':<15} {'Yes' if result['predicted'] else 'No':<10}")
    
    # Show visualization
    display_results(args.image, results, args.threshold, args.save_dir)
    
    # Summarize findings
    predicted_diseases = [r['disease'] for r in results if r['predicted']]
    if 'No Finding' in predicted_diseases:
        print("\nSummary: The model predicts no diseases in this image.")
    elif predicted_diseases:
        print(f"\nSummary: The model predicts the patient has: {', '.join(predicted_diseases)}")
    else:
        print("\nSummary: The model does not predict any diseases above the threshold")

# Main function
def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Improved Chest X-ray Disease Classification')
    subparsers = parser.add_subparsers(dest='mode', help='Mode to run: train or test')
    
    # Training arguments
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data_path', type=str, default='/kaggle/input/data', help='Path to the data directory')
    train_parser.add_argument('--save_dir', type=str, default='improved_outputs', help='Directory to save model outputs')
    train_parser.add_argument('--backbone', type=str, default='densenet121', choices=['densenet121', 'resnet50', 'efficientnet_b3', 'convnext_base'], help='Backbone model architecture')
    train_parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    train_parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    train_parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    train_parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    train_parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    train_parser.add_argument('--image_size', type=int, default=512, help='Image size')
    train_parser.add_argument('--use_metadata', action='store_true', help='Use patient metadata')
    train_parser.add_argument('--balance_classes', action='store_true', help='Balance classes in the dataset')
    train_parser.add_argument('--max_no_finding', type=int, default=None, help='Maximum number of No Finding samples to use')
    train_parser.add_argument('--focal_loss', action='store_true', help='Use focal loss')
    train_parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma parameter for focal loss')
    train_parser.add_argument('--weighted_loss', action='store_true', help='Use weighted loss')
    train_parser.add_argument('--no_finding_weight', type=float, default=2.0, help='Weight for No Finding class')
    train_parser.add_argument('--lr_factor', type=float, default=0.7, help='Learning rate decay factor')
    train_parser.add_argument('--patience', type=int, default=3, help='Patience for learning rate scheduler')
    train_parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    train_parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    train_parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    train_parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    # Testing arguments
    test_parser = subparsers.add_parser('test', help='Test the model')
    test_parser.add_argument('--image', type=str, required=True, help='Path to the X-ray image')
    test_parser.add_argument('--age', type=float, default=50, help='Patient age (years)')
    test_parser.add_argument('--gender', type=str, default='M', choices=['M', 'F'], help='Patient gender (M/F)')
    test_parser.add_argument('--view', type=str, default='PA', help='View position (PA, AP, LATERAL, etc.)')
    test_parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    test_parser.add_argument('--threshold', type=float, default=0.5, help='Prediction threshold (default: 0.5)')
    test_parser.add_argument('--no_finding_threshold', type=float, default=0.6, help='Threshold for No Finding verification')
    test_parser.add_argument('--image_size', type=int, default=512, help='Image size')
    test_parser.add_argument('--ensemble', action='store_true', help='Use model ensemble (provide comma-separated checkpoints)')
    test_parser.add_argument('--save_dir', type=str, default='predictions', help='Directory to save predictions')
    test_parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the appropriate mode
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'test':
        test_model(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 