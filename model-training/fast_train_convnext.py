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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import random
from torch import optim
from torch import nn

# Import integrated model and helper tools
from models.convnext_large_model import IntegratedConvNextModel
from utils.bbox_utils import load_bbox_data, get_image_bboxes
from utils.preprocessing import balance_dataset_weights, balance_dataset_weights_advanced

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, class_specific_gamma=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.class_specific_gamma = class_specific_gamma
        
    def forward(self, inputs, targets):
        # Convert inputs to probabilities using sigmoid
        probs = torch.sigmoid(inputs)
        
        # Calculate focal loss for each class
        if self.class_specific_gamma:
            # Use different gamma for each class based on class frequency
            class_counts = torch.sum(targets, dim=0)
            total_samples = len(targets)
            class_gammas = torch.log(total_samples / (class_counts + 1e-6))
            class_gammas = torch.clamp(class_gammas, min=0.5, max=5.0)
            focal_weights = (1 - probs) ** class_gammas.unsqueeze(0)
        else:
            focal_weights = (1 - probs) ** self.gamma
        
        # Calculate binary cross entropy with focal weights
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        focal_loss = focal_weights * bce_loss
        
        # Return mean loss
        return torch.mean(focal_loss)

# Define disease classes
DISEASE_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
    'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia',
    'No Finding'  # Added No Finding as a class
]

class FastIntegratedDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_list, labels, transforms=None, bbox_data=None, demographic_data=None):
        super().__init__()
        self.data_dir = data_dir
        self.image_list = image_list
        self.labels = labels
        self.transforms = transforms
        self.bbox_data = bbox_data
        self.demographic_data = demographic_data
        self.image_path_cache = {}
        self.demographic_cache = {}
        self.bbox_cache = {}
        
        # Extract image filenames
        self.image_names = []
        for img_path in self.image_list:
            if isinstance(img_path, str):
                self.image_names.append(os.path.basename(img_path))
            else:
                self.image_names.append(img_path)
        
        # Pre-process demographic data
        if self.demographic_data is not None:
            print("Pre-processing demographic data for faster access...")
            for img_name in tqdm(self.image_names, desc="Caching demographics"):
                self._cache_demographics(img_name)
    
    def _cache_demographics(self, img_name):
        """Cache demographic data for faster access"""
        if img_name in self.demographic_cache:
            return
            
        try:
            # Find image in patient data
            patient_data = self.demographic_data[self.demographic_data['Image Index'] == img_name]
            
            if not patient_data.empty:
                # Extract patient age and convert to value between 0 and 1
                try:
                    age_str = patient_data['Patient Age'].iloc[0]
                    if isinstance(age_str, str) and 'Y' in age_str:
                        age = float(age_str.replace('Y', '')) / 100.0
                    else:
                        age = float(age_str) / 100.0
                    age = min(max(age, 0.0), 1.0)
                except (ValueError, TypeError):
                    age = 0.5
                
                # Extract patient gender (male = 1, female = 0)
                gender_value = 1.0 if patient_data['Patient Gender'].iloc[0] == 'M' else 0.0
                
                # Extract view position (PA = 1, other = 0)
                view_position = patient_data['View Position'].iloc[0]
                view_value = 1.0 if view_position == 'PA' else 0.0
                
                # Store processed data
                self.demographic_cache[img_name] = {
                    'age': age,
                    'gender': gender_value,
                    'view': view_value
                }
            else:
                # Use default values if no data found
                self.demographic_cache[img_name] = {
                    'age': 0.5,
                    'gender': 0.5,
                    'view': 0.5
                }
        except Exception as e:
            # Use default values on error
            self.demographic_cache[img_name] = {
                'age': 0.5,
                'gender': 0.5,
                'view': 0.5
            }

    def _cache_bboxes(self, img_name):
        """Cache bounding box data for faster access"""
        if img_name in self.bbox_cache:
            return
            
        try:
            if self.bbox_data is not None:
                # Filter bbox_data for this image
                img_bboxes = get_image_bboxes(img_name, self.bbox_data)
                
                if img_bboxes and len(img_bboxes) > 0:
                    # Use first bbox only
                    bbox = img_bboxes[0]
                    self.bbox_cache[img_name] = [
                        float(bbox['x']), 
                        float(bbox['y']), 
                        float(bbox['x']) + float(bbox['width']), 
                        float(bbox['y']) + float(bbox['height'])
                    ]
                else:
                    self.bbox_cache[img_name] = [0, 0, 1, 1]
            else:
                self.bbox_cache[img_name] = [0, 0, 1, 1]
        except Exception as e:
            self.bbox_cache[img_name] = [0, 0, 1, 1]
        
    def find_image_path(self, image_name):
        """Find image path in subdirectories quickly"""
        if image_name in self.image_path_cache:
            return self.image_path_cache[image_name]
            
        try:
            # Extract folder number from image name (faster)
            img_number = int(image_name.split('_')[0])
            folder_number = (img_number // 1000) % 12 + 1
            
            # Check expected folder first
            expected_folder = f"images_{folder_number:03d}"
            
            # Possible image paths
            paths = [
                os.path.join(self.data_dir, expected_folder, "images", image_name),
                os.path.join(self.data_dir, expected_folder, image_name)
            ]
            
            for path in paths:
                if os.path.exists(path):
                    self.image_path_cache[image_name] = path
                    return path
            
            # Not found in expected folder, search all folders
            for i in range(1, 13):
                folder = f"images_{i:03d}"
                if folder != expected_folder:
                    paths = [
                        os.path.join(self.data_dir, folder, "images", image_name),
                        os.path.join(self.data_dir, folder, image_name)
                    ]
                    
                    for path in paths:
                        if os.path.exists(path):
                            self.image_path_cache[image_name] = path
                            return path
            
            # Image not found
            self.image_path_cache[image_name] = None
            return None
        except:
            self.image_path_cache[image_name] = None
            return None

    def _load_and_transform_image(self, img_path):
        """Load and transform an image"""
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transforms:
                image = self.transforms(image)
            return image
        except:
            return None
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        # Get image name
        img_name = self.image_names[idx]
        
        # Load image
        img_path = self.find_image_path(img_name)
        if img_path:
            image = self._load_and_transform_image(img_path)
        else:
            image = None
        
        if image is None:
            # Return a blank image if loading fails
            image = torch.zeros((3, 384, 384))
        
        # Prepare result
        result = {
            'image': image,
            'label': torch.tensor(self.labels[idx], dtype=torch.float32),
        }
        
        # Get bounding boxes if available
        if self.bbox_data is not None:
            try:
                # Filter bbox_data for this image
                img_bboxes = get_image_bboxes(img_name, self.bbox_data)
                
                if img_bboxes and len(img_bboxes) > 0:
                    # Use first bbox only
                    bbox = img_bboxes[0]
                    result['bbox'] = torch.tensor([
                        float(bbox['x']), 
                        float(bbox['y']), 
                        float(bbox['x']) + float(bbox['width']), 
                        float(bbox['y']) + float(bbox['height'])
                    ], dtype=torch.float32)
                else:
                    # Use default value if no bounding box found
                    result['bbox'] = torch.tensor([0, 0, 1, 1], dtype=torch.float32)
            except Exception as e:
                # Use default value on error
                result['bbox'] = torch.tensor([0, 0, 1, 1], dtype=torch.float32)
        else:
            result['bbox'] = torch.tensor([0, 0, 1, 1], dtype=torch.float32)
        
        # Get demographic information if available
        if self.demographic_data is not None:
            try:
                # Find image in patient data
                patient_data = self.demographic_data[self.demographic_data['Image Index'] == img_name]
                
                if not patient_data.empty:
                    # Extract patient age and convert to value between 0 and 1
                    try:
                        age_str = patient_data['Patient Age'].iloc[0]
                        if isinstance(age_str, str) and 'Y' in age_str:
                            age = float(age_str.replace('Y', '')) / 100.0
                        else:
                            age = float(age_str) / 100.0
                        age = min(max(age, 0.0), 1.0)
                    except (ValueError, TypeError):
                        age = 0.5
                    
                    # Extract patient gender (male = 1, female = 0)
                    gender_value = 1.0 if patient_data['Patient Gender'].iloc[0] == 'M' else 0.0
                    
                    # Extract view position (PA = 1, other = 0)
                    view_position = patient_data['View Position'].iloc[0]
                    view_value = 1.0 if view_position == 'PA' else 0.0
                    
                    # Add demographic data
                    result['age'] = torch.tensor(age, dtype=torch.float32)
                    result['gender'] = torch.tensor(gender_value, dtype=torch.float32)
                    result['view'] = torch.tensor(view_value, dtype=torch.float32)
                else:
                    # Use default values if no data found
                    result['age'] = torch.tensor(0.5, dtype=torch.float32)
                    result['gender'] = torch.tensor(0.5, dtype=torch.float32)
                    result['view'] = torch.tensor(0.5, dtype=torch.float32)
            except Exception as e:
                # Use default values on error
                print(f"Error extracting demographic data for image {img_name}: {str(e)}")
                result['age'] = torch.tensor(0.5, dtype=torch.float32)
                result['gender'] = torch.tensor(0.5, dtype=torch.float32)
                result['view'] = torch.tensor(0.5, dtype=torch.float32)
        else:
            # Use default values if no demographic data available
            result['age'] = torch.tensor(0.5, dtype=torch.float32)
            result['gender'] = torch.tensor(0.5, dtype=torch.float32)
            result['view'] = torch.tensor(0.5, dtype=torch.float32)
        
        return result

def create_fast_data_loaders(args):
    """Create data loaders for fast training"""
    print("Starting data loading (fast mode)...")
    
    # Image transformations
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Read image list
    with open(args.train_val_list, 'r') as f:
        image_list = f.read().splitlines()
        
    # Limit dataset size for debugging
    if args.debug or args.quick_test:
        image_list = image_list[:100]
        
    # Load metadata
    metadata = pd.read_csv(args.csv_path)
    
    # Sample No Finding cases to speed up processing
    if args.sample_no_finding:
        no_finding = metadata[metadata['Finding Labels'] == 'No Finding']
        other_cases = metadata[metadata['Finding Labels'] != 'No Finding']
        sampled_no_finding = no_finding.sample(min(len(no_finding), len(other_cases)))
        metadata = pd.concat([other_cases, sampled_no_finding])
        
    # Filter image list to only include images present in metadata
    image_list = [img for img in image_list if img in metadata['Image Index'].values]
    
    # Load bounding box data if available
    bbox_data = None
    if args.bbox_path:
        try:
            bbox_data = pd.read_csv(args.bbox_path)
            print("Successfully loaded bounding box data")
        except Exception as e:
            print(f"Failed to load bounding box data: {e}")
            print("Continuing without bounding boxes for speed")
            
    # Prepare labels
    labels = []
    for img in image_list:
        findings = metadata[metadata['Image Index'] == img]['Finding Labels'].values[0]
        label = [0] * len(DISEASE_CLASSES)
        for finding in findings.split('|'):
            if finding in DISEASE_CLASSES:
                label[DISEASE_CLASSES.index(finding)] = 1
        labels.append(label)
        
    # Split data
    train_images, val_images, train_labels, val_labels = train_test_split(
        image_list, labels, test_size=0.2, random_state=42,
        stratify=[label[DISEASE_CLASSES.index('No Finding')] for label in labels]
    )
    
    # Create datasets
    train_dataset = FastIntegratedDataset(
        data_dir=args.data_dir,
        image_list=train_images,
        labels=train_labels,
        transforms=train_transform,
        bbox_data=bbox_data if args.use_bbox else None,
        demographic_data=metadata if args.use_metadata else None
    )
    
    val_dataset = FastIntegratedDataset(
        data_dir=args.data_dir,
        image_list=val_images,
        labels=val_labels,
        transforms=val_transform,
        bbox_data=bbox_data if args.use_bbox else None,
        demographic_data=metadata if args.use_metadata else None
    )
    
    # Create data loaders
    if args.use_weighted_sampler:
        # Convert labels to numpy array
        train_labels_array = np.array(train_labels)
        
        # Calculate class weights
        class_counts = np.sum(train_labels_array, axis=0)
        total_samples = len(train_labels_array)
        class_weights = np.log(total_samples / (class_counts + 1e-6))
        class_weights = np.clip(class_weights, 0.5, 5.0)
        
        # Calculate sample weights
        sample_weights = np.zeros(len(train_labels_array))
        for i in range(len(train_labels_array)):
            positive_classes = np.nonzero(train_labels_array[i])[0]
            if len(positive_classes) > 0:
                sample_weights[i] = np.mean(class_weights[positive_classes])
            else:
                sample_weights[i] = 0
        
        # Normalize sample weights
        if np.sum(sample_weights) > 0:
            sample_weights = sample_weights / np.sum(sample_weights) * len(sample_weights)
        
        # Create sampler
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).float(),
            num_samples=len(train_labels_array),
            replacement=True
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            sampler=sampler, num_workers=args.num_workers,
            pin_memory=True, persistent_workers=True
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers,
            pin_memory=True, persistent_workers=True
        )
        
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=True
    )
    
    return train_loader, val_loader

def parse_args():
    parser = argparse.ArgumentParser(description='Fast training of ConvNext model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing the data')
    parser.add_argument('--csv_path', type=str, default='data/Data_Entry_2017.csv',
                        help='Path to the CSV file containing metadata')
    parser.add_argument('--bbox_path', type=str, default=None,
                        help='Path to the CSV file containing bounding box data')
    parser.add_argument('--train_val_list', type=str, default='data/train_val_list.txt',
                        help='Path to the file containing train/val split')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save outputs')
    
    # Model parameters
    parser.add_argument('--num_classes', type=int, default=14,
                        help='Number of classes to predict')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Size of input images')
    parser.add_argument('--use_metadata', action='store_true',
                        help='Whether to use metadata in the model')
    parser.add_argument('--use_bbox', action='store_true',
                        help='Whether to use bounding box data in the model')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Loss function parameters
    parser.add_argument('--use_focal_loss', action='store_true',
                        help='Whether to use focal loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma parameter for focal loss')
    parser.add_argument('--class_specific_gamma', action='store_true',
                        help='Whether to use class-specific gamma for focal loss')
    
    # Data sampling parameters
    parser.add_argument('--use_weighted_sampler', action='store_true',
                        help='Whether to use weighted sampler')
    parser.add_argument('--sample_no_finding', action='store_true',
                        help='Whether to sample No Finding cases')
    
    # Debug parameters
    parser.add_argument('--debug', action='store_true',
                        help='Whether to run in debug mode')
    parser.add_argument('--quick_test', action='store_true',
                        help='Whether to run a quick test')
    
    # Additional training parameters
    parser.add_argument('--use_scheduler', action='store_true',
                        help='Whether to use learning rate scheduler')
    parser.add_argument('--patience', type=int, default=7,
                        help='Patience for learning rate scheduler')
    parser.add_argument('--fp16', action='store_true',
                        help='Whether to use mixed precision training')
    parser.add_argument('--model_variant', type=str, default='large',
                        help='Model variant to use')
    parser.add_argument('--input_size', type=int, default=384,
                        help='Input size for the model')
    parser.add_argument('--boost_low_auroc', action='store_true',
                        help='Whether to boost low AUROC classes')
    parser.add_argument('--use_augmentation', action='store_true',
                        help='Whether to use data augmentation')
    parser.add_argument('--augmentation_strength', type=float, default=1.5,
                        help='Strength of data augmentation')
    parser.add_argument('--warmup_epochs', type=int, default=3,
                        help='Number of warmup epochs')
    parser.add_argument('--cosine_annealing', action='store_true',
                        help='Whether to use cosine annealing')
    parser.add_argument('--class_weights', type=str, default='auto',
                        help='Class weights strategy')
    parser.add_argument('--accumulation_steps', type=int, default=2,
                        help='Gradient accumulation steps')
    parser.add_argument('--freeze_ratio', type=float, default=0.4,
                        help='Ratio of layers to freeze')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0,
                        help='Gradient clipping norm')
    parser.add_argument('--mixup_alpha', type=float, default=0.4,
                        help='Mixup alpha parameter')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing parameter')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Create data loaders
    train_loader, val_loader = create_fast_data_loaders(args)
    
    # Create model
    model = IntegratedConvNextModel(
        num_classes=args.num_classes,
        use_metadata=args.use_metadata,
        use_bbox=args.use_bbox
    ).to(device)
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=args.patience, verbose=True
        )
    else:
        scheduler = None
    
    # Setup loss function
    if args.use_focal_loss:
        criterion = FocalLoss(
            gamma=args.focal_gamma,
            class_specific_gamma=args.class_specific_gamma
        )
    else:
        criterion = nn.BCEWithLogitsLoss()
        
    # Training loop
    best_auroc = 0.0
    scaler = GradScaler() if args.fp16 else None
    
    for epoch in range(args.num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")):
            # Get data
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            bbox = batch.get('bbox', None)
            if bbox is not None:
                bbox = bbox.to(device)
            
            # Get metadata if available
            metadata = None
            if args.use_metadata:
                metadata = {
                    'age': batch['age'].to(device),
                    'gender': batch['gender'].to(device),
                    'view': batch['view'].to(device)
                }
            
            # Forward pass
            if args.fp16:
                with autocast(device_type='cuda'):
                    outputs = model(images, metadata, bbox)
                    loss = criterion(outputs, labels) / args.accumulation_steps
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % args.accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(images, metadata, bbox)
                loss = criterion(outputs, labels) / args.accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % args.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Update metrics
            train_loss += loss.item() * args.accumulation_steps
            train_preds.append(outputs.detach().cpu())
            train_labels.append(labels.detach().cpu())
            
        # Calculate training metrics
        train_loss /= len(train_loader)
        train_preds = torch.cat(train_preds)
        train_labels = torch.cat(train_labels)
        train_auroc = roc_auc_score(train_labels.numpy(), train_preds.numpy())
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Get data
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                bbox = batch.get('bbox', None)
                if bbox is not None:
                    bbox = bbox.to(device)
                
                # Get metadata if available
                metadata = None
                if args.use_metadata:
                    metadata = {
                        'age': batch['age'].to(device),
                        'gender': batch['gender'].to(device),
                        'view': batch['view'].to(device)
                    }
                
                # Forward pass
                outputs = model(images, metadata, bbox)
                loss = criterion(outputs, labels)
                
                # Update metrics
                val_loss += loss.item()
                val_preds.append(outputs.cpu())
                val_labels.append(labels.cpu())
                
        # Calculate validation metrics
        val_loss /= len(val_loader)
        val_preds = torch.cat(val_preds)
        val_labels = torch.cat(val_labels)
        val_auroc = roc_auc_score(val_labels.numpy(), val_preds.numpy())
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step(val_auroc)
        
        # Print metrics
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train AUROC: {train_auroc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val AUROC: {val_auroc:.4f}")
        
        # Save best model
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auroc': val_auroc,
                'args': args
            }, os.path.join(args.output_dir, 'best_model.pth'))
            
        # Early stopping
        if scheduler is not None and epoch - scheduler.last_epoch > 5:
            print("Early stopping triggered")
            break
            
    print(f"\nTraining completed. Best validation AUROC: {best_auroc:.4f}")

if __name__ == "__main__":
    # Fix for Windows
    import platform
    if platform.system() == 'Windows':
        import multiprocessing
        multiprocessing.freeze_support()
    
    main()