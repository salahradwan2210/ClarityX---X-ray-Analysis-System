import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from PIL import Image
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, classification_report
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- Configuration ---
class CFG:
    SEED = 42
    MODEL_NAME = 'convnext_large'
    IMG_SIZE = 512
    BATCH_SIZE = 4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BASE_PATH = 'data'
    IMAGE_DIR = BASE_PATH
    DATA_ENTRY_PATH = os.path.join(BASE_PATH, 'Data_Entry_2017.csv')
    BBOX_PATH = os.path.join(BASE_PATH, 'BBox_List_2017.csv')
    MODEL_PATH = 'best_model_epoch_3_auroc_9004.pth'
    OUTPUT_DIR = 'predictions'
    THRESHOLD = 0.5  # Classification threshold

# Create output directory if it doesn't exist
os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

# --- Class Definitions ---
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding']
TRAIN_CLASSES = [c for c in CLASS_NAMES if c != 'No Finding']
CLASSES_WITH_BBOX = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']

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

# --- Model ---
class AdvancedChestModel(nn.Module):
    def __init__(self, model_name, num_classes, metadata_features=3, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.train_classes = num_classes - 1
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, features_only=False)
        
        try:
            if hasattr(self.model, 'num_features'):
                in_features = self.model.num_features
            elif hasattr(self.model.head, 'in_features'):
                in_features = self.model.head.in_features
            else:
                raise AttributeError("Cannot find standard feature attribute")
        except AttributeError:
            try:
                if hasattr(self.model, 'fc') and hasattr(self.model.fc, 'in_features'):
                    in_features = self.model.fc.in_features
                elif hasattr(self.model, 'classifier') and hasattr(self.model.classifier, 'in_features'):
                    in_features = self.model.classifier.in_features
                else:
                    in_features = 1536  # Default for convnext_large
            except Exception:
                in_features = 1536
                
        self.localization_head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, self.train_classes * 4)
        )
        
        self.metadata_branch = nn.Sequential(
            nn.Linear(metadata_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        self.combined_fc = nn.Sequential(
            nn.LayerNorm(in_features + 128),
            nn.Dropout(0.6),
            nn.Linear(in_features + 128, self.train_classes)
        )

    def forward(self, x_img, x_meta):
        img_features = self.model(x_img)
        bbox_out = self.localization_head(img_features)
        bbox_out = bbox_out.view(bbox_out.size(0), self.train_classes, 4)
        
        if x_meta.shape[1] != self.metadata_branch[0].in_features:
            if x_meta.shape[1] < self.metadata_branch[0].in_features:
                padding = torch.zeros(x_meta.shape[0], self.metadata_branch[0].in_features - x_meta.shape[1], device=x_meta.device)
                x_meta = torch.cat([x_meta, padding], dim=1)
            elif x_meta.shape[1] > self.metadata_branch[0].in_features:
                x_meta = x_meta[:, :self.metadata_branch[0].in_features]
                
        meta_features = self.metadata_branch(x_meta)
        combined_features = torch.cat([img_features, meta_features], dim=1)
        combined_cls_out = self.combined_fc(combined_features)
        
        return combined_cls_out, bbox_out

# --- Dataset for prediction ---
class ChestXrayPredictionDataset(Dataset):
    def __init__(self, image_dir, df, bbox_dict, transform=None):
        self.image_dir = image_dir
        self.df = df.reset_index(drop=True)
        self.bbox_dict = bbox_dict
        self.transform = transform
        self.labels = []
        
        metadata_cols = [col for col in ['Patient Age', 'Patient Gender', 'View Position'] if col in self.df.columns]
        self.metadata_list = []
        
        for idx, row in self.df.iterrows():
            finding = row['Finding Labels']
            img_name = row['Image Index']
            label = np.zeros(len(TRAIN_CLASSES), dtype=np.float32)
            
            for cls in finding.split('|'):
                if cls in TRAIN_CLASSES:
                    cls_idx = TRAIN_CLASSES.index(cls)
                    label[cls_idx] = 1
                    
            self.labels.append(label)
            self.metadata_list.append(row[metadata_cols].values.astype(np.float32))
            
        self.labels = np.array(self.labels)
        self.metadata = np.array(self.metadata_list)
        self.metadata_dim = self.metadata.shape[1] if self.metadata.size > 0 else 0
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['Image Index']
        img_path = find_image_path(img_name, self.image_dir)
        
        if img_path is None:
            return self._get_dummy_item(), img_name
            
        try:
            image = np.array(Image.open(img_path).convert('RGB'))
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            return self._get_dummy_item(), img_name
            
        if self.transform:
            try:
                augmented = self.transform(image=image)
                image = augmented['image']
            except Exception as e:
                print(f"Error applying transforms to {img_name}: {e}")
                return self._get_dummy_item(), img_name
                
        label = torch.from_numpy(self.labels[idx])
        metadata = torch.from_numpy(self.metadata[idx])
        
        return (image, label, metadata), img_name
        
    def _get_dummy_item(self):
        return torch.zeros((3, CFG.IMG_SIZE, CFG.IMG_SIZE)), torch.zeros(len(TRAIN_CLASSES)), torch.zeros(self.metadata_dim)

# --- Transforms for prediction ---
def get_prediction_transforms(img_size):
    return A.Compose([
        A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_LANCZOS4),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# --- Plotting Functions ---
def plot_prediction_results(df_results, save_path):
    plt.figure(figsize=(15, 10))
    
    # Plot average prediction probability per class
    plt.subplot(2, 1, 1)
    avg_probs = df_results[TRAIN_CLASSES].mean().sort_values(ascending=False)
    avg_probs.plot(kind='bar')
    plt.title('Average Prediction Probability per Class')
    plt.ylabel('Probability')
    plt.xticks(rotation=45, ha='right')
    
    # Plot prediction counts (predictions above threshold)
    plt.subplot(2, 1, 2)
    pred_counts = df_results[TRAIN_CLASSES].apply(lambda x: x >= CFG.THRESHOLD).sum().sort_values(ascending=False)
    pred_counts.plot(kind='bar')
    plt.title(f'Number of Predictions per Class (Threshold = {CFG.THRESHOLD})')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_sample_predictions(df_results, image_dir, num_samples=5, save_dir=None):
    if save_dir is None:
        save_dir = os.path.join(CFG.OUTPUT_DIR, 'visualizations')
    os.makedirs(save_dir, exist_ok=True)
    
    # Select random samples
    samples = df_results.sample(min(num_samples, len(df_results)))
    
    for i, (_, row) in enumerate(samples.iterrows()):
        img_name = row['Image Index']
        img_path = find_image_path(img_name, image_dir)
        
        if img_path is None:
            continue
            
        try:
            img = Image.open(img_path).convert('RGB')
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.title(f"Image: {img_name}")
            
            # Add prediction information
            predictions = []
            for cls in TRAIN_CLASSES:
                if row[cls] >= CFG.THRESHOLD:
                    predictions.append(f"{cls}: {row[cls]:.3f}")
                    
            if not predictions:
                predictions.append("No classes above threshold")
                
            plt.xlabel('\n'.join(predictions))
            plt.xticks([])
            plt.yticks([])
            
            plt.savefig(os.path.join(save_dir, f"pred_vis_{i}_{img_name}"))
            plt.close()
        except Exception as e:
            print(f"Error visualizing {img_name}: {e}")

def main():
    print(f"--- Starting Prediction Run ---")
    print(f"Device: {CFG.DEVICE}")
    print(f"Model: {CFG.MODEL_NAME}")
    print(f"Model Path: {CFG.MODEL_PATH}")
    
    # Load data
    print("Loading data entry...")
    df_main = pd.read_csv(CFG.DATA_ENTRY_PATH)
    df = df_main.copy()
    print(f"Dataset size: {len(df)}")
    
    print("Preprocessing metadata...")
    df_processed, _, _ = preprocess_metadata(df)
    print("Metadata OK.")
    
    print("Loading bounding boxes...")
    bbox_dict = load_bounding_boxes(CFG.BBOX_PATH)
    print(f"Loaded {len(bbox_dict)} images with bboxes.")
    
    # Prepare transforms and dataset
    predict_transform = get_prediction_transforms(CFG.IMG_SIZE)
    predict_dataset = ChestXrayPredictionDataset(CFG.IMAGE_DIR, df_processed, bbox_dict, transform=predict_transform)
    predict_loader = DataLoader(predict_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    # Initialize model
    print("Initializing model...")
    model = AdvancedChestModel(CFG.MODEL_NAME, num_classes=len(CLASS_NAMES), metadata_features=predict_dataset.metadata_dim, pretrained=False)
    
    # Load model weights
    print(f"Loading model weights from {CFG.MODEL_PATH}...")
    try:
        state_dict = torch.load(CFG.MODEL_PATH, map_location=CFG.DEVICE)
        # Handle case if state_dict is wrapped in a checkpoint dictionary
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        # Remove 'module.' prefix if present (from DataParallel)
        if next(iter(state_dict.keys())).startswith('module.'):
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model.to(CFG.DEVICE)
    model.eval()
    
    # Predict
    print("Starting prediction...")
    all_preds = []
    all_img_names = []
    
    with torch.no_grad():
        for batch, img_names in tqdm(predict_loader, desc="Predicting"):
            inputs, labels, metadata = batch
            inputs = inputs.to(CFG.DEVICE)
            metadata = metadata.to(CFG.DEVICE)
            
            with torch.amp.autocast(device_type=str(CFG.DEVICE), dtype=torch.float16):
                outputs, _ = model(inputs, metadata)
                preds = outputs.sigmoid().cpu().numpy()
            
            all_preds.append(preds)
            all_img_names.extend(img_names)
    
    # Combine results
    all_preds = np.vstack(all_preds)
    
    # Create results dataframe
    results_dict = {'Image Index': all_img_names}
    for i, cls in enumerate(TRAIN_CLASSES):
        results_dict[cls] = all_preds[:, i]
    
    df_results = pd.DataFrame(results_dict)
    
    # Save results
    results_path = os.path.join(CFG.OUTPUT_DIR, 'prediction_results.csv')
    df_results.to_csv(results_path, index=False)
    print(f"Predictions saved to {results_path}")
    
    # Generate visualization
    plot_path = os.path.join(CFG.OUTPUT_DIR, 'prediction_summary.png')
    plot_prediction_results(df_results, plot_path)
    print(f"Prediction summary plot saved to {plot_path}")
    
    # Visualize sample predictions
    print("Generating sample visualizations...")
    visualize_sample_predictions(df_results, CFG.IMAGE_DIR, num_samples=10)
    print("Visualization complete.")
    
    # Print summary statistics
    print("\n--- Prediction Summary ---")
    print(f"Total images processed: {len(df_results)}")
    print("Average prediction probability per class:")
    for cls, avg_prob in df_results[TRAIN_CLASSES].mean().sort_values(ascending=False).items():
        print(f"  {cls}: {avg_prob:.4f}")
    
    print("\nNumber of detections per class (above threshold):")
    for cls, count in df_results[TRAIN_CLASSES].apply(lambda x: x >= CFG.THRESHOLD).sum().sort_values(ascending=False).items():
        print(f"  {cls}: {count} ({count/len(df_results)*100:.2f}%)")
    
    print("\nPrediction completed successfully!")

if __name__ == '__main__':
    main() 