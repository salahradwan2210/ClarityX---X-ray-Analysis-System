import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Classes that have bounding box annotations
CLASSES_WITH_BBOX = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
                    'Nodule', 'Pneumonia', 'Pneumothorax']

class CFG:
    IMG_SIZE = 320
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5
    BATCH_SIZE = 4

class BBoxModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Use ResNet50 as backbone
        resnet = models.resnet50(weights=None)  # Initialize without pretrained weights
        
        # Remove the last layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Feature dimension for ResNet50
        in_features = 2048
        
        # Simplified spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_features, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Simplified localization head
        self.bbox_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, len(CLASSES_WITH_BBOX) * 4)
        )
        
        # Simplified confidence head - removed sigmoid activation
        self.confidence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, len(CLASSES_WITH_BBOX))  # Raw logits output
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Get backbone features
        features = self.backbone(x)
        
        # Apply spatial attention
        attention = self.spatial_attention(features)
        features_attended = features * attention
        
        # Get bounding boxes and confidence scores
        bbox_pred = self.bbox_head(features_attended)
        confidence = self.confidence_head(features_attended)  # Raw logits
        
        # Reshape bbox predictions
        bbox_pred = bbox_pred.view(-1, len(CLASSES_WITH_BBOX), 4)
        confidence = confidence.view(-1, len(CLASSES_WITH_BBOX))  # Keep as logits
        
        return bbox_pred, confidence, attention

def get_transforms(img_size):
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    
    return intersection / (box1_area + box2_area - intersection)

def train_step(model, batch, criterion, optimizer, device):
    images, targets = batch
    images = images.to(device)
    target_boxes = targets['boxes'].to(device)
    target_labels = targets['labels'].to(device)
    
    # Forward pass
    bbox_pred, confidence, _ = model(images)
    
    # Calculate losses
    bbox_loss = F.smooth_l1_loss(bbox_pred, target_boxes)
    conf_loss = F.binary_cross_entropy(confidence, target_labels)
    
    # Total loss
    loss = bbox_loss + conf_loss
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return {'bbox_loss': bbox_loss.item(), 'conf_loss': conf_loss.item(), 'total_loss': loss.item()}

def predict_boxes(model, image_path, device=CFG.DEVICE, confidence_threshold=CFG.CONFIDENCE_THRESHOLD):
    """Predict bounding boxes for a single image"""
    transform = get_transforms(CFG.IMG_SIZE)
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        bbox_pred, confidence, attention = model(image_tensor)
    
    # Convert predictions to numpy
    bboxes = bbox_pred.cpu().numpy()[0]  # Shape: [num_classes, 4]
    scores = confidence.cpu().numpy()[0]  # Shape: [num_classes]
    
    # Filter predictions by confidence
    predictions = []
    for i, (bbox, score) in enumerate(zip(bboxes, scores)):
        if score >= confidence_threshold:
            predictions.append({
                'class': CLASSES_WITH_BBOX[i],
                'confidence': float(score),
                'bbox': bbox.tolist()
            })
    
    return predictions, attention.cpu().numpy()[0]

def visualize_predictions(image_path, predictions, attention_map, output_path):
    """Visualize bbox predictions with attention map"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Plot original image with boxes
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    for pred in predictions:
        bbox = pred['bbox']
        x1, y1 = int(bbox[0] * width), int(bbox[1] * height)
        x2, y2 = int((bbox[0] + bbox[2]) * width), int((bbox[1] + bbox[3]) * height)
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                        fill=False, color='red', linewidth=2))
        plt.text(x1, y1-10, f"{pred['class']}: {pred['confidence']:.2f}", 
                color='red', fontsize=8)
    plt.title("Predictions")
    
    # Plot attention map
    plt.subplot(1, 2, 2)
    attention_resized = cv2.resize(attention_map, (width, height))
    plt.imshow(attention_resized, cmap='jet', alpha=0.5)
    plt.title("Attention Map")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close() 