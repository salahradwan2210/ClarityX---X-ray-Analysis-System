import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import timm
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
import matplotlib.cm as cm

# التكوين
class CFG:
    SEED = 42
    MODEL_NAME = 'convnext_large'
    IMG_SIZE = 512  # زيادة حجم الصورة لالتقاط المزيد من التفاصيل
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CHECKPOINT_PATH = 'best_model_epoch_9_auroc_0.9692.pth'
    DROPOUT_HEAD = 0.2
    DROPOUT_META = 0.1
    THRESHOLD = 0.45  # العتبة الافتراضية للتشخيص
    SHOW_TOP_N = 3  # عدد النتائج الأعلى للعرض
    USE_ATTENTION_MAPS = True  # استخدام خرائط الانتباه للتحليل البصري

# قائمة الأمراض
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding']
TRAIN_CLASSES = [c for c in CLASS_NAMES if c != 'No Finding']

# النموذج المحسن
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
                    with torch.no_grad():
                        dummy_input = torch.randn(1, 3, 32, 32)
                        dummy_output = self.model(dummy_input)
                        in_features = dummy_output.shape[-1]
            except Exception:
                in_features = 1536  # الافتراضي لنموذج convnext_large
                
        # تحسين وحدة الانتباه
        self.attention = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, in_features // 8),  # زيادة حجم الطبقة الوسطى
            nn.GELU(),
            nn.Dropout(0.1),  # إضافة dropout للتعميم
            nn.Linear(in_features // 8, in_features),
            nn.Sigmoid()
        )
        
        # تحسين رأس التحديد المكاني
        self.localization_head = nn.Sequential(
            nn.LayerNorm(in_features), 
            nn.Dropout(0.1),
            nn.Linear(in_features, in_features // 2),
            nn.GELU(),
            nn.Linear(in_features // 2, self.train_classes * 4)
        )
        
        # تحسين فرع البيانات الوصفية
        self.metadata_branch = nn.Sequential(
            nn.Linear(metadata_features, 128),  # زيادة حجم الطبقة
            nn.LayerNorm(128), 
            nn.ReLU(), 
            nn.Dropout(CFG.DROPOUT_META), 
            nn.Linear(128, 256),  # زيادة حجم الطبقة
            nn.LayerNorm(256), 
            nn.ReLU()
        )
        
        # تحسين الطبقة المشتركة النهائية
        self.combined_fc = nn.Sequential(
            nn.LayerNorm(in_features + 256),  # زيادة حجم الميزات المدمجة
            nn.Dropout(CFG.DROPOUT_HEAD), 
            nn.Linear(in_features + 256, in_features // 2),
            nn.GELU(),
            nn.Dropout(CFG.DROPOUT_HEAD / 2),
            nn.Linear(in_features // 2, self.train_classes)
        )
    
    def forward(self, x_img, x_meta):
        img_features = self.model(x_img)
        attention_weights = self.attention(img_features)
        img_features = img_features * attention_weights
        
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
        
        return combined_cls_out, bbox_out, attention_weights

# وظائف المساعدة
def get_transforms(img_size):
    try:
        return A.Compose([
            A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_AREA),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    except:
        print("Warning: Albumentations not installed. Falling back to basic Torchvision transforms.")
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            normalize,
        ])

def load_model(checkpoint_path, device, metadata_features=3):
    model = AdvancedChestModel(CFG.MODEL_NAME, num_classes=len(CLASS_NAMES), metadata_features=metadata_features, pretrained=False)
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            if next(iter(state_dict)).startswith('module.'):
                state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            print(f"Model loaded successfully from '{checkpoint_path}'.")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Warning: Checkpoint '{checkpoint_path}' not found. Using untrained model.")
    model.to(device)
    model.eval()
    return model

def predict_image(image_path, model, device, threshold=CFG.THRESHOLD, debug=False):
    # قراءة الصورة وتطبيق التحويلات
    try:
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
        transform = get_transforms(CFG.IMG_SIZE)
        
        # استخدام AlbumentationsTransform
        try:
            transformed = transform(image=img_np)
            img_tensor = transformed['image']
        except:
            img_tensor = transform(img_np)
        
        # تجهيز البيانات الوصفية الافتراضية (تستخدم عندما لا تتوفر بيانات وصفية حقيقية)
        # السن، الجنس، وضعية الصورة
        metadata = torch.tensor([0.6, 0.5, 0.5], dtype=torch.float32)  # قيم افتراضية
        
        # التنبؤ
        with torch.no_grad():
            img_tensor = img_tensor.unsqueeze(0).to(device)
            metadata = metadata.unsqueeze(0).to(device)
            outputs, bbox_outputs, attention_map = model(img_tensor, metadata)
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
        
        # الحصول على النتائج
        disease_results = []
        for i, cls_name in enumerate(TRAIN_CLASSES):
            disease_results.append({'disease': cls_name, 'probability': float(probabilities[i])})
        
        # ترتيب النتائج حسب الاحتمالية
        disease_results = sorted(disease_results, key=lambda x: x['probability'], reverse=True)
        
        # تحديد ما إذا كانت الحالة طبيعية
        max_prob = max(probabilities)
        is_normal = max_prob < threshold
        
        # إرجاع النتائج
        results = {
            'predictions': disease_results,
            'is_normal': is_normal,
            'max_probability': float(max_prob),
            'attention_map': attention_map.cpu().numpy()[0] if CFG.USE_ATTENTION_MAPS else None
        }
        
        if debug:
            print(f"Max Probability: {max_prob:.4f}")
            print(f"Is Normal: {is_normal}")
            for item in disease_results[:5]:
                print(f"{item['disease']}: {item['probability']:.4f}")
                
        return results, img_np
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None

def generate_visualization(image, attention_map, predictions, threshold=CFG.THRESHOLD, top_n=CFG.SHOW_TOP_N):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # الصورة الأصلية
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # إضافة خريطة الانتباه
    if attention_map is not None:
        # تغيير حجم خريطة الانتباه لتطابق الصورة
        heatmap = cv2.resize(attention_map, (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # تحويل تنسيق الألوان
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # دمج الصورة مع خريطة الانتباه
        superimposed_img = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        axes[1].imshow(superimposed_img)
        axes[1].set_title('Attention Map')
        axes[1].axis('off')
    else:
        # نسخة من الصورة الأصلية إذا لم تكن خريطة الانتباه متاحة
        axes[1].imshow(image)
        axes[1].set_title('Processing')
        axes[1].axis('off')
    
    # إضافة تصنيفات التشخيص
    is_normal = predictions['is_normal']
    disease_preds = predictions['predictions']
    
    plt.figtext(0.5, 0.01, 'DIAGNOSIS RESULTS', fontsize=16, ha='center', weight='bold')
    
    if is_normal:
        plt.figtext(0.5, 0.05, 'NO FINDINGS DETECTED', fontsize=14, ha='center', color='green')
    else:
        plt.figtext(0.5, 0.05, 'POTENTIAL FINDINGS DETECTED:', fontsize=14, ha='center', color='red')
        # عرض أعلى N نتائج
        for i, pred in enumerate(disease_preds[:top_n]):
            if pred['probability'] > threshold:
                color = 'red'
            else:
                color = 'grey'
            
            plt.figtext(0.5, 0.1 + i * 0.03, 
                      f"{pred['disease']}: {pred['probability']:.2f}",
                      fontsize=12, ha='center', color=color)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    return fig

def main():
    parser = argparse.ArgumentParser(description='Chest X-ray Disease Diagnosis')
    parser.add_argument('image_path', type=str, help='Path to the X-ray image')
    parser.add_argument('--threshold', type=float, default=CFG.THRESHOLD, help='Threshold for positive predictions')
    parser.add_argument('--checkpoint', type=str, default=CFG.CHECKPOINT_PATH, help='Path to model checkpoint')
    parser.add_argument('--save', type=str, default=None, help='Save visualization to specified file path')
    parser.add_argument('--debug', action='store_true', help='Print debug information')
    
    args = parser.parse_args()
    
    # التحقق من وجود الصورة
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        return
    
    # تحميل النموذج
    model = load_model(args.checkpoint, CFG.DEVICE)
    
    # التنبؤ
    results, img = predict_image(args.image_path, model, CFG.DEVICE, args.threshold, args.debug)
    if results is None:
        print("Failed to process the image.")
        return
    
    # توليد الرسم البياني
    fig = generate_visualization(img, results['attention_map'], results)
    
    # حفظ أو عرض النتيجة
    if args.save:
        plt.savefig(args.save, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {args.save}")
    else:
        plt.show()
    
    # طباعة النتائج
    print("\n=== DIAGNOSIS RESULTS ===")
    if results['is_normal']:
        print("NO FINDINGS DETECTED")
    else:
        print("POTENTIAL FINDINGS DETECTED:")
        for i, pred in enumerate(results['predictions'][:CFG.SHOW_TOP_N]):
            if pred['probability'] > args.threshold:
                print(f"{pred['disease']}: {pred['probability']:.4f} (POSITIVE)")
            else:
                print(f"{pred['disease']}: {pred['probability']:.4f}")

if __name__ == "__main__":
    main() 