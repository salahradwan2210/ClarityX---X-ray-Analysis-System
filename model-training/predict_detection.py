import os
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

# استيراد النموذج
from models.detection_model import create_detection_model

# تعريف الفئات
CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
    'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

def parse_args():
    """تحليل معلمات سطر الأوامر"""
    parser = argparse.ArgumentParser(description='Predict using a trained detection model')
    parser.add_argument('--image', type=str, required=True, help='Path to the image or directory of images')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'densenet121', 'convnext'], help='Backbone model')
    parser.add_argument('--output_dir', type=str, default='./detection_predictions', help='Directory to save outputs')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--batch', action='store_true', help='Process a directory of images')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--show', action='store_true', help='Show the predictions')
    return parser.parse_args()

def load_model(model_path, backbone, device):
    """
    تحميل النموذج المدرب
    
    المعلمات:
        model_path (str): مسار النموذج المدرب
        backbone (str): اسم العمود الفقري
        device (torch.device): الجهاز
        
    الإرجاع:
        nn.Module: النموذج المدرب
    """
    # إنشاء النموذج
    model = create_detection_model(
        num_classes=len(CLASS_NAMES),
        backbone_name=backbone,
        pretrained=False
    )
    
    # تحميل الأوزان
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

def preprocess_image(image_path):
    """
    معالجة الصورة للتنبؤ
    
    المعلمات:
        image_path (str): مسار الصورة
        
    الإرجاع:
        torch.Tensor: الصورة المعالجة
    """
    # تحميل الصورة
    image = Image.open(image_path).convert('RGB')
    
    # تطبيق التحويلات
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # تحويل الصورة
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor, image

def visualize_predictions(image, predictions, threshold=0.5, output_path=None, show=False):
    """
    تصور التنبؤات
    
    المعلمات:
        image (PIL.Image): الصورة الأصلية
        predictions (dict): التنبؤات
        threshold (float): عتبة الثقة
        output_path (str): مسار حفظ الصورة
        show (bool): عرض الصورة
    """
    # تحويل الصورة إلى مصفوفة NumPy
    img_np = np.array(image)
    
    # تحويل إلى BGR لـ OpenCV
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # الألوان لكل فئة
    colors = [
        (255, 0, 0),      # أحمر
        (0, 255, 0),      # أخضر
        (0, 0, 255),      # أزرق
        (255, 255, 0),    # أصفر
        (255, 0, 255),    # وردي
        (0, 255, 255),    # سماوي
        (128, 0, 0),      # بني محمر
        (0, 128, 0),      # أخضر داكن
        (0, 0, 128),      # أزرق داكن
        (128, 128, 0),    # زيتوني
        (128, 0, 128),    # أرجواني
        (0, 128, 128),    # فيروزي
        (128, 128, 128),  # رمادي
        (255, 128, 0)     # برتقالي
    ]
    
    # رسم مربعات الإحاطة المتنبأ بها
    detection_results = predictions['detection'][0]
    for box, label, score in zip(
        detection_results['boxes'].cpu().numpy(),
        detection_results['labels'].cpu().numpy(),
        detection_results['scores'].cpu().numpy()
    ):
        if score > threshold:
            x1, y1, x2, y2 = map(int, box)
            class_id = label - 1  # -1 لأننا أضفنا 1 سابقاً
            
            if class_id < len(CLASS_NAMES):
                class_name = CLASS_NAMES[class_id]
                color = colors[class_id % len(colors)]
                
                # رسم المربع
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
                
                # إضافة التسمية
                cv2.putText(img_bgr, f"{class_name}: {score:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # رسم التنبؤات العامة
    cls_preds = predictions['classification']['probabilities'][0]
    y_pos = 30
    for i, (cls_name, prob) in enumerate(zip(CLASS_NAMES, cls_preds)):
        if prob > threshold:
            color = colors[i % len(colors)]
            cv2.putText(img_bgr, f"{cls_name}: {prob:.2f}", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_pos += 25
    
    # حفظ الصورة
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img_bgr)
        print(f"Saved prediction to {output_path}")
    
    # عرض الصورة
    if show:
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('Predictions')
        plt.show()
    
    return img_bgr

def process_single_image(image_path, model, device, output_dir, threshold=0.5, show=False):
    """
    معالجة صورة واحدة
    
    المعلمات:
        image_path (str): مسار الصورة
        model (nn.Module): النموذج
        device (torch.device): الجهاز
        output_dir (str): مجلد الإخراج
        threshold (float): عتبة الثقة
        show (bool): عرض الصورة
    """
    # معالجة الصورة
    image_tensor, original_image = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # التنبؤ
    with torch.no_grad():
        predictions = model.predict(image_tensor, threshold=threshold)
    
    # تصور التنبؤات
    output_path = os.path.join(output_dir, f"pred_{os.path.basename(image_path)}")
    visualize_predictions(original_image, predictions, threshold, output_path, show)
    
    # طباعة التنبؤات
    print("\nClassification Results:")
    print("-" * 40)
    for i, (cls_name, prob) in enumerate(zip(CLASS_NAMES, predictions['classification']['probabilities'][0])):
        if prob > threshold:
            print(f"{cls_name:20s}: {prob:.4f} - POSITIVE")
    
    print("\nDetection Results:")
    print("-" * 40)
    detection_results = predictions['detection'][0]
    for box, label, score in zip(
        detection_results['boxes'].cpu().numpy(),
        detection_results['labels'].cpu().numpy(),
        detection_results['scores'].cpu().numpy()
    ):
        if score > threshold:
            class_id = label - 1
            if class_id < len(CLASS_NAMES):
                class_name = CLASS_NAMES[class_id]
                x1, y1, x2, y2 = map(int, box)
                print(f"{class_name:20s}: {score:.4f} - Box: [{x1}, {y1}, {x2}, {y2}]")

def process_directory(directory_path, model, device, output_dir, threshold=0.5, show=False):
    """
    معالجة مجلد من الصور
    
    المعلمات:
        directory_path (str): مسار المجلد
        model (nn.Module): النموذج
        device (torch.device): الجهاز
        output_dir (str): مجلد الإخراج
        threshold (float): عتبة الثقة
        show (bool): عرض الصور
    """
    # الحصول على قائمة ملفات الصور
    image_extensions = ['.png', '.jpg', '.jpeg']
    image_files = []
    
    for root, _, files in os.walk(directory_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        print(f"No image files found in {directory_path}")
        return
    
    print(f"Found {len(image_files)} image files")
    
    # معالجة كل صورة
    for image_path in image_files:
        print(f"\nProcessing {image_path}")
        process_single_image(image_path, model, device, output_dir, threshold, show)

def main():
    """الدالة الرئيسية"""
    # تحليل المعلمات
    args = parse_args()
    
    # إنشاء مجلد الإخراج
    os.makedirs(args.output_dir, exist_ok=True)
    
    # تحديد الجهاز
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # تحميل النموذج
    model = load_model(args.model, args.backbone, device)
    print(f"Model loaded from {args.model}")
    
    # معالجة الصور
    if args.batch:
        process_directory(args.image, model, device, args.output_dir, args.threshold, args.show)
    else:
        process_single_image(args.image, model, device, args.output_dir, args.threshold, args.show)

if __name__ == "__main__":
    main() 