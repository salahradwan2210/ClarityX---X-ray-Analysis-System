import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from models.densenet121_model import DenseNet121Model
from utils import CLASS_NAMES, valid_transform

def load_model(model_path, model_type='densenet121', num_classes=14, device='cuda'):
    """
    تحميل النموذج المدرب
    
    المعلمات:
        model_path (str): مسار ملف النموذج
        model_type (str): نوع النموذج
        num_classes (int): عدد الفئات
        device (str): الجهاز المستخدم
        
    الإرجاع:
        model (nn.Module): النموذج المحمل
    """
    # إنشاء النموذج
    model = DenseNet121Model(num_classes=num_classes, pretrained=False)
    
    # تحميل الأوزان
    checkpoint = torch.load(model_path, map_location=device)
    
    # طباعة معلومات عن النموذج المحفوظ
    print("\nCheckpoint keys:", checkpoint.keys())
    if 'model_state_dict' in checkpoint:
        print("\nModel state dict keys:", list(checkpoint['model_state_dict'].keys())[:5], "...")
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        print("\nState dict keys:", list(checkpoint['state_dict'].keys())[:5], "...")
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("\nLoading entire checkpoint as model state dict")
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model

def predict_image(model, image_path, device='cuda', threshold=0.5):
    """
    التنبؤ باستخدام النموذج المدرب
    
    المعلمات:
        model (nn.Module): النموذج المدرب
        image_path (str): مسار الصورة
        device (str): الجهاز المستخدم
        threshold (float): عتبة التنبؤ
        
    الإرجاع:
        dict: نتائج التنبؤ
    """
    # التأكد من وجود الصورة
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # تحميل الصورة وتطبيق التحويلات
    image = Image.open(image_path).convert('RGB')
    
    # تطبيق التحويلات
    tensor = valid_transform(image).unsqueeze(0).to(device)
    
    # التنبؤ
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
    
    # تحويل الاحتمالات إلى تنبؤات
    preds = (probs >= threshold).astype(int)
    
    # إنشاء قاموس النتائج
    results = {
        'probabilities': probs,
        'predictions': preds,
        'class_names': CLASS_NAMES
    }
    
    # طباعة النتائج
    print("\nPrediction results:")
    for i, (prob, pred, cls) in enumerate(zip(probs, preds, CLASS_NAMES)):
        print(f"{cls}: {prob:.4f} ({pred})")
    
    return results

def visualize_prediction(image_path, predictions, output_path=None):
    """
    تصور نتائج التنبؤ
    
    المعلمات:
        image_path (str): مسار الصورة
        predictions (dict): نتائج التنبؤ
        output_path (str): مسار ملف الإخراج
    """
    # التأكد من وجود الصورة
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # تحميل الصورة
    image = Image.open(image_path).convert('RGB')
    
    # إنشاء الرسم
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # عرض الصورة
    ax1.imshow(image)
    ax1.set_title("Chest X-ray")
    ax1.axis('off')
    
    # عرض التنبؤات
    probs = predictions['probabilities']
    class_names = predictions['class_names']
    
    # ترتيب التنبؤات تنازلياً
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    sorted_names = [class_names[i] for i in sorted_idx]
    
    # رسم التنبؤات
    ax2.barh(np.arange(len(sorted_names)), sorted_probs, color='skyblue')
    ax2.set_yticks(np.arange(len(sorted_names)))
    ax2.set_yticklabels(sorted_names)
    ax2.set_xlabel('Probability')
    ax2.set_title('Predicted Diseases')
    ax2.set_xlim(0, 1)
    
    # حفظ الرسم
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    
    plt.show()

def parse_args():
    """تحليل معلمات سطر الأوامر"""
    parser = argparse.ArgumentParser(description='Predict using DenseNet121 model for chest X-ray classification')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default=None, help='Path to output visualization')
    parser.add_argument('--threshold', type=float, default=0.5, help='Prediction threshold')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    return parser.parse_args()

def main():
    """الدالة الرئيسية"""
    # تحليل المعلمات
    args = parse_args()
    
    # تحديد الجهاز
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # تحميل النموذج
    model = load_model(args.model_path, device=device)
    
    # التنبؤ باستخدام الصورة
    predictions = predict_image(model, args.image, device=device, threshold=args.threshold)
    
    # تصور التنبؤ
    visualize_prediction(args.image, predictions, output_path=args.output)

if __name__ == "__main__":
    main() 