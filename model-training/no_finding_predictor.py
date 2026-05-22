import torch
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from pathlib import Path

def load_model(model_path, device='cuda'):
    """
    تحميل النموذج المدرب
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model

def predict_with_no_finding(image_path, model, transform, threshold=0.4, device='cuda'):
    """
    تنبؤ مع أخذ 'No Finding' في الاعتبار
    
    Args:
        image_path: مسار الصورة
        model: النموذج المدرب
        transform: التحويلات المطلوبة للصورة
        threshold: العتبة للتنبؤ بـ 'No Finding'
        device: الجهاز (CPU/GPU)
    
    Returns:
        predicted_class: الفئة المتوقعة
        confidence: درجة الثقة
        all_probs: احتمالات جميع الفئات
    """
    # الفئات المرضية
    disease_classes = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
        'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
        'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]
    
    # تحميل وتحويل الصورة
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # التنبؤ
    with torch.no_grad():
        predictions = model(image_tensor)
        if isinstance(predictions, dict):
            # إذا كان النموذج يعيد قاموس (بعض النماذج المتقدمة)
            predictions = predictions.get('class_logits', predictions)
        
        # تحويل التنبؤات إلى احتمالات
        probabilities = torch.sigmoid(predictions).cpu().numpy()[0]
    
    # التحقق من حالة 'No Finding'
    max_prob = np.max(probabilities)
    max_idx = np.argmax(probabilities)
    
    if max_prob < threshold:
        predicted_class = "No Finding"
        confidence = 1.0 - max_prob  # الثقة في أنها 'No Finding'
    else:
        predicted_class = disease_classes[max_idx]
        confidence = max_prob
    
    # إرجاع النتيجة
    return {
        'predicted_class': predicted_class,
        'confidence': float(confidence),
        'all_probabilities': {cls: float(prob) for cls, prob in zip(disease_classes, probabilities)},
        'no_finding_threshold': threshold,
        'is_no_finding': max_prob < threshold
    }

def process_batch_with_no_finding(image_dir, model, transform, threshold=0.4, device='cuda'):
    """
    معالجة مجموعة من الصور مع التعامل مع 'No Finding'
    """
    results = []
    
    # البحث عن الصور في المجلد
    image_paths = []
    for extension in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(list(Path(image_dir).glob(extension)))
    
    for img_path in image_paths:
        try:
            result = predict_with_no_finding(str(img_path), model, transform, threshold, device)
            result['image_path'] = str(img_path)
            results.append(result)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    return results

def save_results_to_csv(results, output_file):
    """
    حفظ النتائج في ملف CSV
    """
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def visualize_prediction(image_path, prediction, output_path=None):
    """
    إنشاء تصور للتنبؤ
    """
    # تحميل الصورة
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # استخراج البيانات
    predicted_class = prediction['predicted_class']
    confidence = prediction['confidence']
    probabilities = prediction['all_probabilities']
    
    # إنشاء عرض مرئي
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # عرض الصورة
    ax1.imshow(image)
    ax1.set_title(f"Prediction: {predicted_class} ({confidence:.2f})")
    ax1.axis('off')
    
    # عرض شريط الاحتمالات
    if predicted_class == "No Finding":
        # إضافة 'No Finding' كقيمة إضافية
        classes = list(probabilities.keys()) + ['No Finding']
        probs = list(probabilities.values()) + [confidence]
    else:
        classes = list(probabilities.keys())
        probs = list(probabilities.values())
    
    # ترتيب الاحتمالات تنازلياً
    if predicted_class == "No Finding":
        sorted_idx = np.argsort(probs)[-6:]  # أعلى 5 فئات + No Finding
    else:
        sorted_idx = np.argsort(probs)[-5:]  # أعلى 5 فئات فقط
    
    sorted_classes = [classes[i] for i in sorted_idx]
    sorted_probs = [probs[i] for i in sorted_idx]
    
    # رسم شريط الاحتمالات
    ax2.barh(sorted_classes, sorted_probs, color='skyblue')
    ax2.set_xlim(0, 1)
    ax2.set_xlabel('Probability')
    ax2.set_title('Top Predictions')
    
    # حفظ أو عرض الرسم
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def main():
    """
    المنهج الرئيسي للتنفيذ
    """
    import argparse
    from torchvision import transforms
    
    parser = argparse.ArgumentParser(description='Predict with No Finding handling')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory with images')
    parser.add_argument('--output_dir', type=str, default='predictions_with_no_finding', help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.4, help='Threshold for No Finding')
    parser.add_argument('--input_size', type=int, default=384, help='Input image size')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # التأكد من وجود مجلد الإخراج
    os.makedirs(args.output_dir, exist_ok=True)
    
    # تعريف التحويلات
    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # تحميل النموذج
    model = load_model(args.model_path, args.device)
    
    # معالجة الصور
    results = process_batch_with_no_finding(args.image_dir, model, transform, args.threshold, args.device)
    
    # حفظ النتائج
    save_results_to_csv(results, os.path.join(args.output_dir, 'predictions.csv'))
    
    # إنشاء تصور للنتائج
    for result in results:
        output_path = os.path.join(args.output_dir, f"{Path(result['image_path']).stem}_result.png")
        visualize_prediction(result['image_path'], result, output_path)
    
    print(f"Processing completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 