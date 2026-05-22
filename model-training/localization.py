import numpy as np
import os
import imageio.v2 as imageio
import skimage.transform
import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import torch.nn as nn
from torch.autograd import Function
import torch.backends.cudnn as cudnn
import argparse
import matplotlib.pyplot as plt
from PIL import Image

# استيراد النماذج
from models.convnext import ConvNextModel
from models.densenet import DenseNet121Model, DenseNet169Model
from models.resnet import ResNet152Model

# استيراد التحويلات
from utils.transforms import valid_transform

# تعريف الفئات
CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
    'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

def debug_print(msg):
    """طباعة رسائل التصحيح"""
    print(f"[DEBUG] {msg}")

def parse_args():
    """تحليل معلمات سطر الأوامر"""
    parser = argparse.ArgumentParser(description='Generate Grad-CAM visualizations for chest X-ray images')
    parser.add_argument('--image', type=str, required=True, help='Path to the image')
    parser.add_argument('--model', type=str, default='best_model_auroc_0.891.pth', help='Path to the model')
    parser.add_argument('--model_type', type=str, default='convnext', 
                        choices=['convnext', 'densenet121', 'densenet169', 'resnet152'], 
                        help='Model type')
    parser.add_argument('--output', type=str, default='heatmap_output.png', help='Output file path')
    parser.add_argument('--threshold', type=float, default=0.5, help='Prediction threshold')
    parser.add_argument('--top_k', type=int, default=3, help='Show top K predictions')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    return parser.parse_args()

class GradCAM:
    """
    تنفيذ خوارزمية Grad-CAM لتصور المناطق المهمة في الصورة
    
    المراجع:
    - Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
    - https://arxiv.org/abs/1610.02391
    """
    def __init__(self, model, target_layer, cuda=False):
        self.model = model
        self.target_layer = target_layer
        self.cuda = cuda
        
        self.gradients = None
        self.activations = None
        
        # تسجيل الخطافات
        self.target_layer.register_forward_hook(self._hook_feature)
        self.target_layer.register_full_backward_hook(self._hook_gradient)
        
        # وضع النموذج في وضع التقييم
        self.model.eval()

    def _hook_feature(self, module, input, output):
        """تخزين مخرجات الطبقة المستهدفة"""
        self.activations = output.detach()

    def _hook_gradient(self, module, grad_in, grad_out):
        """تخزين تدرجات الطبقة المستهدفة"""
        self.gradients = grad_out[0].detach()

    def _normalize(self, grads):
        """تطبيع التدرجات"""
        return grads / (torch.sqrt(torch.mean(torch.square(grads))) + 1e-5)

    def _compute_grad_weights(self, grads):
        """حساب أوزان التدرج"""
        return torch.mean(grads, dim=(2, 3), keepdim=True)

    def forward(self, x):
        """تمرير الصورة عبر النموذج"""
        return self.model(x)

    def backward(self, idx):
        """حساب التدرجات للفئة المحددة"""
        self.model.zero_grad()
        
        # الحصول على نتيجة التنبؤ للفئة المحددة
        output = self.model.forward(self.input)
        
        # إنشاء تنسور واحد للفئة المحددة
        target = torch.zeros_like(output)
        target[0, idx] = 1.0
        
        # حساب التدرجات
        output.backward(gradient=target, retain_graph=True)
        
    def __call__(self, x, idx=None):
        """
        إنشاء خريطة حرارية للصورة
        
        المعلمات:
            x (torch.Tensor): صورة الإدخال
            idx (int): مؤشر الفئة المستهدفة
            
        الإرجاع:
            cam (numpy.ndarray): الخريطة الحرارية
        """
        # تخزين الإدخال
        self.input = x
        
        # تمرير الصورة عبر النموذج
        output = self.forward(x)
        
        if idx is None:
            # استخدام الفئة ذات أعلى احتمالية
            idx = torch.argmax(output, dim=1).item()
        
        # حساب التدرجات
        self.backward(idx)

        # الحصول على التدرجات والتنشيطات
        gradients = self.gradients
        activations = self.activations
        
        # حساب أوزان التدرج
        weights = self._compute_grad_weights(gradients)
        
        # إنشاء الخريطة الحرارية
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = torch.relu(cam)  # تطبيق ReLU للتركيز على المناطق الإيجابية
        
        # تطبيع الخريطة الحرارية
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-7)
        
        # تغيير حجم الخريطة الحرارية لتتناسب مع حجم الصورة الأصلية
        cam = torch.nn.functional.interpolate(cam, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        
        # تحويل إلى numpy
        cam = cam.cpu().detach().numpy()[0, 0]
        
        return cam

def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    """
    حفظ الخريطة الحرارية Grad-CAM
    
    المعلمات:
        filename (str): مسار الملف للحفظ
        gcam (numpy.ndarray): الخريطة الحرارية
        raw_image (numpy.ndarray): الصورة الأصلية
        paper_cmap (bool): استخدام نفس خريطة الألوان كما في الورقة البحثية
    """
    # تحويل الصورة الأصلية إلى نطاق 0-255
    raw_image = raw_image.astype(np.uint8)
    
    # تحويل الخريطة الحرارية إلى خريطة ألوان
    if paper_cmap:
        cmap = cv2.COLORMAP_JET
    else:
        cmap = cv2.COLORMAP_VIRIDIS
        
    gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cmap)
    
    # دمج الصورة الأصلية مع الخريطة الحرارية
    gcam = cv2.addWeighted(raw_image, 0.5, gcam, 0.5, 0)
    
    # حفظ الصورة
    cv2.imwrite(filename, gcam)

def get_target_layer(model, model_type):
    """
    الحصول على الطبقة المستهدفة للنموذج
    
    المعلمات:
        model (nn.Module): النموذج
        model_type (str): نوع النموذج
        
    الإرجاع:
        target_layer (nn.Module): الطبقة المستهدفة
    """
    if model_type == 'convnext':
        # استهداف الطبقة الأخيرة من ConvNeXt
        return model.model.stages[-1][-1].block[-1]
    elif model_type.startswith('densenet'):
        # استهداف الطبقة الأخيرة من DenseNet
        return model.model.features.norm5
    elif model_type == 'resnet152':
        # استهداف الطبقة الأخيرة من ResNet
        return model.model.layer4[-1]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def predict_and_visualize(image_path, model, model_type, device, output_path, threshold=0.5, top_k=3):
    """
    التنبؤ وتصور الخرائط الحرارية للأمراض
    
    المعلمات:
        image_path (str): مسار الصورة
        model (nn.Module): النموذج المدرب
        model_type (str): نوع النموذج
        device (torch.device): الجهاز (CPU أو GPU)
        output_path (str): مسار الملف للحفظ
        threshold (float): عتبة التنبؤ
        top_k (int): عدد أعلى التنبؤات للعرض
    """
    # تحميل الصورة
    image = Image.open(image_path).convert('RGB')
    raw_image = np.array(image)
    
    # تطبيق التحويلات
    image_tensor = valid_transform(image).unsqueeze(0).to(device)
    
    # التنبؤ
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
    
    # إنشاء قاموس التنبؤات
    predictions = {}
    for i, class_name in enumerate(CLASS_NAMES):
        predictions[class_name] = float(probabilities[i])
    
    # ترتيب التنبؤات تنازلياً
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    
    # طباعة النتائج
    print("\nPrediction Results:")
    print("-" * 40)
    for disease, prob in sorted_preds:
        status = "POSITIVE" if prob >= threshold else "NEGATIVE"
        print(f"{disease:20s}: {prob:.4f} - {status}")
    
    # الحصول على الطبقة المستهدفة
    target_layer = get_target_layer(model, model_type)
    
    # إنشاء كائن Grad-CAM
    grad_cam = GradCAM(model, target_layer, cuda=(device.type == 'cuda'))
    
    # إنشاء صورة متعددة للخرائط الحرارية
    fig, axes = plt.subplots(1, top_k + 1, figsize=(15, 5))
    
    # عرض الصورة الأصلية
    axes[0].imshow(raw_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # عرض الخرائط الحرارية لأعلى K تنبؤات
    for i in range(min(top_k, len(sorted_preds))):
        disease, prob = sorted_preds[i]
        disease_idx = CLASS_NAMES.index(disease)
        
        # إنشاء الخريطة الحرارية
        cam = grad_cam(image_tensor, disease_idx)
        
        # عرض الخريطة الحرارية
        axes[i+1].imshow(raw_image)
        axes[i+1].imshow(cam, cmap='jet', alpha=0.5)
        axes[i+1].set_title(f'{disease}: {prob:.4f}')
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")
    plt.show()

def main():
    """الدالة الرئيسية"""
    # تحليل المعلمات
    args = parse_args()
    
    # تحديد الجهاز
    cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print(f"Using device: {device}")
    
    # إنشاء النموذج
    if args.model_type == "convnext":
        model = ConvNextModel(num_classes=len(CLASS_NAMES))
    elif args.model_type == "densenet121":
        model = DenseNet121Model(num_classes=len(CLASS_NAMES))
    elif args.model_type == "densenet169":
        model = DenseNet169Model(num_classes=len(CLASS_NAMES))
    elif args.model_type == "resnet152":
        model = ResNet152Model(num_classes=len(CLASS_NAMES))
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # تحميل وزن النموذج
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Model loaded: {args.model}")
    print(f"Model type: {args.model_type}")
    
    # التنبؤ وتصور الخرائط الحرارية
    predict_and_visualize(
        image_path=args.image,
        model=model,
        model_type=args.model_type,
        device=device,
        output_path=args.output,
        threshold=args.threshold,
        top_k=args.top_k
    )

if __name__ == "__main__":
    main()
