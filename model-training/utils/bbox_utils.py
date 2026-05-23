import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def load_bbox_data(bbox_file_path="./data/BBox_List_2017.csv"):
    """
    تحميل بيانات مربعات الإحاطة من ملف CSV
    
    المعلمات:
        bbox_file_path (str): مسار ملف مربعات الإحاطة
        
    الإرجاع:
        DataFrame: بيانات مربعات الإحاطة
    """
    try:
        bbox_data = pd.read_csv(bbox_file_path)
        print(f"Loaded {len(bbox_data)} bounding box annotations")
        return bbox_data
    except Exception as e:
        print(f"Error loading bounding box data: {str(e)}")
        return None

def get_image_bboxes(image_name, bbox_data=None):
    """
    الحصول على مربعات الإحاطة لصورة محددة
    
    المعلمات:
        image_name (str): اسم ملف الصورة
        bbox_data (DataFrame): بيانات مربعات الإحاطة
        
    الإرجاع:
        list: قائمة بمربعات الإحاطة للصورة المحددة
    """
    if bbox_data is None:
        return []
        
    try:
        # البحث عن مربعات الإحاطة للصورة المحددة
        image_bboxes = bbox_data[bbox_data['Image Index'] == image_name]
        
        if len(image_bboxes) == 0:
            return []
            
        # تحويل البيانات إلى قائمة من القواميس
        bbox_list = []
        for _, row in image_bboxes.iterrows():
            try:
                bbox = {
                    'label': row['Finding Label'],
                    'x': int(float(row['Bbox [x'])),
                    'y': int(float(row['y'])),
                    'width': int(float(row['w'])),
                    'height': int(float(row['h]']))
                }
                bbox_list.append(bbox)
            except (KeyError, ValueError, TypeError) as e:
                # قم بمعالجة أخطاء التحويل أو المفاتيح المفقودة بصمت
                continue
            
        return bbox_list
    except Exception as e:
        # قم بإرجاع قائمة فارغة في حالة حدوث أي خطأ
        return []

def draw_bboxes_on_image(image_path, bbox_list, output_path=None):
    """
    رسم مربعات الإحاطة على الصورة
    
    المعلمات:
        image_path (str): مسار الصورة
        bbox_list (list): قائمة بمربعات الإحاطة
        output_path (str): مسار حفظ الصورة (اختياري)
        
    الإرجاع:
        PIL.Image: الصورة مع مربعات الإحاطة
    """
    try:
        # فتح الصورة
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        # رسم كل مربع إحاطة
        colors = {
            'Atelectasis': (255, 0, 0),      # أحمر
            'Cardiomegaly': (0, 255, 0),     # أخضر
            'Effusion': (0, 0, 255),         # أزرق
            'Infiltration': (255, 255, 0),   # أصفر
            'Mass': (255, 0, 255),           # وردي
            'Nodule': (0, 255, 255),         # سماوي
            'Pneumonia': (128, 0, 0),        # بني محمر
            'Pneumothorax': (0, 128, 0)      # أخضر داكن
        }
        
        for bbox in bbox_list:
            label = bbox['label']
            x = bbox['x']
            y = bbox['y']
            width = bbox['width']
            height = bbox['height']
            
            # اختيار اللون بناءً على التسمية
            color = colors.get(label, (255, 255, 255))  # أبيض افتراضي
            
            # رسم المربع
            draw.rectangle([x, y, x + width, y + height], outline=color, width=3)
            
            # إضافة التسمية
            draw.text((x, y - 15), label, fill=color)
        
        # حفظ الصورة إذا تم تحديد مسار
        if output_path:
            image.save(output_path)
            print(f"Image with bounding boxes saved to {output_path}")
            
        return image
    except Exception as e:
        print(f"Error drawing bounding boxes: {str(e)}")
        return None

def calculate_iou(bbox1, bbox2):
    """
    حساب تقاطع الاتحاد (IoU) بين مربعي إحاطة
    
    المعلمات:
        bbox1 (dict): مربع الإحاطة الأول {x, y, width, height}
        bbox2 (dict): مربع الإحاطة الثاني {x, y, width, height}
        
    الإرجاع:
        float: قيمة تقاطع الاتحاد (0-1)
    """
    # حساب إحداثيات الزوايا
    x1_1, y1_1 = bbox1['x'], bbox1['y']
    x2_1, y2_1 = bbox1['x'] + bbox1['width'], bbox1['y'] + bbox1['height']
    
    x1_2, y1_2 = bbox2['x'], bbox2['y']
    x2_2, y2_2 = bbox2['x'] + bbox2['width'], bbox2['y'] + bbox2['height']
    
    # حساب مساحة التقاطع
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0  # لا يوجد تقاطع
        
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # حساب مساحة الاتحاد
    bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = bbox1_area + bbox2_area - intersection_area
    
    # حساب تقاطع الاتحاد
    iou = intersection_area / union_area
    
    return iou

def evaluate_heatmap_with_bbox(heatmap, bbox_list, threshold=0.5):
    """
    تقييم الخريطة الحرارية باستخدام مربعات الإحاطة
    
    المعلمات:
        heatmap (numpy.ndarray): الخريطة الحرارية
        bbox_list (list): قائمة بمربعات الإحاطة
        threshold (float): عتبة الخريطة الحرارية
        
    الإرجاع:
        dict: نتائج التقييم
    """
    # تحويل الخريطة الحرارية إلى قناع ثنائي
    binary_heatmap = (heatmap > threshold).astype(np.uint8)
    
    # العثور على المكونات المتصلة في الخريطة الحرارية
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_heatmap)
    
    # تحويل المكونات إلى مربعات إحاطة
    heatmap_bboxes = []
    for i in range(1, num_labels):  # نبدأ من 1 لتجاوز الخلفية
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        
        heatmap_bboxes.append({
            'x': x,
            'y': y,
            'width': width,
            'height': height
        })
    
    # حساب أفضل تقاطع اتحاد لكل مربع إحاطة حقيقي
    results = {
        'true_positives': 0,
        'false_positives': len(heatmap_bboxes),
        'false_negatives': len(bbox_list),
        'iou_scores': []
    }
    
    for gt_bbox in bbox_list:
        best_iou = 0
        best_idx = -1
        
        for i, pred_bbox in enumerate(heatmap_bboxes):
            iou = calculate_iou(gt_bbox, pred_bbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        
        if best_iou > 0.5:  # عتبة تقاطع الاتحاد
            results['true_positives'] += 1
            results['false_negatives'] -= 1
            results['false_positives'] -= 1
            results['iou_scores'].append(best_iou)
    
    # حساب الدقة والاستدعاء
    if results['true_positives'] + results['false_positives'] > 0:
        results['precision'] = results['true_positives'] / (results['true_positives'] + results['false_positives'])
    else:
        results['precision'] = 0
        
    if results['true_positives'] + results['false_negatives'] > 0:
        results['recall'] = results['true_positives'] / (results['true_positives'] + results['false_negatives'])
    else:
        results['recall'] = 0
        
    # حساب F1
    if results['precision'] + results['recall'] > 0:
        results['f1'] = 2 * results['precision'] * results['recall'] / (results['precision'] + results['recall'])
    else:
        results['f1'] = 0
        
    # حساب متوسط تقاطع الاتحاد
    if len(results['iou_scores']) > 0:
        results['mean_iou'] = sum(results['iou_scores']) / len(results['iou_scores'])
    else:
        results['mean_iou'] = 0
        
    return results 