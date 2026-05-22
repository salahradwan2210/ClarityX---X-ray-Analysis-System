import argparse
import os
import sys
from utils.bbox_utils import load_bbox_data, get_image_bboxes, draw_bboxes_on_image
from PIL import Image

def parse_args():
    """تحليل معلمات سطر الأوامر"""
    parser = argparse.ArgumentParser(description='Visualize bounding boxes from BBox_List_2017.csv')
    parser.add_argument('--image', type=str, required=True, help='Path to the image or directory of images')
    parser.add_argument('--bbox_file', type=str, default='./data/BBox_List_2017.csv', help='Path to the bounding box CSV file')
    parser.add_argument('--output_dir', type=str, default='bbox_outputs', help='Directory to save output images')
    parser.add_argument('--batch', action='store_true', help='Process a directory of images')
    return parser.parse_args()

def process_single_image(image_path, bbox_data, output_dir):
    """معالجة صورة واحدة"""
    # الحصول على اسم الملف من المسار
    image_name = os.path.basename(image_path)
    
    # الحصول على مربعات الإحاطة للصورة
    bbox_list = get_image_bboxes(image_name, bbox_data)
    
    if not bbox_list:
        print(f"No bounding boxes found for {image_name}")
        return
    
    print(f"Found {len(bbox_list)} bounding boxes for {image_name}")
    
    # إنشاء مجلد الإخراج إذا لم يكن موجوداً
    os.makedirs(output_dir, exist_ok=True)
    
    # رسم مربعات الإحاطة على الصورة
    output_path = os.path.join(output_dir, f"bbox_{image_name}")
    image_with_bbox = draw_bboxes_on_image(image_path, bbox_list, output_path)
    
    if image_with_bbox:
        print(f"Image with bounding boxes saved to {output_path}")
    else:
        print(f"Failed to process {image_name}")

def process_directory(directory_path, bbox_data, output_dir):
    """معالجة مجلد من الصور"""
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
        process_single_image(image_path, bbox_data, output_dir)

def main():
    """الدالة الرئيسية"""
    # تحليل المعلمات
    args = parse_args()
    
    # تحميل بيانات مربعات الإحاطة
    bbox_data = load_bbox_data(args.bbox_file)
    
    if bbox_data is None:
        print(f"Failed to load bounding box data from {args.bbox_file}")
        sys.exit(1)
    
    # معالجة الصور
    if args.batch:
        process_directory(args.image, bbox_data, args.output_dir)
    else:
        process_single_image(args.image, bbox_data, args.output_dir)

if __name__ == "__main__":
    main() 