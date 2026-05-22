import os
from PIL import Image
import pandas as pd

def check_images(image_dir, df):
    errors = []
    for img_name in df['Image Index']:
        img_path = find_image_path(img_name, image_dir)
        if img_path is None:
            errors.append(f"Image not found: {img_name}")
            continue
        try:
            img = Image.open(img_path)
            img.verify()
            img.close()
        except Exception as e:
            errors.append(f"Corrupted image {img_name}: {e}")
    return errors

def find_image_path(image_name, base_folder):
    for i in range(1, 13):
        image_path = os.path.join(base_folder, f'images_{i:03d}', 'images', image_name)
        if os.path.exists(image_path):
            return image_path
    image_path_flat = os.path.join(base_folder, 'images', image_name)
    if os.path.exists(image_path_flat):
        return image_path_flat
    return None

if __name__ == "__main__":
    df = pd.read_csv('data/Data_Entry_2017.csv')
    errors = check_images('data', df)
    if errors:
        print(f"Found {len(errors)} errors:")
        for err in errors:
            print(err)
    else:
        print("All images are valid.")