import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def process_bbox_data(csv_path, output_dir='./'):
    # Read the original CSV file
    print("Reading CSV file...")
    df = pd.read_csv(csv_path)
    
    # Group by Image Index to combine multiple boxes
    print("Processing bounding boxes...")
    grouped = df.groupby('Image Index').agg({
        'Finding Label': list,
        'Bbox [x': list,
        'y': list,
        'w': list,
        'h]': list
    }).reset_index()
    
    # Create bounding box annotations in the required format
    def create_bbox_list(row):
        bbox_list = []
        for label, x, y, w, h in zip(row['Finding Label'], 
                                    row['Bbox [x'], 
                                    row['y'],
                                    row['w'],
                                    row['h]']):
            bbox_list.append({
                'label': label,
                'x': float(x),
                'y': float(y),
                'width': float(w),
                'height': float(h)
            })
        return str(bbox_list)
    
    # Apply the transformation
    print("Creating annotation format...")
    grouped['Bounding Boxes'] = grouped.apply(create_bbox_list, axis=1)
    
    # Keep only necessary columns
    final_df = grouped[['Image Index', 'Bounding Boxes']]
    
    # Split into train and validation sets
    print("Splitting into train and validation sets...")
    train_df, val_df = train_test_split(final_df, test_size=0.2, random_state=42)
    
    # Save the processed files
    print("Saving processed files...")
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'train_bbox.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_bbox.csv'), index=False)
    
    print(f"Done! Created train_bbox.csv ({len(train_df)} images) and val_bbox.csv ({len(val_df)} images)")
    return train_df, val_df

if __name__ == '__main__':
    bbox_csv = 'data/BBox_List_2017.csv'
    train_df, val_df = process_bbox_data(bbox_csv)
    
    # Print some statistics
    print("\nDataset Statistics:")
    print(f"Total images: {len(train_df) + len(val_df)}")
    print(f"Training images: {len(train_df)}")
    print(f"Validation images: {len(val_df)}") 