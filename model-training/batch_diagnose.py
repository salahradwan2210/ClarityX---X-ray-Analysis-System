import os
import argparse
import glob
import pandas as pd
from diagnose_image import load_model, diagnose_image
import torch

def process_directory(model, input_dir, output_dir, extensions=('*.jpg', '*.jpeg', '*.png')):
    """
    Process all images in a directory and save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all images
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))
    
    # Remove duplicates and sort
    image_files = sorted(list(set(image_files)))
    
    if not image_files:
        print(f"No images found in {input_dir} with extensions {extensions}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    results = []
    for i, img_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(img_path)}")
        
        try:
            # Generate output path
            rel_path = os.path.relpath(img_path, input_dir)
            output_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + '_diagnosis.png')
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Diagnose image
            findings = diagnose_image(model, img_path, output_path, show_image=False)
            
            # Collect results
            for cls_name, prob in findings:
                results.append({
                    'image_path': img_path,
                    'diagnosis': cls_name,
                    'probability': prob
                })
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, 'diagnosis_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        
        # Generate summary
        summary = df.groupby('diagnosis').agg(
            count=('image_path', 'nunique'),
            avg_probability=('probability', 'mean')
        ).sort_values('count', ascending=False).reset_index()
        
        summary_path = os.path.join(output_dir, 'diagnosis_summary.csv')
        summary.to_csv(summary_path, index=False)
        print(f"Summary saved to {summary_path}")
        
        # Print summary
        print("\nDiagnosis Summary:")
        print("=" * 50)
        print(f"{'Diagnosis':<20} | {'Count':<10} | {'Avg Probability':<15}")
        print("-" * 50)
        for _, row in summary.iterrows():
            print(f"{row['diagnosis']:<20} | {row['count']:<10} | {row['avg_probability']:.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Batch process chest X-ray images')
    parser.add_argument('--input', type=str, required=True, help='Input directory containing images')
    parser.add_argument('--output', type=str, default='diagnosis_results', help='Output directory for results')
    parser.add_argument('--model', type=str, default='fine_tuned_model_best_auroc_0.9688.pth', help='Path to model weights')
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.isdir(args.input):
        print(f"Error: Input directory {args.input} does not exist")
        return
    
    # Load model
    try:
        model = load_model(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Process directory
    process_directory(model, args.input, args.output)
    
    print("Batch processing complete!")

if __name__ == "__main__":
    main() 