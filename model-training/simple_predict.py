import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import argparse

# Define the class names
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

def visualize_with_fixed_boxes(image_path, conditions, output_path=None):
    """
    Visualize chest X-ray with fixed bounding boxes for specified conditions
    
    Args:
        image_path: Path to the X-ray image
        conditions: Dictionary of condition names and probabilities
        output_path: Path to save the output visualization
    """
    # Load the image
    image = Image.open(image_path).convert('RGB')
    orig_width, orig_height = image.size
    
    # Set up the visualization
    plt.figure(figsize=(12, 10))
    plt.imshow(image)
    plt.axis('off')
    
    # Draw bounding boxes for conditions
    detected_conditions = []
    
    for class_name, prob in conditions.items():
        if class_name in CLASS_NAMES:
            detected_conditions.append(f"{class_name}: {prob:.2f}")
            
            # Predefined bounding box coordinates for common conditions
            if class_name == "Atelectasis":
                # Atelectasis typically affects the lower lobes
                y_pixel = orig_height * 0.4  # Start around middle of lung
                h_pixel = orig_height * 0.5  # Cover bottom half
                x_pixel = orig_width * 0.1   # Start from left side
                w_pixel = orig_width * 0.35  # Cover left lung area
                
            elif class_name == "Effusion":
                # Pleural effusion typically affects the lower lateral aspects
                y_pixel = orig_height * 0.6  # Start at lower third
                h_pixel = orig_height * 0.35 # Cover bottom portion
                x_pixel = orig_width * 0.7   # Right side
                w_pixel = orig_width * 0.25  # Width of right lateral region
                
            elif class_name == "Cardiomegaly":
                # Enlarged heart is central and lower
                x_pixel = orig_width * 0.3
                y_pixel = orig_height * 0.4
                w_pixel = orig_width * 0.4
                h_pixel = orig_height * 0.4
                
            elif class_name == "Pneumothorax":
                # Can occur on either side, upper lobes
                x_pixel = orig_width * 0.6
                y_pixel = orig_height * 0.2
                w_pixel = orig_width * 0.3
                h_pixel = orig_height * 0.4
                
            elif class_name == "Infiltration":
                # Can be diffuse or patchy
                x_pixel = orig_width * 0.25
                y_pixel = orig_height * 0.25
                w_pixel = orig_width * 0.5
                h_pixel = orig_height * 0.5
                
            elif class_name == "Mass":
                # Often localized
                x_pixel = orig_width * 0.6
                y_pixel = orig_height * 0.3
                w_pixel = orig_width * 0.2
                h_pixel = orig_height * 0.2
                
            elif class_name == "Nodule":
                # Small, localized
                x_pixel = orig_width * 0.4
                y_pixel = orig_height * 0.3
                w_pixel = orig_width * 0.15
                h_pixel = orig_height * 0.15
                
            elif class_name == "Pneumonia":
                # Can be lobar or diffuse
                x_pixel = orig_width * 0.2
                y_pixel = orig_height * 0.3
                w_pixel = orig_width * 0.3
                h_pixel = orig_height * 0.4
                
            else:
                # Default bounding box for other conditions
                x_pixel = orig_width * 0.3
                y_pixel = orig_height * 0.3
                w_pixel = orig_width * 0.4
                h_pixel = orig_height * 0.4
            
            # Draw bounding box
            rect = patches.Rectangle(
                (x_pixel, y_pixel), w_pixel, h_pixel, 
                linewidth=2, edgecolor='r', facecolor='none'
            )
            plt.gca().add_patch(rect)
            
            # Position text label above the box
            plt.text(
                x_pixel, y_pixel-5, f"{class_name}: {prob:.2f}", 
                color='white', fontsize=10, bbox=dict(facecolor='red', alpha=0.7)
            )
    
    # If no conditions detected, add "No Finding" to the title
    if not detected_conditions:
        plt.title("No Finding", fontsize=16)
    else:
        # Add a summary of all detected conditions to the figure
        plt.figtext(0.5, 0.01, ", ".join(detected_conditions), wrap=True, 
                    horizontalalignment='center', fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.8))
    
    # Save or display the result
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"Result saved to {output_path}")
    else:
        plt.tight_layout()
        plt.show()
    
    return detected_conditions

def main():
    parser = argparse.ArgumentParser(description='Chest X-ray Visualization with Fixed Bounding Boxes')
    parser.add_argument('--image', type=str, required=True, help='Path to the chest X-ray image')
    parser.add_argument('--output', type=str, default=None, help='Path to save the output visualization')
    args = parser.parse_args()
    
    # Check if image path is provided and exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        return
        
    # Define conditions for testing (replace with your actual conditions)
    # This would normally come from a model, but we're using fixed values for demonstration
    conditions = {
        "Atelectasis": 0.96,
        "Effusion": 0.92
    }
    
    # Process the image with fixed bounding boxes
    print(f"Processing image: {args.image}")
    detected_conditions = visualize_with_fixed_boxes(args.image, conditions, args.output)
    
    # Print detected conditions
    if detected_conditions:
        print("\nDetected conditions:")
        for condition in detected_conditions:
            print(f"  - {condition}")
    else:
        print("\nNo conditions detected.")

if __name__ == "__main__":
    main() 