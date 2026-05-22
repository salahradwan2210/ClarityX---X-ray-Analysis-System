import os
import argparse
from train_advanced import train_advanced

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Run advanced chest X-ray classifier training")
    
    # Add optional override arguments
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--image_size", type=int, default=512, help="Image size")
    parser.add_argument("--start_from_scratch", action="store_true", help="Start training from scratch")
    parser.add_argument("--checkpoint_path", type=str, default="advanced_outputs/checkpoint_latest.pth", 
                        help="Path to checkpoint file for resuming training")
    parser.add_argument("--base_lr", type=float, default=5e-6, help="Base learning rate for resuming training")
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    checkpoint_exists = os.path.exists(args.checkpoint_path)
    if checkpoint_exists:
        print(f"Found checkpoint at {args.checkpoint_path}")
        print("Training will resume from the checkpoint")
    else:
        print(f"Warning: Checkpoint not found at {args.checkpoint_path}")
        print("Training will start from scratch")
    
    # Set paths from test_train.py
    base_path = 'data'
    resized_base_path = 'data_resized_512'
    image_dir = resized_base_path
    data_entry_path = os.path.join(base_path, 'Data_Entry_2017.csv')
    bbox_path = os.path.join(base_path, 'BBox_List_2017.csv')
    train_val_list_path = os.path.join(base_path, 'train_val_list.txt')
    
    # Configure training arguments
    training_args = argparse.Namespace(
        # Data paths
        data_path=image_dir,
        csv_path=data_entry_path,
        list_path=train_val_list_path,
        output_dir="advanced_outputs",
        
        # Training parameters
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        base_lr=args.base_lr,
        head_lr=args.base_lr * 3,  # Head LR is typically higher than base
        min_lr=1e-7,
        warmup_start_lr=1e-6,
        weight_decay=0.01,
        dropout=0.4,
        label_smoothing=0.1,
        mixup_alpha=0.5,
        mixup_prob=0.7,
        warmup_epochs=2,
        seed=42,
        patience=7,
        num_workers=2,
        
        # Checkpoint for resuming
        checkpoint_path=args.checkpoint_path,
        start_from_scratch=args.start_from_scratch,
        
        # Other options
        include_no_finding=False
    )
    
    print("Starting advanced training with the following configuration:")
    print(f"• Data path: {training_args.data_path}")
    print(f"• CSV path: {training_args.csv_path}")
    print(f"• List path: {training_args.list_path}")
    print(f"• Epochs: {training_args.epochs}")
    print(f"• Batch size: {training_args.batch_size}")
    print(f"• Image size: {training_args.image_size}")
    print(f"• Base learning rate: {training_args.base_lr}")
    print(f"• Checkpoint path: {training_args.checkpoint_path}")
    print(f"• Starting from scratch: {training_args.start_from_scratch}")
    
    # Run training
    best_auroc = train_advanced(training_args)
    print(f"Training completed with best AUROC: {best_auroc:.4f}") 