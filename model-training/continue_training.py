import os
import torch
import traceback
import sys
import multiprocessing

print("Script starting...")
print(f"Python version: {sys.version}")
print(f"Platform: {sys.platform}")

# Fix for Windows multiprocessing
if sys.platform.startswith('win'):
    # Set multiprocessing start method to 'spawn'
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("Set multiprocessing start method to 'spawn'")
    except RuntimeError:
        print("Multiprocessing start method already set")
    except Exception as e:
        print(f"Error setting multiprocessing start method: {e}")

# Import after setting multiprocessing method
try:
    print("Importing from test_train...")
    from test_train import *
    print("Import successful")
    
    def main():
        print("=== Advanced Chest X-ray Model Fine-tuning ===")
        print(f"Python version: {sys.version}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Override config for Windows compatibility
        CFG.NUM_WORKERS = 0  # Disable multiprocessing for DataLoader
        
        print(f"Loading model from: {CFG.CHECKPOINT_LOAD_PATH}")
        print(f"Image size: {CFG.IMG_SIZE}x{CFG.IMG_SIZE}")
        print(f"Batch size: {CFG.BATCH_SIZE} (x{CFG.ACCUM_STEPS} accumulation steps)")
        print(f"Learning rates: {CFG.LR} (backbone), {CFG.HEAD_LR} (head)")
        print(f"Bounding box loss weight: {CFG.BBOX_LOSS_WEIGHT}")
        print(f"Device: {CFG.DEVICE}")
        print(f"Workers: {CFG.NUM_WORKERS}")
        
        # Function to load model with architecture changes
        def load_model_with_compatible_layers(model, checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=CFG.DEVICE)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Handle potential module prefix
            if next(iter(state_dict)).startswith('module.'):
                state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
            
            # Get current model state dict
            model_state_dict = model.state_dict()
            
            # Create a new state dict with only compatible layers
            compatible_state_dict = {}
            incompatible_keys = []
            
            for name, param in state_dict.items():
                if name in model_state_dict:
                    # Check if shapes match
                    if param.shape == model_state_dict[name].shape:
                        compatible_state_dict[name] = param
                    else:
                        incompatible_keys.append((name, param.shape, model_state_dict[name].shape))
                else:
                    incompatible_keys.append((name, param.shape, None))
            
            # Load compatible weights
            model.load_state_dict(compatible_state_dict, strict=False)
            
            # Print stats
            total_params = len(state_dict)
            loaded_params = len(compatible_state_dict)
            print(f"Successfully loaded {loaded_params}/{total_params} parameters ({loaded_params/total_params*100:.1f}%)")
            print(f"Incompatible layers: {len(incompatible_keys)}")
            
            # Print first few incompatible keys
            if incompatible_keys:
                print("Sample of incompatible layers:")
                for i, (name, old_shape, new_shape) in enumerate(incompatible_keys[:5]):
                    print(f"  {name}: checkpoint shape {old_shape}, model shape {new_shape}")
                if len(incompatible_keys) > 5:
                    print(f"  ... and {len(incompatible_keys) - 5} more")
            
            return loaded_params / total_params
        
        # Initialize model
        model = AdvancedChestModel(CFG.MODEL_NAME, metadata_features=3, pretrained=False)
        model.to(CFG.DEVICE)
        
        # Load checkpoint with compatibility handling
        if os.path.exists(CFG.CHECKPOINT_LOAD_PATH):
            try:
                load_percentage = load_model_with_compatible_layers(model, CFG.CHECKPOINT_LOAD_PATH)
                if load_percentage < 0.7:
                    print(f"Warning: Only {load_percentage*100:.1f}% of parameters were loaded. Model may not perform as expected.")
                print(f"Successfully loaded model from {CFG.CHECKPOINT_LOAD_PATH}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                traceback.print_exc()
                exit(1)
        else:
            print(f"Checkpoint file not found: {CFG.CHECKPOINT_LOAD_PATH}")
            exit(1)
        
        # Load dataset
        print("Loading data...")
        df_main = pd.read_csv(CFG.DATA_ENTRY_PATH)
        df = df_main.copy()
        df_findings = df[df['Finding Labels'] != 'No Finding'].copy()
        df_nofinding = df[df['Finding Labels'] == 'No Finding'].copy()
        nofinding_sample_fraction = 0.35
        df_nofinding_sampled = df_nofinding.sample(frac=nofinding_sample_fraction, random_state=CFG.SEED)
        df_combined = pd.concat([df_findings, df_nofinding_sampled], ignore_index=True)
        df_processed = df_combined.sample(frac=1, random_state=CFG.SEED).reset_index(drop=True)
        print(f"Dataset size: {len(df_processed)} (after sampling)")
        
        # Preprocessing metadata
        print("Preprocessing metadata...")
        df_processed, _, _ = preprocess_metadata(df_processed)
        
        # Split data
        print("Splitting data...")
        train_df, valid_df = train_test_split(df_processed, test_size=0.2, random_state=CFG.SEED)
        print(f"Train: {len(train_df)}, Valid: {len(valid_df)}")
        
        # Load bounding boxes
        print("Loading bounding boxes...")
        bbox_dict = load_bounding_boxes(CFG.BBOX_PATH)
        print(f"Loaded {len(bbox_dict)} images with bounding boxes")
        
        # Set up transforms
        train_transform = get_transforms(CFG.IMG_SIZE, is_train=True)
        valid_transform = get_transforms(CFG.IMG_SIZE, is_train=False)
        
        # Create datasets
        print("Creating datasets...")
        train_dataset = ChestXrayDataset(CFG.IMAGE_DIR, train_df, bbox_dict, transform=train_transform, train=True)
        valid_dataset = ChestXrayDataset(CFG.IMAGE_DIR, valid_df, bbox_dict, transform=valid_transform, train=False)
        
        # Calculate sample weights for imbalance
        print("Calculating sample weights...")
        labels_np = train_dataset.labels
        pos_counts = np.maximum(labels_np.sum(axis=0), 1e-6)
        class_weights_sampler = 1.0 / pos_counts
        no_finding_samples = np.all(labels_np == 0, axis=1)
        sample_weights = np.maximum(np.max(labels_np * class_weights_sampler, axis=1), 1e-6)
        sample_weights[no_finding_samples] *= 2.0  # Weight for "No Finding" samples
        
        # Create sampler and data loaders
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, sampler=sampler, 
                                num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=CFG.BATCH_SIZE * 2, shuffle=False, 
                                num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=False)
        
        # Set up optimizer with differential learning rates
        print("Setting up optimizer...")
        backbone_params = [p for n, p in model.model.named_parameters() if p.requires_grad]
        head_params = [p for n, p in model.named_parameters() if p.requires_grad and 'model.' not in n]
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': CFG.LR, 'name': 'backbone'}, 
            {'params': head_params, 'lr': CFG.HEAD_LR, 'name': 'heads'}
        ], weight_decay=CFG.WEIGHT_DECAY)
        
        # Set up scheduler
        print("Setting up scheduler...")
        num_train_optimizer_steps = math.ceil(len(train_loader) / CFG.ACCUM_STEPS) * CFG.EPOCHS
        num_warmup_optimizer_steps = math.ceil(len(train_loader) / CFG.ACCUM_STEPS) * CFG.WARMUP_EPOCHS
        scheduler = None
        if CFG.SCHEDULER == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=num_train_optimizer_steps - num_warmup_optimizer_steps, 
                eta_min=CFG.MIN_LR
            )
        
        # Set up loss functions
        criterion_cls = FocalLoss(label_smoothing=CFG.LABEL_SMOOTHING).to(CFG.DEVICE)
        criterion_bbox = nn.SmoothL1Loss(reduction='mean').to(CFG.DEVICE)
        
        # Start training
        print("\n=== Starting Fine-tuning ===")
        scaler = torch.amp.GradScaler()
        best_auroc = 0.0
        metrics = defaultdict(list)
        epochs_no_improve = 0
        
        # Run training loop
        for epoch in range(CFG.EPOCHS):
            # Train one epoch
            train_loss, train_cls_loss, train_bbox_loss = train_one_epoch(
                model, train_loader, optimizer, criterion_cls, criterion_bbox, 
                CFG.DEVICE, scaler, epoch, scheduler
            )
            
            # Validate
            val_loss, val_cls_loss, val_bbox_loss, mean_auroc, class_aurocs, all_targets, all_outputs, \
            no_finding_accuracy, no_finding_threshold, confusion_matrices = validate_one_epoch(
                model, valid_loader, criterion_cls, criterion_bbox, CFG.DEVICE, epoch
            )
            
            # Update metrics
            metrics['train_loss'].append(train_loss)
            metrics['train_cls_loss'].append(train_cls_loss)
            metrics['train_bbox_loss'].append(train_bbox_loss)
            metrics['val_loss'].append(val_loss)
            metrics['val_cls_loss'].append(val_cls_loss)
            metrics['val_bbox_loss'].append(val_bbox_loss)
            metrics['mean_auroc'].append(mean_auroc)
            metrics['no_finding_accuracy'].append(no_finding_accuracy)
            
            # Save current model
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'metrics': dict(metrics),
                'class_aurocs': class_aurocs,
                'no_finding_threshold': no_finding_threshold
            }
            torch.save(checkpoint_data, f'fine_tuned_model_epoch_{epoch+1}.pth')
            
            # Save best model
            if mean_auroc > best_auroc:
                best_auroc = mean_auroc
                best_model_path = f'fine_tuned_model_best_auroc_{mean_auroc:.4f}.pth'
                torch.save(checkpoint_data, best_model_path)
                print(f"*** Saved new best model with AUROC: {mean_auroc:.4f} at {best_model_path} ***")
                epochs_no_improve = 0
                
                # Generate plots
                plot_training_progress(metrics, CFG.EPOCHS, save_path=f'fine_tuning_progress_best_epoch_{epoch+1}.png')
                plot_class_metrics(class_aurocs, save_path=f'fine_tuning_class_aurocs_best_epoch_{epoch+1}.png')
                plot_roc_curves(all_targets, all_outputs, TRAIN_CLASSES, epoch, mean_auroc, save_path=f'fine_tuning_roc_curves_best_epoch_{epoch+1}.png')
                plot_confusion_matrices(confusion_matrices, TRAIN_CLASSES + ['No Finding'], epoch, save_path=f'fine_tuning_confusion_matrices_best_epoch_{epoch+1}')
            else:
                epochs_no_improve += 1
                
            # Print summary
            print(f"\nEpoch {epoch+1}/{CFG.EPOCHS} Summary:")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.2e} (Backbone), {optimizer.param_groups[1]['lr']:.2e} (Heads)")
            print(f"  Train Loss: {train_loss:.4f} (Cls: {train_cls_loss:.4f}, Box: {train_bbox_loss:.4f})")
            print(f"  Valid Loss: {val_loss:.4f} (Cls: {val_cls_loss:.4f}, Box: {val_bbox_loss:.4f})")
            print(f"  Mean Valid AUROC: {mean_auroc:.4f} (Best: {best_auroc:.4f})")
            print(f"  No Finding Accuracy: {no_finding_accuracy:.2f}% at threshold {no_finding_threshold:.2f}")
            
            # Check early stopping
            if epochs_no_improve >= CFG.PATIENCE:
                print(f"Early stopping triggered after {CFG.PATIENCE} epochs without improvement.")
                break
        
        print("\n=== Fine-tuning Complete ===")
        print(f"Best validation AUROC achieved: {best_auroc:.4f}")

except Exception as e:
    print(f"Error in main script: {e}")
    traceback.print_exc()

if __name__ == "__main__":
    # This is required for Windows to properly handle multiprocessing
    multiprocessing.freeze_support()
    main() 