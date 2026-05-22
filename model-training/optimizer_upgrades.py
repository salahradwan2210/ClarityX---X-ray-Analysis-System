import torch
import torch.nn as nn

def enable_gradient_checkpointing(model):
    """
    Enable gradient checkpointing for a model to reduce memory usage
    
    Parameters:
        model (nn.Module): The model to enable gradient checkpointing for
        
    Returns:
        model (nn.Module): The model with gradient checkpointing enabled
    """
    if hasattr(model, 'set_grad_checkpointing'):
        model.set_grad_checkpointing(True)
    else:
        # For ConvNext models that don't have set_grad_checkpointing method
        for module in model.modules():
            if isinstance(module, nn.modules.transformer.Transformer):
                module.gradient_checkpointing = True
            elif hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True
    
    return model

def freeze_layers(model, freeze_ratio=0.7):
    """
    Freeze a portion of the model's layers to reduce memory usage and speed up training
    
    Parameters:
        model (nn.Module): The model to freeze layers for
        freeze_ratio (float): The ratio of layers to freeze (0.0-1.0)
        
    Returns:
        model (nn.Module): The model with frozen layers
    """
    trainable_params = []
    frozen_params = []
    
    # Get all parameters
    all_params = list(model.parameters())
    total_params = len(all_params)
    
    # Calculate how many parameters to freeze
    num_freeze = int(total_params * freeze_ratio)
    
    # Freeze early layers
    for i, param in enumerate(all_params):
        if i < num_freeze:
            param.requires_grad = False
            frozen_params.append(param)
        else:
            trainable_params.append(param)
    
    print(f"Frozen parameters: {len(frozen_params)} / {total_params}")
    print(f"Trainable parameters: {len(trainable_params)} / {total_params}")
    
    return model

def apply_model_optimizations(model, freeze_ratio=0.7, use_gradient_checkpointing=True):
    """
    Apply memory and performance optimizations to a model
    
    Parameters:
        model (nn.Module): The model to optimize
        freeze_ratio (float): The ratio of layers to freeze (0.0-1.0)
        use_gradient_checkpointing (bool): Whether to use gradient checkpointing
        
    Returns:
        model (nn.Module): The optimized model
    """
    # Free memory before making changes
    torch.cuda.empty_cache()
    
    # Freeze layers to reduce memory usage and speed up training
    model = freeze_layers(model, freeze_ratio)
    
    # Enable gradient checkpointing to reduce memory usage
    if use_gradient_checkpointing:
        model = enable_gradient_checkpointing(model)
    
    return model

class GradientAccumulationStep:
    """
    Context manager for gradient accumulation
    
    Usage:
        accumulation_steps = 4
        with GradientAccumulationStep(model, optimizer, accumulation_steps, scaler=None):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss = loss / accumulation_steps  # Scale the loss
            # Backward and step are handled by the context manager
    """
    def __init__(self, model, optimizer, accumulation_steps=1, scaler=None):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.scaler = scaler
        self.step_counter = 0
        
    def __enter__(self):
        self.step_counter += 1
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
        
        # Handle accumulation step
        if self.step_counter % self.accumulation_steps == 0:
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
            
        return True

def optimize_loaded_data(dataset, cache_size=1000):
    """
    Optimize a dataset for faster loading
    
    Parameters:
        dataset (torch.utils.data.Dataset): The dataset to optimize
        cache_size (int): The number of images to cache in memory
        
    Returns:
        dataset (torch.utils.data.Dataset): The optimized dataset
    """
    # Pin memory for CUDA transfers
    if hasattr(dataset, 'pin_memory'):
        dataset.pin_memory = True
    
    # Enable caching for the first cache_size images
    if hasattr(dataset, 'cache_images') and callable(dataset.cache_images):
        dataset.cache_images(limit=cache_size)
    
    return dataset 