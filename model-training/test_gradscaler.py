import warnings
import torch
from torch.amp import autocast, GradScaler

# Show all warnings
warnings.filterwarnings('always')

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Create a GradScaler instance
print("Creating GradScaler...")
scaler = GradScaler()
print("GradScaler created")

# Create a simple model and optimizer
model = torch.nn.Linear(10, 2).to(device)
optimizer = torch.optim.Adam(model.parameters())

# Create a dummy input
x = torch.randn(1, 10).to(device)

# Forward pass with autocast
print("Starting forward pass with autocast...")
with autocast(device_type=device.type, enabled=True):
    y = model(x)
    loss = y.sum()
    
print("Forward pass completed")

# Backward pass with scaler
print("Starting backward pass with scaler...")
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
print("Backward pass completed")

print("Test completed successfully") 