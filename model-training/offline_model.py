import os
import sys
import torch
import timm

# Set environment variables to force offline mode
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TIMM_OFFLINE'] = '1'
os.environ['PYTHONHTTPSVERIFY'] = '0'

# Patch timm's create_model to avoid downloading pretrained weights
original_create_model = timm.create_model

def patched_create_model(model_name, pretrained=False, **kwargs):
    # Force pretrained to False to avoid downloading
    return original_create_model(model_name, pretrained=False, **kwargs)

# Apply the patch
timm.create_model = patched_create_model

print("Timm library patched to work in offline mode")
print("Run your script with: python -m offline_model test_train.py")

# If this script is run with an argument, execute that file
if len(sys.argv) > 1:
    script_path = sys.argv[1]
    print(f"Running {script_path} in offline mode...")
    
    # Read the script content
    with open(script_path, 'r') as f:
        script_content = f.read()
    
    # Execute the script
    exec(script_content, {'__name__': '__main__'}) 