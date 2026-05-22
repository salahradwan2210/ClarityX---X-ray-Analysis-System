import os
import sys
import timm
import torch
import warnings

print("Patching timm library to work offline...")

# Set environment variables
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TIMM_OFFLINE'] = '1'

# Monkey patch timm's create_model function
original_create_model = timm.create_model

def patched_create_model(model_name, pretrained=True, **kwargs):
    if pretrained:
        warnings.warn(f"Pretrained weights for {model_name} will not be loaded due to offline mode")
    return original_create_model(model_name, pretrained=False, **kwargs)

timm.create_model = patched_create_model

# Monkey patch torch hub
original_load_state_dict_from_url = torch.hub.load_state_dict_from_url

def patched_load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True, check_hash=False, file_name=None):
    print(f"[OFFLINE MODE] Skipping download from: {url}")
    print(f"Using local checkpoint instead")
    raise RuntimeError("Download prevented in offline mode")

torch.hub.load_state_dict_from_url = patched_load_state_dict_from_url

print("Patching complete. Now run your script with:")
print("python -c \"import patch_timm; import test_train; test_train.main()\"")

# If arguments are provided, run the specified script
if len(sys.argv) > 1:
    script_path = sys.argv[1]
    print(f"Running {script_path} with patched timm...")
    
    # Execute the script
    with open(script_path, 'r') as f:
        exec(f.read()) 