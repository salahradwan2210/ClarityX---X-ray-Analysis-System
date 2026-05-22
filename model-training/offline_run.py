import os
import sys
import torch

# Set environment variables to force offline mode
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TIMM_OFFLINE'] = '1'
os.environ['PYTHONHTTPSVERIFY'] = '0'

# Monkey patch torch hub to work offline
def patched_load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True, check_hash=False, file_name=None):
    print(f"[OFFLINE MODE] Would have downloaded: {url}")
    print(f"Using local checkpoint instead: {CFG.CHECKPOINT_LOAD_PATH}")
    return torch.load(CFG.CHECKPOINT_LOAD_PATH, map_location=map_location)

# Apply the patch
torch.hub.load_state_dict_from_url = patched_load_state_dict_from_url

print("Running test_train.py in offline mode...")
print("This will use your local model checkpoint instead of downloading pretrained weights")

# Import and run the test_train.py script
with open('test_train.py', 'r') as f:
    script = f.read()
    
# Execute the script with modifications
# Replace the model creation with non-pretrained version
script = script.replace('pretrained=True', 'pretrained=False')

# Execute the modified script
exec(script, {'__name__': '__main__', 'CFG': __import__('test_train').CFG}) 