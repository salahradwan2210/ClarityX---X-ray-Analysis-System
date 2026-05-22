# Offline Chest X-Ray Diagnosis System

This system allows you to diagnose chest X-ray images without requiring an internet connection. The solution is designed to work completely offline, avoiding any SSL errors or download attempts.

## Quick Start

The easiest way to run the system is using the batch file:

```
run_offline.bat path/to/your/xray/image.png
```

This will:
1. Set up the necessary environment variables
2. Run the diagnosis script in offline mode
3. Display the results with attention map visualization

## Additional Options

You can also run the script directly with more options:

```
python offline_diagnose.py path/to/your/xray/image.png --threshold 0.45 --debug --save output.png
```

Options:
- `--threshold <value>`: Set the threshold for positive predictions (default: 0.45)
- `--checkpoint <path>`: Specify a different model checkpoint file
- `--debug`: Display detailed debugging information
- `--save <path>`: Save the visualization to a file instead of displaying it

## How It Works

The system:
1. Uses a modified ConvNext Large model architecture
2. Loads the pretrained weights from a local checkpoint file
3. Prevents any internet access attempts by setting environment variables
4. Processes your X-ray image and provides a diagnosis
5. Generates an attention map showing which areas of the image the model focused on

## Troubleshooting

If you encounter errors:

1. **Missing checkpoint file**: Ensure `best_model_epoch_9_auroc_0.9692.pth` is in the same directory as the script, or specify a different checkpoint with `--checkpoint`

2. **Module errors**: Make sure you have all required dependencies installed:
   ```
   pip install torch torchvision timm albumentations opencv-python pillow matplotlib numpy pandas
   ```

3. **Image loading errors**: Verify that your image path is correct and the image is a valid X-ray in a supported format (PNG, JPG, etc.)

4. **Memory issues**: If you're running out of memory, try using a CPU by setting:
   ```
   set CUDA_VISIBLE_DEVICES=-1
   ```
   before running the script

## Extending the System

To use this with different models or for different tasks:

1. Modify the `CFG` class parameters in `offline_diagnose.py`
2. Replace the checkpoint file with your own trained model
3. Update the `CLASS_NAMES` list if needed for different labels 