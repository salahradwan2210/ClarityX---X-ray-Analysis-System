# Advanced Chest X-ray Classification and Localization

This project implements an advanced deep learning model for chest X-ray classification and localization, capable of detecting 14 different thoracic diseases and providing bounding box localization for 8 of these conditions.

## Model Architecture

The model uses a ConvNext Large backbone with several enhancements:
- Attention mechanism for better feature focus
- Enhanced localization head for accurate bounding box prediction
- Metadata integration (patient age, gender, view position)
- Advanced training techniques (focal loss, mixup augmentation)

## Performance

The model achieves a mean AUROC of 0.9688 across 14 disease classes, with high localization accuracy.

## Prerequisites

- Python 3.8+
- PyTorch 1.10+
- CUDA-capable GPU (recommended)
- Required packages: torch, torchvision, timm, albumentations, opencv-python, scikit-learn, matplotlib, pandas, numpy

Install dependencies:
```
pip install torch torchvision timm albumentations opencv-python scikit-learn matplotlib pandas numpy
```

## Training

### Continue Training

To continue training from a checkpoint with enhanced localization focus:

```
python continue_training.py
```

This script will:
1. Load the best model checkpoint
2. Apply enhanced architecture with focus on localization
3. Continue training with optimized hyperparameters
4. Save checkpoints and performance visualizations

### Training Configuration

Key parameters in `test_train.py`:
- `IMG_SIZE`: Input image resolution (512x512)
- `BATCH_SIZE`: Batch size (4)
- `ACCUM_STEPS`: Gradient accumulation steps (8)
- `LR`: Learning rate for backbone (5e-6)
- `HEAD_LR`: Learning rate for classification head (1e-5)
- `BBOX_LOSS_WEIGHT`: Weight for bounding box loss (1.0)

## Inference

### Single Image Diagnosis

To diagnose a single chest X-ray image:

```
python diagnose_image.py --image path/to/image.jpg --output diagnosis_result.png
```

Options:
- `--image`: Path to the X-ray image (required)
- `--model`: Path to the model weights (default: fine_tuned_model_best_auroc_0.9688.pth)
- `--output`: Path to save the output image with visualizations
- `--no-display`: Do not display the image (useful for batch processing)

### Batch Processing

To process multiple images in a directory:

```
python batch_diagnose.py --input path/to/images/ --output results/
```

Options:
- `--input`: Directory containing images to process (required)
- `--output`: Directory to save results (default: diagnosis_results)
- `--model`: Path to the model weights

## Output

The diagnosis output includes:
- Disease classification with confidence scores
- Bounding box localization for supported conditions
- Visual overlay of findings
- CSV reports with detailed results

## Dataset

This model was trained on the NIH Chest X-ray dataset, which contains 112,120 X-ray images from 30,805 unique patients.

## References

- Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. (2017). ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2097-2106).

## License

This project is licensed under the MIT License - see the LICENSE file for details. 