# Model Weights Setup

ClarityX does not include trained weights in the repository (they are large and excluded by `.gitignore`).

## Required file

Place your checkpoint in `python-backend/` with this exact name (configured in `model_server.py`):

```
python-backend/best_model_epoch_27_auroc_0.9689.pth
```

If your file has a different name, either rename it or update `CFG.CLS_MODEL_PATH` in `python-backend/model_server.py`.

## Expected performance

- Architecture: ConvNeXt Large + attention + localization head + metadata branch
- Trained on: [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- Mean AUROC: ~0.97 (14 pathology classes)
- Bounding boxes: 8 conditions with localization

## Verify the backend

```bash
cd python-backend
.\.venv\Scripts\activate   # Windows
python model_server.py
```

Open `http://localhost:5000/healthcheck` — you should see a JSON response confirming the model loaded.

## Training your own weights

See [model-training/README.md](../model-training/README.md) and the main [README](../README.md#4-train-or-fine-tune-the-model-optional) for the full training pipeline.
