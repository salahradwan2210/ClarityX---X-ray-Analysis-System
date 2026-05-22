# تشغيل نموذج الذكاء الاصطناعي الحقيقي للأشعة السينية للصدر

هذا الدليل يشرح كيفية تشغيل نموذج الذكاء الاصطناعي الحقيقي لتحليل الأشعة السينية للصدر في المشروع. النموذج هو ConvNeXt Large مدرب على مجموعة بيانات CheXpert.

## المتطلبات الأساسية

1. بيئة Python 3.8 أو أحدث
2. وحدة معالجة رسومات (GPU) موصى بها
3. ملف النموذج: `best_model_epoch_3_auroc_9004.pth`

## خطوات التثبيت والتشغيل

### الخطوة 1: إعداد البيئة

```bash
# إنشاء وتفعيل بيئة Python افتراضية
python -m venv venv
.\venv\Scripts\activate  # على Windows
source venv/bin/activate  # على Linux/Mac

# تثبيت المكتبات المطلوبة
python -m pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org
pip install fastapi uvicorn torch torchvision timm numpy pillow python-multipart --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

### الخطوة 2: وضع ملف النموذج في المكان الصحيح

تأكد من وضع ملف النموذج `best_model_epoch_3_auroc_9004.pth` في نفس المجلد مع `model_server.py`.

### الخطوة 3: تشغيل خادم النموذج

```bash
# انتقل إلى مجلد python-backend
cd python-backend

# تشغيل خادم النموذج
python model_server.py
```

سيبدأ خادم FastAPI على المنفذ 5000. ينبغي أن ترى رسالة مشابهة لهذه:

```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:5000 (Press CTRL+C to quit)
```

### الخطوة 4: اختبار الخادم

يمكنك التحقق من أن الخادم يعمل بشكل صحيح من خلال زيارة:

```
http://localhost:5000/healthcheck
```

يجب أن ترى استجابة JSON تشير إلى أن النموذج تم تحميله بنجاح.

## استكشاف الأخطاء وإصلاحها

### مشكلة في تحميل النموذج

إذا واجهت أخطاء متعلقة بـ PyTorch 2.6 والمتغيرات العالمية الآمنة، تأكد من أن ملف `model_server.py` يحتوي على الكود التالي في بدايته:

```python
# Explicitly add safe globals for torch loading if using PyTorch 2.6+
try:
    from torch.serialization import add_safe_globals
    import numpy.core.multiarray
    add_safe_globals([numpy.core.multiarray.scalar])
    print("Added safe globals for PyTorch 2.6+")
except (ImportError, AttributeError):
    print("Running on older PyTorch version, skipping safe globals configuration")
```

### مشاكل SSL عند تثبيت المكتبات

إذا واجهت مشاكل SSL عند تثبيت المكتبات، استخدم الأعلام `--trusted-host` كما هو موضح في أوامر التثبيت أعلاه.

## كيفية عمل النظام بالكامل

1. قم بتشغيل خادم النموذج باستخدام الأوامر أعلاه
2. قم بتشغيل تطبيق Next.js باستخدام `npm run dev`
3. عند تحليل صورة أشعة سينية، سيرسل التطبيق طلبًا إلى خادم النموذج على المنفذ 5000
4. سيعالج خادم النموذج الصورة ويعيد التنبؤات
5. سيعرض التطبيق النتائج بما في ذلك المناطق المحددة في الصورة

## ملاحظات مهمة

- في حالة عدم توفر GPU، سيستخدم النموذج وحدة المعالجة المركزية (CPU) ولكن الأداء سيكون أبطأ.
- تأكد من أن خادم النموذج يعمل قبل استخدام التطبيق.
- إذا فشل تحميل النموذج، سيستخدم الخادم منطق احتياطي لإنشاء تنبؤات وهمية للأغراض التجريبية.

## Model Architecture

The AI model is built using the ConvNeXt Large architecture with several enhancements:

1. **Base Model**: ConvNeXt Large pre-trained on ImageNet
2. **Attention Mechanism**: A self-attention layer to focus on relevant image regions
3. **Localization Head**: A branch that predicts bounding boxes for detected conditions
4. **Metadata Integration**: Patient metadata (age, sex, view position) is incorporated through a dedicated branch
5. **Combined Classifier**: The final layer combines image features and metadata to predict 14 chest conditions

## API Reference

### POST /predict

Analyze a chest X-ray image and return detected conditions with probabilities and bounding boxes.

**Request:**
- `image`: X-ray image file (JPG, PNG)
- `age`: Patient age (integer)
- `sex`: Patient sex (0 for female, 1 for male)
- `view_position`: View position (0 for PA, 1 for AP)

**Response:**
```json
{
  "detections": {
    "atelectasis": 0.2,
    "cardiomegaly": 0.3,
    "effusion": 0.6,
    "infiltration": 0.2,
    "mass": 0.1,
    "nodule": 0.1,
    "pneumonia": 0.1,
    "pneumothorax": 0.1,
    "consolidation": 0.2,
    "edema": 0.1,
    "emphysema": 0.1,
    "fibrosis": 0.8,
    "pleural_thickening": 0.2,
    "hernia": 0.1,
    "no_finding": 0.1
  },
  "boxes": {
    "fibrosis": {
      "x": 0.6,
      "y": 0.3,
      "width": 0.2,
      "height": 0.3
    },
    "effusion": {
      "x": 0.3,
      "y": 0.4,
      "width": 0.25,
      "height": 0.2
    }
  }
}
```

## Deployment Considerations

For production deployment:

1. **Security**: Implement proper authentication and HTTPS
2. **Scaling**: Consider deploying the model server with multiple workers
3. **Resource Management**: Optimize batch size and model loading based on available resources
4. **Monitoring**: Add logging and monitoring to track model performance and errors

## Troubleshooting

- **CUDA Out of Memory**: Reduce the batch size or use a smaller model
- **Model Not Found**: Ensure the model file path is correct
- **CORS Issues**: Configure CORS in the FastAPI app to allow your frontend domain
- **Slow Predictions**: Consider model optimization techniques like quantization or using a GPU 