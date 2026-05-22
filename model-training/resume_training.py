import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time

# استيراد الدوال من ملفات المشروع
from advanced_model import (
    set_seed, CLASS_NAMES, TRAIN_CLASSES, AdvancedXrayDataset, 
    mixup_data, mixup_criterion, AdvancedXrayModel, advanced_loss
)
from train_advanced import (
    preprocess_metadata, load_image_list, analyze_dataset, 
    plot_training_progress, plot_class_metrics, WarmupCosineScheduler,
    evaluate_model
)

def main():
    # تعيين بذرة عشوائية للتكرار
    set_seed(42)
    
    # تهيئة الجهاز
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"استخدام جهاز: {device}")
    
    if torch.cuda.is_available():
        print(f"نوع GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    # مسارات البيانات
    base_path = 'data'
    resized_base_path = 'data_resized_512'
    image_dir = resized_base_path
    data_entry_path = os.path.join(base_path, 'Data_Entry_2017.csv')
    train_val_list_path = os.path.join(base_path, 'train_val_list.txt')
    
    # مسار الشيكبوينت والمخرجات
    output_dir = "advanced_outputs"
    checkpoint_path = os.path.join(output_dir, "checkpoint_latest.pth")
    
    # التحقق من وجود الشيكبوينت
    if not os.path.exists(checkpoint_path):
        print(f"خطأ: لم يتم العثور على الشيكبوينت في {checkpoint_path}")
        return
    
    print(f"تحميل الشيكبوينت من {checkpoint_path}")
    
    # تحميل الشيكبوينت
    try:
        checkpoint = torch.load(checkpoint_path)
        
        # طباعة المعلومات
        start_epoch = checkpoint['epoch'] + 1
        best_auroc = checkpoint['best_auroc']
        print(f"تم تحميل الشيكبوينت بنجاح - سيتم البدء من الإيبوك {start_epoch}")
        print(f"أفضل AUROC سابق: {best_auroc:.4f}")
    except Exception as e:
        print(f"خطأ في تحميل الشيكبوينت: {e}")
        return
    
    # تحميل البيانات
    print("تحميل وتجهيز البيانات...")
    valid_images = set(load_image_list(train_val_list_path))
    print(f"تم العثور على {len(valid_images)} صورة في ملف القائمة")
    
    # تحميل بيانات CSV
    df = pd.read_csv(data_entry_path)
    df = df[df['Image Index'].isin(valid_images)]
    df = df[df['Finding Labels'] != 'No Finding']
    print(f"إجمالي الصور القابلة للاستخدام: {len(df)}")
    
    # معالجة البيانات الوصفية
    df, gender_encoder, view_encoder = preprocess_metadata(df)
    
    # تقسيم إلى تدريب/اختبار
    train_df = df.sample(frac=0.8, random_state=42)
    valid_df = df.drop(train_df.index)
    
    # تهيئة تحويلات الصور
    image_size = 512
    train_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Rotate(limit=20, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.5),
        ], p=0.5),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    valid_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # إنشاء مجموعات البيانات
    print("إنشاء مجموعات البيانات والمحملات...")
    train_dataset = AdvancedXrayDataset(image_dir, train_df, transform=train_transform, train=True)
    valid_dataset = AdvancedXrayDataset(image_dir, valid_df, transform=valid_transform, train=False)
    
    # إنشاء محملات البيانات
    batch_size = 8
    num_workers = 2
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # إنشاء النموذج
    print("تهيئة النموذج...")
    model = AdvancedXrayModel(
        num_classes=len(CLASS_NAMES), 
        metadata_features=3, 
        dropout_rate=0.4
    )
    model = model.to(device, memory_format=torch.channels_last)
    
    # تحميل أوزان النموذج
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # المحسن ومجدول معدل التعلم
    # استخدام معدل تعلم أقل للاستئناف
    base_lr = 5e-6  # معدل تعلم أقل للاستئناف
    head_lr = base_lr * 3
    
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'model' in n], 'lr': base_lr},
        {'params': [p for n, p in model.named_parameters() if 'model' not in n], 'lr': head_lr}
    ], weight_decay=0.01)
    
    # تحميل حالة المحسن
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # تحديث معدلات التعلم
    for i, param_group in enumerate(optimizer.param_groups):
        if i == 0:  # طبقات القاعدة
            param_group['lr'] = base_lr
        else:  # طبقات الرأس
            param_group['lr'] = head_lr
    
    # استعادة مقاييس التدريب
    metrics = checkpoint['metrics']
    
    # إنشاء مجدول معدل التعلم
    epochs = 30
    scheduler = WarmupCosineScheduler(
        optimizer, 
        warmup_epochs=2, 
        total_epochs=epochs,
        min_lr=1e-7,
        warmup_start_lr=1e-6
    )
    
    # تحديث حالة المجدول
    for _ in range(start_epoch):
        scheduler.step()
    
    # إعداد مقياس للتدريب بدقة مختلطة
    scaler = torch.cuda.amp.GradScaler()
    
    # متغيرات التتبع للإيقاف المبكر
    best_auroc = checkpoint['best_auroc']
    epochs_no_improve = 0
    patience = 7
    
    print("\n" + "="*50)
    print(f"استئناف التدريب من الإيبوك {start_epoch+1} إلى الإيبوك {epochs}")
    print("="*50)
    
    # حلقة التدريب
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        
        print(f"الإيبوك {epoch+1}/{epochs} - LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # مرحلة التدريب
        for step, (images, labels, metadata) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)
            
            # تطبيق mixup باحتمالية
            use_mixup = (np.random.random() < 0.7) and (0.5 > 0)
            
            # تصفير تدرجات المعلمات
            optimizer.zero_grad(set_to_none=True)
            
            # التمرير الأمامي مع الدقة المختلطة
            with torch.cuda.amp.autocast():
                if use_mixup:
                    # تطبيق mixup
                    mixed_images, mixed_metadata, labels_a, labels_b, lam = mixup_data(
                        images, labels, metadata, alpha=0.5
                    )
                    
                    # التمرير الأمامي مع المدخلات المختلطة
                    cls_out, combined_out = model(mixed_images, mixed_metadata)
                    
                    # حساب الخسارة باستخدام معيار mixup
                    loss = lam * advanced_loss(cls_out, combined_out, labels_a, device, 0.1) + \
                           (1 - lam) * advanced_loss(cls_out, combined_out, labels_b, device, 0.1)
                else:
                    # التمرير الأمامي العادي
                    cls_out, combined_out = model(images, metadata)
                    
                    # حساب الخسارة العادي
                    loss = advanced_loss(cls_out, combined_out, labels, device, 0.1)
            
            # التمرير الخلفي مع قياس التدرج
            scaler.scale(loss).backward()
            
            # قطع التدرج
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # تحديث الأوزان
            scaler.step(optimizer)
            scaler.update()
            
            # تحديث المقاييس
            train_loss += loss.item()
            
            # إخراج التقدم
            if (step + 1) % 20 == 0:
                print(f"  الخطوة {step+1}/{len(train_loader)} - الخسارة: {loss.item():.4f}")
            
            # تنظيف الذاكرة
            del images, labels, metadata, cls_out, combined_out, loss
            torch.cuda.empty_cache()
        
        # حساب متوسط خسارة التدريب
        avg_train_loss = train_loss / len(train_loader)
        metrics['train_loss'].append(avg_train_loss)
        metrics['lr'].append(optimizer.param_groups[0]['lr'])
        
        # مرحلة التحقق
        model.eval()
        val_loss = 0.0
        all_targets = []
        all_outputs = []
        
        print("التحقق من النموذج...")
        with torch.no_grad(), torch.cuda.amp.autocast():
            for images, labels, metadata in valid_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                metadata = metadata.to(device, non_blocking=True)
                
                # التمرير الأمامي
                cls_out, combined_out = model(images, metadata)
                
                # حساب خسارة التحقق
                loss = advanced_loss(cls_out, combined_out, labels, device, 0.1)
                val_loss += loss.item()
                
                # تخزين المخرجات لحساب المقاييس
                all_targets.append(labels.cpu())
                all_outputs.append(torch.sigmoid(combined_out).cpu())  # استخدام sigmoid للاحتمالية
                
                # تنظيف الذاكرة
                del images, labels, metadata, cls_out, combined_out, loss
                torch.cuda.empty_cache()
        
        # حساب متوسط خسارة التحقق
        avg_val_loss = val_loss / len(valid_loader)
        metrics['val_loss'].append(avg_val_loss)
        
        # تجميع كل المخرجات والأهداف
        all_targets = torch.cat(all_targets, dim=0)
        all_outputs = torch.cat(all_outputs, dim=0)
        
        # حساب AUROC لكل فئة
        class_aurocs = {}
        for i, cls_name in enumerate(TRAIN_CLASSES):
            if len(torch.unique(all_targets[:, i])) > 1:  # التأكد من وجود عينات إيجابية وسلبية
                auroc = roc_auc_score(all_targets[:, i].numpy(), all_outputs[:, i].numpy())
                class_aurocs[cls_name] = auroc
                print(f"{cls_name}: AUROC = {auroc:.4f}")
            else:
                print(f"تخطي {cls_name} - لا يوجد ما يكفي من التسميات الفريدة")
                class_aurocs[cls_name] = 0.0
        
        # حساب متوسط AUROC
        mean_auroc = np.mean(list(class_aurocs.values()))
        metrics['mean_auroc'].append(mean_auroc)
        
        # طباعة نتائج الإيبوك
        print(f"\nالإيبوك {epoch+1}/{epochs} - النتائج:")
        print(f"خسارة التدريب: {avg_train_loss:.4f}")
        print(f"خسارة التحقق: {avg_val_loss:.4f}")
        print(f"متوسط AUROC: {mean_auroc:.4f}")
        print(f"معدل التعلم الحالي: {optimizer.param_groups[0]['lr']:.6f}")
        
        # تحديث مجدول معدل التعلم
        scheduler.step()
        
        # رسم وحفظ المقاييس
        plot_training_progress(metrics, save_path=os.path.join(output_dir, f"training_progress_epoch_{epoch+1}.png"))
        plot_class_metrics(class_aurocs, save_path=os.path.join(output_dir, f"class_metrics_epoch_{epoch+1}.png"))
        
        # حفظ الشيكبوينت
        checkpoint_path = os.path.join(output_dir, "checkpoint_latest.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.__dict__,
            'metrics': metrics,
            'best_auroc': best_auroc
        }, checkpoint_path)
        
        # التحقق من أفضل نموذج
        if mean_auroc > best_auroc:
            best_auroc = mean_auroc
            best_model_path = os.path.join(output_dir, f"best_model_auroc_{best_auroc:.4f}.pth")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'class_aurocs': class_aurocs,
                'mean_auroc': mean_auroc
            }, best_model_path)
            
            print(f"✅ تم حفظ نموذج جديد أفضل! متوسط AUROC: {best_auroc:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"📉 لا يوجد تحسن لمدة {epochs_no_improve} إيبوك. أفضل AUROC: {best_auroc:.4f}")
        
        # التحقق من الإيقاف المبكر
        if epochs_no_improve >= patience:
            print(f"🛑 تم تفعيل الإيقاف المبكر بعد الإيبوك {epoch+1}.")
            break
    
    print("\n" + "="*50)
    print(f"اكتمل التدريب. أفضل متوسط AUROC: {best_auroc:.4f}")
    print("="*50)

if __name__ == "__main__":
    main() 