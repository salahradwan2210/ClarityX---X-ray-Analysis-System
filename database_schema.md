# هيكل قاعدة البيانات (Database Schema)

## 1. جدول المستخدمين (users)
| العمود | النوع | الوصف |
|--------|--------|--------|
| id | uuid | المفتاح الرئيسي |
| email | varchar | البريد الإلكتروني (فريد) |
| full_name | varchar | الاسم الكامل |
| role | varchar | الدور (admin, doctor, radiologist) |
| license_number | varchar | رقم الترخيص (للكادر الطبي) |
| created_at | timestamp | تاريخ الإنشاء |
| updated_at | timestamp | تاريخ التحديث |

## 2. جدول المرضى (patients)
| العمود | النوع | الوصف |
|--------|--------|--------|
| id | uuid | المفتاح الرئيسي |
| full_name | varchar | اسم المريض |
| gender | varchar | الجنس |
| age | integer | العمر |
| phone | varchar | رقم الهاتف |
| created_by | uuid | معرف المستخدم الذي أضاف المريض |
| created_at | timestamp | تاريخ الإضافة |
| updated_at | timestamp | تاريخ التحديث |

## 3. جدول التحليلات (analyses)
| العمود | النوع | الوصف |
|--------|--------|--------|
| id | uuid | المفتاح الرئيسي |
| patient_id | uuid | معرف المريض (مفتاح خارجي) |
| image_url | varchar | رابط صورة الأشعة |
| view_position | varchar | وضع الأشعة (PA, AP, Lateral) |
| status | varchar | حالة التحليل (pending, completed, failed) |
| created_by | uuid | معرف المستخدم الذي أجرى التحليل |
| created_at | timestamp | تاريخ التحليل |
| updated_at | timestamp | تاريخ التحديث |

## 4. جدول النتائج (results)
| العمود | النوع | الوصف |
|--------|--------|--------|
| id | uuid | المفتاح الرئيسي |
| analysis_id | uuid | معرف التحليل (مفتاح خارجي) |
| disease | varchar | اسم المرض |
| probability | float | نسبة الاحتمالية |
| bounding_box | jsonb | إحداثيات مربع تحديد المنطقة (x, y, width, height) |
| created_at | timestamp | تاريخ إضافة النتيجة |
| updated_at | timestamp | تاريخ تحديث النتيجة |

## العلاقات بين الجداول
1. **users -> patients**: علاقة واحد-للكثير (one-to-many)
   - المستخدم يمكنه إضافة عدة مرضى
   - المريض يتبع لمستخدم واحد

2. **users -> analyses**: علاقة واحد-للكثير (one-to-many)
   - المستخدم يمكنه إجراء عدة تحليلات
   - التحليل يتبع لمستخدم واحد

3. **patients -> analyses**: علاقة واحد-للكثير (one-to-many)
   - المريض يمكن أن يكون له عدة تحليلات
   - التحليل يتبع لمريض واحد

4. **analyses -> results**: علاقة واحد-للكثير (one-to-many)
   - التحليل يمكن أن يكون له عدة نتائج
   - النتيجة تتبع لتحليل واحد

## ملاحظات إضافية
- جميع الجداول تستخدم UUID كمعرفات رئيسية
- تم إضافة timestamps (created_at, updated_at) لجميع الجداول للتتبع
- تم استخدام JSONB لتخزين بيانات Bounding Box لمرونة أكبر
- تم إضافة مفاتيح خارجية مع CASCADE DELETE حيثما كان مناسباً 