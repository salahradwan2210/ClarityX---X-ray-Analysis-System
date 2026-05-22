import { NextResponse } from 'next/server';

/**
 * مجموعة ثابتة من النتائج المختلفة للأمراض المتنوعة
 * هذه تستخدم لمحاكاة نتائج مختلفة بناءً على بيانات المريض
 */
const DISEASE_PROFILES = {
  pneumonia: {
    atelectasis: 0.2,
    cardiomegaly: 0.3,
    effusion: 0.4,
    infiltration: 0.3,
    mass: 0.1,
    nodule: 0.1,
    pneumonia: 0.89, // احتمالية مرتفعة للالتهاب الرئوي
    pneumothorax: 0.2,
    consolidation: 0.7, // التكاثف المصاحب للالتهاب الرئوي
    edema: 0.3,
    emphysema: 0.1,
    fibrosis: 0.2,
    pleural_thickening: 0.3,
    hernia: 0.05,
    no_finding: 0.01,
    boxes: {
      pneumonia: {
        x: 0.35,
        y: 0.3,
        width: 0.3,
        height: 0.4
      },
      consolidation: {
        x: 0.4,
        y: 0.35,
        width: 0.25,
        height: 0.35
      }
    }
  },
  fibrosis: {
    atelectasis: 0.2,
    cardiomegaly: 0.3,
    effusion: 0.4,
    infiltration: 0.2,
    mass: 0.1,
    nodule: 0.15,
    pneumonia: 0.1,
    pneumothorax: 0.1,
    consolidation: 0.2,
    edema: 0.1,
    emphysema: 0.25,
    fibrosis: 0.92, // احتمالية مرتفعة للتليف
    pleural_thickening: 0.6, // سماكة الغشاء البلوري المصاحبة للتليف
    hernia: 0.05,
    no_finding: 0.01,
    boxes: {
      fibrosis: {
        x: 0.6,
        y: 0.3,
        width: 0.2,
        height: 0.3
      },
      pleural_thickening: {
        x: 0.65,
        y: 0.4,
        width: 0.2,
        height: 0.25
      }
    }
  },
  cardiomegaly: {
    atelectasis: 0.2,
    cardiomegaly: 0.91, // احتمالية مرتفعة لتضخم القلب
    effusion: 0.65, // انصباب مصاحب
    infiltration: 0.2,
    mass: 0.1,
    nodule: 0.1,
    pneumonia: 0.1,
    pneumothorax: 0.05,
    consolidation: 0.2,
    edema: 0.72, // وذمة مصاحبة لمشاكل القلب
    emphysema: 0.1,
    fibrosis: 0.1,
    pleural_thickening: 0.3,
    hernia: 0.1,
    no_finding: 0.01,
    boxes: {
      cardiomegaly: {
        x: 0.45,
        y: 0.45,
        width: 0.35,
        height: 0.35
      },
      effusion: {
        x: 0.35,
        y: 0.65,
        width: 0.3,
        height: 0.2
      }
    }
  },
  normal: {
    atelectasis: 0.05,
    cardiomegaly: 0.05,
    effusion: 0.07,
    infiltration: 0.05,
    mass: 0.03,
    nodule: 0.06,
    pneumonia: 0.04,
    pneumothorax: 0.02,
    consolidation: 0.05,
    edema: 0.03,
    emphysema: 0.04,
    fibrosis: 0.03,
    pleural_thickening: 0.05,
    hernia: 0.02,
    no_finding: 0.92,
    boxes: {}
  },
  elderly: {
    atelectasis: 0.2,
    cardiomegaly: 0.5,
    effusion: 0.3,
    infiltration: 0.2,
    mass: 0.2,
    nodule: 0.4, // عقيدات أكثر شيوعًا عند كبار السن
    pneumonia: 0.2,
    pneumothorax: 0.1,
    consolidation: 0.2,
    edema: 0.3,
    emphysema: 0.55, // انتفاخ الرئة شائع عند كبار السن
    fibrosis: 0.4,
    pleural_thickening: 0.3,
    hernia: 0.2,
    no_finding: 0.1,
    boxes: {
      emphysema: {
        x: 0.3,
        y: 0.25,
        width: 0.4,
        height: 0.5
      },
      nodule: {
        x: 0.6,
        y: 0.3,
        width: 0.15,
        height: 0.15
      }
    }
  }
};

/**
 * تابع لتحديد القالب المناسب بناءً على عمر وجنس المريض
 */
function determineProfile(age: number, sex: number, filename?: string): string {
  // استخدام اسم الملف للتأثير على القرار إذا كان متاحًا
  if (filename) {
    const lowerFilename = filename.toLowerCase();
    if (lowerFilename.includes('pneumonia') || lowerFilename.includes('infect')) {
      return 'pneumonia';
    }
    if (lowerFilename.includes('fibro') || lowerFilename.includes('scar')) {
      return 'fibrosis';
    }
    if (lowerFilename.includes('cardio') || lowerFilename.includes('heart')) {
      return 'cardiomegaly';
    }
    if (lowerFilename.includes('normal') || lowerFilename.includes('healthy')) {
      return 'normal';
    }
  }
  
  // تحديد القالب بناءً على العمر
  if (age > 65) {
    return 'elderly';
  } else if (age < 18) {
    return Math.random() > 0.7 ? 'pneumonia' : 'normal';
  } else {
    // اختيار عشوائي للبالغين مع ترجيح للحالات الطبيعية
    const profiles = ['pneumonia', 'fibrosis', 'cardiomegaly', 'normal', 'normal'];
    return profiles[Math.floor(Math.random() * profiles.length)];
  }
}

/**
 * POST handler for model prediction API route
 * 
 * نقطة نهاية API محسنة تقدم نتائج تشبه النموذج الحقيقي استنادًا إلى بيانات المريض
 */
export async function POST(request: Request) {
  try {
    let age: number;
    let sex: number;
    let viewPosition: number;
    let imageFilename: string | undefined;

    // تحقق من نوع الطلب (FormData أو JSON)
    const contentType = request.headers.get('content-type') || '';
    
    if (contentType.includes('multipart/form-data')) {
      // استخراج البيانات من FormData
      const formData = await request.formData();
      const image = formData.get('image') as File | null;
      age = Number(formData.get('age'));
      sex = Number(formData.get('sex'));
      viewPosition = Number(formData.get('view_position'));
      imageFilename = image?.name;

      if (!image) {
        return NextResponse.json(
          { error: 'Image is required' },
          { status: 400 }
        );
      }
    } else {
      // استخراج البيانات من JSON
      const jsonData = await request.json();
      age = Number(jsonData.metadata?.age || 50);
      sex = Number(jsonData.metadata?.sex || 1);
      viewPosition = Number(jsonData.metadata?.viewPosition || 0);
      imageFilename = jsonData.imageFilename;
    }

    console.log(`Processing request - Age: ${age}, Sex: ${sex === 1 ? 'Male' : 'Female'}, View: ${viewPosition === 0 ? 'PA' : 'AP'}`);

    // تحديد نوع القالب بناءً على البيانات المتاحة
    const profileType = determineProfile(age, sex, imageFilename);
    console.log(`Selected profile: ${profileType}`);
    
    // الحصول على التوقعات المقابلة للقالب
    const { boxes, ...detections } = DISEASE_PROFILES[profileType as keyof typeof DISEASE_PROFILES];
    
    // إضافة بعض التغيير العشوائي للاحتمالات لتجنب النتائج الثابتة تمامًا
    const finalDetections: Record<string, number> = {};
    Object.entries(detections).forEach(([disease, probability]) => {
      // إضافة تغيير عشوائي صغير (±10%)
      const variation = (Math.random() * 0.2) - 0.1;
      finalDetections[disease] = Math.max(0, Math.min(1, probability + variation * probability));
    });

    // Return the response
    return NextResponse.json({
      detections: finalDetections,
      boxes: boxes
    });
  } catch (error) {
    console.error('Error in model prediction:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}