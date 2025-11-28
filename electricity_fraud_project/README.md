# نظام كشف سرقة الكهرباء بالذكاء الاصطناعي
# Electricity Fraud Detection System - AI Powered

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Flask-3.0-green.svg" alt="Flask">
  <img src="https://img.shields.io/badge/ML-Scikit--Learn-orange.svg" alt="ML">
  <img src="https://img.shields.io/badge/AI-Hugging%20Face-yellow.svg" alt="AI">
  <img src="https://img.shields.io/badge/License-MIT-red.svg" alt="License">
</p>

---

## نظرة عامة | Overview

نظام متكامل لكشف سرقة الكهرباء في مصر باستخدام تقنيات الذكاء الاصطناعي وتعلم الآلة. يهدف المشروع إلى مساعدة شركات توزيع الكهرباء في اكتشاف حالات السرقة وتقليل الفاقد الفني والتجاري.

An integrated system for detecting electricity theft in Egypt using AI and Machine Learning technologies. The project aims to help electricity distribution companies detect fraud cases and reduce technical and commercial losses.

---

## المشكلة | Problem Statement

تعاني مصر من خسائر كبيرة في قطاع الكهرباء بسبب:
- سرقة الكهرباء عبر التلاعب بالعدادات
- التوصيلات غير الشرعية
- التهرب من دفع الفواتير
- الفاقد الفني في الشبكات

**الخسائر المقدرة:** مليارات الجنيهات سنوياً

---

## الحل المقترح | Proposed Solution

نظام ذكي يستخدم:
- **Machine Learning Models:** للتنبؤ باحتمالية السرقة
- **AI Analysis:** لتحليل الأنماط وتوليد التوصيات
- **Geographic Analysis:** لتحديد المناطق عالية الخطورة
- **Real-time Alerts:** للتنبيه الفوري عن الحالات المشبوهة

---

## المميزات | Features

### 1. نماذج تعلم آلي متعددة
- **Logistic Regression:** للتصنيف الخطي
- **Random Forest:** للتعامل مع البيانات المعقدة
- **XGBoost:** للأداء العالي

### 2. تحليل بالذكاء الاصطناعي
- تنبيهات ذكية مولدة بـ AI
- تحليل الرسوم البيانية
- توصيات عملية مخصصة

### 3. لوحة تحكم تفاعلية
- إحصائيات لحظية
- رسوم بيانية متعددة
- تقارير قابلة للتصدير

### 4. كشف العملاء عاليي الخطورة
- تصنيف حسب درجة الخطورة
- قائمة بالعملاء المطلوب فحصهم
- عوامل الخطر لكل عميل

---

## التقنيات المستخدمة | Tech Stack

| Category | Technology |
|----------|------------|
| Backend | Python, Flask |
| ML | Scikit-Learn, XGBoost |
| AI | Hugging Face (LLaMA 3.2) |
| Data | Pandas, NumPy |
| Frontend | HTML5, CSS3, JavaScript |
| Charts | Plotly.js |
| Icons | Font Awesome |

---

## هيكل المشروع | Project Structure

```
electricity_fraud_detection/
├── app.py                 # Flask Application
├── requirements.txt       # Python Dependencies
├── .env                   # Environment Variables (API Keys)
├── templates/
│   └── index.html        # Frontend Interface
├── uploads/              # Uploaded Data Files
└── README.md             # Documentation
```

---

## التثبيت والتشغيل | Installation

### 1. استنساخ المشروع
```bash
git clone https://github.com/username/electricity-fraud-detection.git
cd electricity-fraud-detection
```

### 2. إنشاء بيئة افتراضية
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. تثبيت المتطلبات
```bash
pip install -r requirements.txt
```

### 4. إعداد ملف البيئة
```bash
cp .env.example .env
# Edit .env and add your Hugging Face API key
```

### 5. تشغيل التطبيق
```bash
python app.py
```

### 6. فتح المتصفح
```
http://localhost:5000
```

---

## صيغة البيانات | Data Format

يجب أن يحتوي ملف CSV على الأعمدة التالية:

| Column | Description | Type |
|--------|-------------|------|
| customer_id | رقم العميل | String |
| name | اسم العميل | String |
| governorate | المحافظة | String |
| tenure_months | فترة الاشتراك | Integer |
| contract_type | نوع العقد | String |
| meter_type | نوع العداد | String |
| monthly_charges | الفاتورة الشهرية | Float |
| consumption_kwh | الاستهلاك | Float |
| payment_arrears | المتأخرات (شهور) | Integer |
| complaints_count | عدد الشكاوى | Integer |
| meter_reading_issues | مشاكل العداد | Integer |
| fraud_flag | علامة السرقة (0/1) | Integer |

---

## مقاييس الأداء | Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 85% | 82% | 78% | 80% | 88% |
| Random Forest | 92% | 89% | 87% | 88% | 94% |
| XGBoost | 94% | 91% | 90% | 90% | 96% |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload` | POST | رفع ملف البيانات |
| `/api/stats` | GET | الإحصائيات العامة |
| `/api/alerts` | GET | التنبيهات الذكية |
| `/api/model-comparison` | GET | مقارنة النماذج |
| `/api/predict` | POST | فحص مشترك جديد |
| `/api/high-risk-customers` | GET | العملاء عاليو الخطورة |
| `/api/geographic` | GET | التحليل الجغرافي |
| `/api/export-report` | GET | تصدير التقرير |

---

## القيمة المضافة لمصر | Value for Egypt

1. **تقليل الخسائر المالية:** كشف السرقات يوفر مليارات الجنيهات
2. **تحسين جودة الخدمة:** توزيع عادل للكهرباء
3. **دعم التحول الرقمي:** تطبيق تقنيات AI في القطاع الحكومي
4. **توفير الطاقة:** تقليل الفاقد يعني توفير الوقود
5. **العدالة الاجتماعية:** منع التهرب يحمي المشتركين الملتزمين

---

## التطوير المستقبلي | Future Development

- [ ] تطبيق موبايل للمفتشين
- [ ] تكامل مع أنظمة العدادات الذكية
- [ ] نموذج Deep Learning محسّن
- [ ] خريطة تفاعلية للمناطق
- [ ] نظام إنذار مبكر بالرسائل

---

## المساهمون | Contributors

- **[اسم الطالب]** - Team Leader / ML Developer
- **[اسم الطالب]** - Backend Developer
- **[اسم الطالب]** - Frontend Developer
- **[اسم الطالب]** - Data Analyst

---

## الرخصة | License

This project is licensed under the MIT License.

---

## شكر وتقدير | Acknowledgments

- مبادرة رواد مصر الرقمية (DEPI)
- وزارة الاتصالات وتكنولوجيا المعلومات
- Hugging Face للـ AI Models

---

<p align="center">
  <b>صنع بـ ❤️ في مصر</b><br>
  Digital Egypt Pioneers Initiative - Round 3
</p>
