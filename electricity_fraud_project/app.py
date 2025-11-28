"""
نظام كشف سرقة الكهرباء بالذكاء الاصطناعي
Electricity Fraud Detection System - AI Powered
Version: 4.0.0
Developed for: Digital Egypt Pioneers Initiative (DEPI)
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import os
import json
from huggingface_hub import InferenceClient
import warnings
import time
from datetime import datetime
from functools import lru_cache
from collections import defaultdict
from dotenv import load_dotenv
import io
import base64

# ML Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings('ignore')
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Hugging Face Configuration - من ملف .env
HF_API_KEY = os.getenv('HF_API_KEY', '')
hf_client = InferenceClient(token=HF_API_KEY) if HF_API_KEY else None

# DATA STORE 
data_store = {
    'df': None,
    'df_original': None,
    'stats': {},
    'loaded': False,
    'cache': {},
    'last_update': None,
    'geographic_analysis': {},
    'ml_models': {},
    'model_comparison': {},
    'feature_importance': {},
    'scaler': None,
    'label_encoders': {},
    'ai_alerts': [],
    'class_imbalance_report': {},
    'high_risk_customers': []
}

#  UTILITY FUNCTIONS 

def get_cache(key):
    if key in data_store['cache']:
        cached_data, timestamp = data_store['cache'][key]
        if time.time() - timestamp < 300:
            return cached_data
    return None

def set_cache(key, value):
    data_store['cache'][key] = (value, time.time())

def clear_cache():
    data_store['cache'] = {}

def clean_ai_response(text):
    """تنظيف استجابة الـ AI"""
    chars_to_remove = ['*', '#', '•', '►', '▸', '→', '⁃']
    for char in chars_to_remove:
        text = text.replace(char, '')
    
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

# AI FUNCTIONS 

def analyze_with_ai(prompt, max_tokens=800):
    """تحليل باستخدام Hugging Face AI"""
    if not hf_client:
        raise Exception("لم يتم تكوين مفتاح API للذكاء الاصطناعي")
    
    try:
        messages = [
            {
                "role": "system",
                "content": """أنت محلل بيانات خبير متخصص في قطاع الكهرباء المصري وكشف السرقات.
                
قواعد الكتابة:
- اكتب بالعربية الفصحى البسيطة
- لا تستخدم أي رموز أو إيموجي
- كن مباشراً ومحدداً
- استخدم الأرقام والنسب لدعم تحليلك
- قدم توصيات عملية قابلة للتنفيذ
- راعِ السياق المصري في تحليلاتك"""
            },
            {"role": "user", "content": prompt}
        ]
        
        response = hf_client.chat_completion(
            messages=messages,
            model="meta-llama/Llama-3.2-3B-Instruct",
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9
        )
        
        text = response.choices[0].message.content.strip()
        return clean_ai_response(text)
    
    except Exception as e:
        print(f"AI Error: {e}")
        raise Exception(f"خطأ في الاتصال بالذكاء الاصطناعي: {str(e)}")

# DATA PREPROCESSING 

def preprocess_data(df):
    """معالجة البيانات وتجهيزها للنماذج"""
    df_processed = df.copy()
    
    
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0] if len(df_processed[col].mode()) > 0 else 'Unknown')
    
    
    label_encoders = {}
    cols_to_encode = ['contract_type', 'meter_type', 'governorate']
    
    for col in cols_to_encode:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
    
    data_store['label_encoders'] = label_encoders
    
    
    if 'monthly_charges' in df_processed.columns and 'consumption_kwh' in df_processed.columns:
        df_processed['charge_per_kwh'] = df_processed['monthly_charges'] / (df_processed['consumption_kwh'] + 1)
        df_processed['consumption_charge_ratio'] = df_processed['consumption_kwh'] / (df_processed['monthly_charges'] + 1)
    
    if 'tenure_months' in df_processed.columns and 'consumption_kwh' in df_processed.columns:
        df_processed['consumption_per_tenure'] = df_processed['consumption_kwh'] / (df_processed['tenure_months'] + 1)
    
    if 'payment_arrears' in df_processed.columns and 'complaints_count' in df_processed.columns:
        df_processed['risk_score_basic'] = df_processed['payment_arrears'] * 2 + df_processed['complaints_count']
    
    return df_processed

def generate_class_imbalance_report(df):
    """تقرير عدم توازن الفئات"""
    if 'fraud_flag' not in df.columns:
        return {}
    
    fraud_counts = df['fraud_flag'].value_counts()
    total = len(df)
    
    normal_count = int(fraud_counts.get(0, 0))
    fraud_count = int(fraud_counts.get(1, 0))
    imbalance_ratio = float(round(normal_count / max(fraud_count, 1), 2))
    
    report = {
        'total_samples': int(total),
        'normal_count': normal_count,
        'fraud_count': fraud_count,
        'normal_percentage': float(round(normal_count / total * 100, 2)),
        'fraud_percentage': float(round(fraud_count / total * 100, 2)),
        'imbalance_ratio': imbalance_ratio,
        'is_imbalanced': bool(imbalance_ratio > 2)
    }
    
    
    if report['is_imbalanced']:
        report['recommendations'] = [
            'استخدام SMOTE لزيادة عينات فئة السرقة',
            'استخدام class_weight في النماذج',
            'التركيز على Precision و Recall بدلاً من Accuracy فقط'
        ]
    else:
        report['recommendations'] = ['البيانات متوازنة نسبياً']
    
    return report

# ML MODELS 

def prepare_features(df):
    """تجهيز الـ features للنماذج"""
    feature_cols = [
        'tenure_months', 'monthly_charges', 'consumption_kwh',
        'payment_arrears', 'complaints_count', 'service_calls',
        'avg_payment_delay_days', 'last_payment_days', 'meter_reading_issues'
    ]
    
    
    encoded_cols = [col for col in df.columns if col.endswith('_encoded')]
    feature_cols.extend(encoded_cols)
    
     
    engineered_cols = ['charge_per_kwh', 'consumption_charge_ratio', 'consumption_per_tenure', 'risk_score_basic']
    for col in engineered_cols:
        if col in df.columns:
            feature_cols.append(col)
    
    available_cols = [col for col in feature_cols if col in df.columns]
    
    return available_cols

def train_models(df):
    """تدريب النماذج المتعددة"""
    df_processed = preprocess_data(df)
    feature_cols = prepare_features(df_processed)
    
    X = df_processed[feature_cols].copy()
    y = df_processed['fraud_flag'].copy()
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    data_store['scaler'] = scaler
    
    models = {}
    results = {}
    
    
    print("Training Logistic Regression...")
    lr_params = {'C': [0.1, 1, 10], 'max_iter': [1000]}
    lr = GridSearchCV(
        LogisticRegression(class_weight='balanced', random_state=42),
        lr_params, cv=3, scoring='f1'
    )
    lr.fit(X_train_scaled, y_train_balanced)
    models['logistic_regression'] = lr.best_estimator_
    
    
    print("Training Random Forest...")
    rf_params = {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}
    rf = GridSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=42),
        rf_params, cv=3, scoring='f1'
    )
    rf.fit(X_train_balanced, y_train_balanced)
    models['random_forest'] = rf.best_estimator_
    
    
    if XGBOOST_AVAILABLE:
        print("Training XGBoost...")
        scale_pos_weight = len(y_train[y_train==0]) / max(len(y_train[y_train==1]), 1)
        xgb_params = {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.1, 0.2]}
        xgb_model = GridSearchCV(
            xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss'),
            xgb_params, cv=3, scoring='f1'
        )
        xgb_model.fit(X_train_balanced, y_train_balanced)
        models['xgboost'] = xgb_model.best_estimator_
    
    
    for name, model in models.items():
        if name == 'logistic_regression':
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            'accuracy': float(round(accuracy_score(y_test, y_pred) * 100, 2)),
            'precision': float(round(precision_score(y_test, y_pred, zero_division=0) * 100, 2)),
            'recall': float(round(recall_score(y_test, y_pred, zero_division=0) * 100, 2)),
            'f1_score': float(round(f1_score(y_test, y_pred, zero_division=0) * 100, 2)),
            'roc_auc': float(round(roc_auc_score(y_test, y_prob) * 100, 2)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        
        if hasattr(model, 'feature_importances_'):
            importance = {k: float(v) for k, v in zip(feature_cols, model.feature_importances_)}
            data_store['feature_importance'][name] = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        elif hasattr(model, 'coef_'):
            importance = {k: float(v) for k, v in zip(feature_cols, np.abs(model.coef_[0]))}
            data_store['feature_importance'][name] = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    data_store['ml_models'] = models
    data_store['model_comparison'] = results
    data_store['feature_cols'] = feature_cols
    
    
    best_model_name = max(results, key=lambda x: results[x]['f1_score'])
    data_store['best_model'] = best_model_name
    
    return results

#  AI ALERTS 

def generate_ai_alerts(df, stats):
    """توليد 20 تنبيه ذكي باستخدام AI بناءً على تحليل شامل للبيانات"""
    alerts = []
    
    
    fraud_rate = stats.get('fraud_rate', 0)
    total = len(df)
    fraud_count = int(df['fraud_flag'].sum())
    normal_count = total - fraud_count
    
    
    gov_fraud = df.groupby('governorate').agg({
        'fraud_flag': ['sum', 'mean', 'count']
    }).round(2)
    gov_fraud.columns = ['fraud_cases', 'fraud_rate', 'total']
    gov_fraud = gov_fraud.sort_values('fraud_rate', ascending=False)
    top_5_gov = gov_fraud.head(5)
    
    
    fraud_consumption = df[df['fraud_flag']==1]['consumption_kwh'].mean()
    normal_consumption = df[df['fraud_flag']==0]['consumption_kwh'].mean()
    consumption_diff = ((normal_consumption - fraud_consumption) / normal_consumption * 100) if normal_consumption > 0 else 0
    
    
    fraud_charges = df[df['fraud_flag']==1]['monthly_charges'].mean()
    normal_charges = df[df['fraud_flag']==0]['monthly_charges'].mean()
    
    
    total_arrears = df['payment_arrears'].sum()
    fraud_arrears = df[df['fraud_flag']==1]['payment_arrears'].mean()
    
    
    avg_complaints = df['complaints_count'].mean()
    fraud_complaints = df[df['fraud_flag']==1]['complaints_count'].mean()
    meter_issues_fraud = df[df['fraud_flag']==1]['meter_reading_issues'].mean()
    
    
    high_risk_count = len(df[(df['fraud_flag']==1) & (df['payment_arrears']>=3)])
    
    
    estimated_loss = fraud_count * fraud_charges * 12
    
    
    model_perf = data_store.get('model_comparison', {})
    best_model = max(model_perf.items(), key=lambda x: x[1].get('f1_score', 0))[0] if model_perf else 'غير متوفر'
    best_f1 = model_perf.get(best_model, {}).get('f1_score', 0) if model_perf else 0
    
    
    prompt = f"""أنت محلل بيانات خبير في شركة الكهرباء المصرية. مهمتك توليد 20 تنبيه ذكي ومتنوع بناءً على التحليل التالي:

═══════════════════════════════════════
📊 إحصائيات عامة:
═══════════════════════════════════════
• إجمالي المشتركين: {total:,} مشترك
• حالات السرقة المكتشفة: {fraud_count:,} حالة
• نسبة السرقة الإجمالية: {fraud_rate:.1f}%
• الخسائر السنوية المقدرة: {estimated_loss:,.0f} جنيه

═══════════════════════════════════════
📍 تحليل المحافظات (الأعلى خطورة):
═══════════════════════════════════════
{top_5_gov.to_string()}

═══════════════════════════════════════
⚡ تحليل الاستهلاك:
═══════════════════════════════════════
• متوسط استهلاك العملاء العاديين: {normal_consumption:.0f} كيلووات
• متوسط استهلاك حالات السرقة: {fraud_consumption:.0f} كيلووات
• الفرق في الاستهلاك: {consumption_diff:.1f}%

═══════════════════════════════════════
💰 تحليل الفواتير:
═══════════════════════════════════════
• متوسط فاتورة العملاء العاديين: {normal_charges:.0f} جنيه
• متوسط فاتورة حالات السرقة: {fraud_charges:.0f} جنيه

═══════════════════════════════════════
⚠️ المؤشرات الحرجة:
═══════════════════════════════════════
• إجمالي المتأخرات: {total_arrears:.0f} شهر
• متوسط متأخرات السارقين: {fraud_arrears:.1f} شهر
• متوسط شكاوى السارقين: {fraud_complaints:.1f}
• مشاكل العداد للسارقين: {meter_issues_fraud:.1f}
• عملاء خطورة قصوى (سرقة + متأخرات ≥3): {high_risk_count} عميل

═══════════════════════════════════════
🤖 أداء نماذج الذكاء الاصطناعي:
═══════════════════════════════════════
• أفضل نموذج: {best_model}
• دقة F1-Score: {best_f1}%

═══════════════════════════════════════
📝 المطلوب:
═══════════════════════════════════════
اكتب بالضبط 20 تنبيه متنوع وذكي. كل تنبيه في سطر منفصل.

التنبيهات يجب أن تشمل:
1. تنبيهات حرجة (5): خسائر مالية، مناطق خطرة، إجراءات عاجلة
2. تنبيهات عالية (5): أنماط مشبوهة، عملاء للفحص، مؤشرات خطر
3. تنبيهات متوسطة (5): توصيات تحسين، مراقبة مناطق، تحديثات
4. تنبيهات معلوماتية (5): إحصائيات، نتائج تحليل، أداء النظام

القواعد:
- كل تنبيه جملة واحدة مختصرة (15-25 كلمة)
- استخدم أرقام وإحصائيات حقيقية من البيانات
- اذكر أسماء المحافظات الفعلية
- لا تستخدم ترقيم أو رموز أو نقاط
- اجعل التنبيهات متنوعة وغير مكررة
- ابدأ كل تنبيه بفعل أو كلمة قوية"""

    try:
        ai_response = analyze_with_ai(prompt, max_tokens=1500)
        alert_lines = [line.strip() for line in ai_response.split('\n') if line.strip() and len(line.strip()) > 20]
        
        
        alert_types = ['critical']*5 + ['high']*5 + ['medium']*5 + ['info']*5
        
        for i, line in enumerate(alert_lines[:20]):
            
            clean_line = line.strip()
            for char in ['*', '#', '•', '►', '-', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '0.', '10.', '11.', '12.', '13.', '14.', '15.', '16.', '17.', '18.', '19.', '20.']:
                clean_line = clean_line.replace(char, '').strip()
            
            if len(clean_line) > 15:
                alerts.append({
                    'type': alert_types[i] if i < len(alert_types) else 'info',
                    'message': clean_line,
                    'timestamp': datetime.now().strftime('%H:%M')
                })
        
        
        if len(alerts) < 20:
            extra_alerts = [
                {'type': 'critical', 'message': f'تم رصد {fraud_count} حالة سرقة تمثل {fraud_rate:.1f}% من إجمالي المشتركين'},
                {'type': 'critical', 'message': f'الخسائر السنوية المقدرة تتجاوز {estimated_loss:,.0f} جنيه وتحتاج إجراءات عاجلة'},
                {'type': 'high', 'message': f'يوجد {high_risk_count} عميل بدرجة خطورة قصوى يحتاجون فحص ميداني فوري'},
                {'type': 'high', 'message': f'متوسط استهلاك السارقين أقل بنسبة {consumption_diff:.0f}% من العملاء العاديين'},
                {'type': 'medium', 'message': f'نظام الذكاء الاصطناعي يعمل بكفاءة {best_f1}% في كشف حالات السرقة'},
                {'type': 'info', 'message': f'تم تحليل بيانات {total:,} مشترك بنجاح وتحديث قاعدة البيانات'}
            ]
            for alert in extra_alerts:
                if len(alerts) < 20:
                    alert['timestamp'] = datetime.now().strftime('%H:%M')
                    alerts.append(alert)
    
    except Exception as e:
        print(f"AI Alert Error: {e}")
        
        alerts = [
            {'type': 'critical', 'message': f'تنبيه عاجل: تم اكتشاف {fraud_count} حالة سرقة كهرباء بنسبة {fraud_rate:.1f}%', 'timestamp': datetime.now().strftime('%H:%M')},
            {'type': 'critical', 'message': f'خسائر مالية تقدر بـ {estimated_loss:,.0f} جنيه سنوياً تحتاج إجراءات فورية', 'timestamp': datetime.now().strftime('%H:%M')},
            {'type': 'high', 'message': f'يوجد {high_risk_count} عميل بدرجة خطورة قصوى مطلوب فحصهم عاجلاً', 'timestamp': datetime.now().strftime('%H:%M')},
            {'type': 'high', 'message': f'الفرق في الاستهلاك بين العاديين والسارقين يصل لـ {consumption_diff:.0f}%', 'timestamp': datetime.now().strftime('%H:%M')},
            {'type': 'medium', 'message': f'متوسط المتأخرات لحالات السرقة {fraud_arrears:.1f} شهر', 'timestamp': datetime.now().strftime('%H:%M')},
            {'type': 'info', 'message': f'تم تحليل {total:,} مشترك بنجاح', 'timestamp': datetime.now().strftime('%H:%M')}
        ]
    
    data_store['ai_alerts'] = alerts
    return alerts

# HIGH RISK CUSTOMERS 

def identify_high_risk_customers(df):
    """تحديد العملاء عاليي الخطورة للفحص"""
    high_risk = []
    
    for idx, row in df.iterrows():
        risk_score = 0
        risk_factors = []
        
        
        if row.get('fraud_flag', 0) == 1:
            risk_score += 40
            risk_factors.append('مُصنف كحالة سرقة')
        
        if row.get('payment_arrears', 0) >= 3:
            risk_score += 20
            risk_factors.append(f"متأخرات {row['payment_arrears']} شهر")
        
        if row.get('meter_reading_issues', 0) >= 2:
            risk_score += 20
            risk_factors.append(f"مشاكل عداد: {row['meter_reading_issues']}")
        
        if row.get('complaints_count', 0) >= 3:
            risk_score += 15
            risk_factors.append(f"شكاوى: {row['complaints_count']}")
        
        if row.get('avg_payment_delay_days', 0) > 45:
            risk_score += 15
            risk_factors.append(f"تأخير سداد: {row['avg_payment_delay_days']} يوم")
        
    
        if row.get('consumption_kwh', 0) > 0 and row.get('monthly_charges', 0) > 0:
            ratio = row['monthly_charges'] / row['consumption_kwh']
            if ratio < 0.3 or ratio > 0.8:
                risk_score += 10
                risk_factors.append('نسبة فاتورة/استهلاك غير طبيعية')
        
        if risk_score >= 50:
            high_risk.append({
                'customer_id': row.get('customer_id', f'C{idx}'),
                'name': row.get('name', 'غير محدد'),
                'governorate': row.get('governorate', 'غير محدد'),
                'risk_score': min(risk_score, 100),
                'risk_factors': risk_factors,
                'monthly_charges': row.get('monthly_charges', 0),
                'consumption_kwh': row.get('consumption_kwh', 0),
                'priority': 'عاجل' if risk_score >= 70 else 'مرتفع'
            })
    
    
    high_risk = sorted(high_risk, key=lambda x: x['risk_score'], reverse=True)
    data_store['high_risk_customers'] = high_risk[:50]  # أعلى 50 عميل
    
    return high_risk[:50]

# GEOGRAPHIC ANALYSIS 

def analyze_geographic_patterns(df):
    """تحليل الأنماط الجغرافية"""
    if 'governorate' not in df.columns:
        return {}
    
    geo_analysis = {}
    
    for gov in df['governorate'].unique():
        gov_data = df[df['governorate'] == gov]
        fraud_count = int(gov_data['fraud_flag'].sum())
        total = len(gov_data)
        fraud_rate = (fraud_count / total * 100) if total > 0 else 0
        
        geo_analysis[gov] = {
            'total_subscribers': total,
            'fraud_count': fraud_count,
            'fraud_rate': round(fraud_rate, 1),
            'avg_consumption': round(gov_data['consumption_kwh'].mean(), 1),
            'avg_charges': round(gov_data['monthly_charges'].mean(), 1),
            'total_arrears': int(gov_data['payment_arrears'].sum()),
            'risk_level': 'critical' if fraud_rate > 40 else 'high' if fraud_rate > 25 else 'medium' if fraud_rate > 15 else 'low'
        }
    
    geo_analysis = dict(sorted(geo_analysis.items(), key=lambda x: x[1]['fraud_rate'], reverse=True))
    data_store['geographic_analysis'] = geo_analysis
    
    return geo_analysis

# EXPORT REPORT

def generate_export_report(df, stats):
    """توليد تقرير شامل للتصدير"""
    report = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'summary': {},
        'ai_analysis': '',
        'high_risk_customers': [],
        'recommendations': [],
        'model_performance': {}
    }
    
    
    report['summary'] = {
        'total_subscribers': len(df),
        'fraud_cases': int(df['fraud_flag'].sum()),
        'fraud_rate': round(df['fraud_flag'].mean() * 100, 2),
        'total_revenue': round(df['monthly_charges'].sum(), 2),
        'at_risk_revenue': round(df[df['fraud_flag']==1]['monthly_charges'].sum(), 2),
        'avg_consumption': round(df['consumption_kwh'].mean(), 2),
        'total_arrears_months': int(df['payment_arrears'].sum())
    }
    
    
    prompt = f"""اكتب تقريراً تنفيذياً موجزاً عن حالة سرقة الكهرباء:

الإحصائيات:
- إجمالي المشتركين: {report['summary']['total_subscribers']:,}
- حالات السرقة: {report['summary']['fraud_cases']:,} ({report['summary']['fraud_rate']:.1f}%)
- الإيرادات المعرضة للخطر: {report['summary']['at_risk_revenue']:,.0f} جنيه
- إجمالي المتأخرات: {report['summary']['total_arrears_months']} شهر

أداء النماذج:
{json.dumps(data_store.get('model_comparison', {}), indent=2, ensure_ascii=False)}

اكتب التقرير في 4 فقرات:
1. ملخص الوضع الحالي
2. أهم المخاطر المكتشفة
3. التوصيات العاجلة
4. خطة العمل المقترحة

اكتب بشكل مهني ومباشر بدون رموز أو ترقيم."""

    try:
        report['ai_analysis'] = analyze_with_ai(prompt, max_tokens=600)
    except:
        report['ai_analysis'] = 'تعذر توليد التحليل'
    
   
    report['high_risk_customers'] = data_store.get('high_risk_customers', [])[:20]
    
    
    report['model_performance'] = data_store.get('model_comparison', {})
    
    
    if report['summary']['fraud_rate'] > 30:
        report['recommendations'] = [
            'تشكيل فريق طوارئ للتفتيش الميداني',
            'مراجعة شاملة لأنظمة القراءة والفوترة',
            'تركيب عدادات ذكية في المناطق عالية الخطورة',
            'تفعيل نظام الإنذار المبكر للاستهلاك غير الطبيعي'
        ]
    elif report['summary']['fraud_rate'] > 20:
        report['recommendations'] = [
            'زيادة وتيرة الجولات التفتيشية',
            'تحديث قاعدة بيانات العدادات',
            'تدريب فرق القراءة على اكتشاف التلاعب'
        ]
    else:
        report['recommendations'] = [
            'الحفاظ على مستوى المراقبة الحالي',
            'تحديث دوري للنماذج التنبؤية'
        ]
    
    return report

#  INSIGHTS GENERATION 

def generate_insights(df, stats):
    """توليد رؤى تحليلية"""
    insights = []
    
    
    prompt = f"""حلل بيانات شركة كهرباء مصرية وأعطني 6 رؤى تحليلية مهمة:

البيانات:
- المشتركين: {len(df):,}
- نسبة السرقة: {stats.get('fraud_rate', 0):.1f}%
- متوسط الفاتورة: {stats.get('avg_charges', 0):.0f} جنيه
- متوسط الاستهلاك: {stats.get('avg_consumption', 0):.0f} كيلووات

توزيع العقود:
{df['contract_type'].value_counts().to_string() if 'contract_type' in df.columns else 'غير متوفر'}

توزيع العدادات:
{df['meter_type'].value_counts().to_string() if 'meter_type' in df.columns else 'غير متوفر'}

اكتب 6 رؤى قصيرة ومفيدة، كل رؤية في سطر واحد.
ركز على الأنماط والعلاقات والفرص والمخاطر.
لا تستخدم ترقيم أو رموز."""

    try:
        ai_response = analyze_with_ai(prompt, max_tokens=500)
        insight_lines = [line.strip() for line in ai_response.split('\n') if line.strip()]
        insights = insight_lines[:6]
    except Exception as e:
        print(f"Insights Error: {e}")
        insights = []
    
    return insights

#  FLASK ROUTES 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'لم يتم اختيار ملف'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'لم يتم اختيار ملف'}), 400
        
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return jsonify({'error': 'صيغة الملف غير مدعومة. استخدم CSV أو Excel'}), 400
        
        
        required_cols = ['fraud_flag', 'monthly_charges', 'consumption_kwh']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return jsonify({'error': f'أعمدة مفقودة: {", ".join(missing_cols)}'}), 400
        
        
        data_store['df'] = df
        data_store['df_original'] = df.copy()
        data_store['loaded'] = True
        data_store['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        clear_cache()
        
        
        stats = {
            'total': len(df),
            'fraud_count': int(df['fraud_flag'].sum()),
            'fraud_rate': round(df['fraud_flag'].mean() * 100, 2),
            'avg_charges': round(df['monthly_charges'].mean(), 2),
            'avg_consumption': round(df['consumption_kwh'].mean(), 2),
            'total_revenue': round(df['monthly_charges'].sum(), 2)
        }
        data_store['stats'] = stats
        
        
        data_store['class_imbalance_report'] = generate_class_imbalance_report(df)
        analyze_geographic_patterns(df)
        identify_high_risk_customers(df)
        
        
        print("Training ML models...")
        train_models(df)
        
        
        print("Generating AI alerts...")
        generate_ai_alerts(df, stats)
        
        return jsonify({
            'success': True,
            'message': 'تم رفع البيانات وتحليلها بنجاح',
            'stats': stats,
            'rows': len(df),
            'columns': len(df.columns)
        })
    
    except Exception as e:
        print(f"Upload Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/check_data')
def check_data():
    return jsonify({
        'loaded': data_store['loaded'],
        'last_update': data_store['last_update']
    })

@app.route('/api/stats')
def get_stats():
    if not data_store['loaded']:
        return jsonify({'error': 'لم يتم رفع بيانات'}), 400
    
    return jsonify({
        'success': True,
        'stats': data_store['stats']
    })

@app.route('/api/alerts')
def get_alerts():
    """الحصول على التنبيهات الذكية"""
    if not data_store['loaded']:
        return jsonify({'error': 'لم يتم رفع بيانات'}), 400
    
    
    if not data_store['ai_alerts']:
        generate_ai_alerts(data_store['df'], data_store['stats'])
    
    return jsonify({
        'success': True,
        'alerts': data_store['ai_alerts']
    })

@app.route('/api/refresh-alerts', methods=['POST'])
def refresh_alerts():
    """تحديث التنبيهات"""
    if not data_store['loaded']:
        return jsonify({'error': 'لم يتم رفع بيانات'}), 400
    
    try:
        alerts = generate_ai_alerts(data_store['df'], data_store['stats'])
        return jsonify({
            'success': True,
            'alerts': alerts
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-comparison')
def get_model_comparison():
    """مقارنة أداء النماذج"""
    if not data_store['loaded']:
        return jsonify({'error': 'لم يتم رفع بيانات'}), 400
    
    return jsonify({
        'success': True,
        'comparison': data_store.get('model_comparison', {}),
        'best_model': data_store.get('best_model', ''),
        'feature_importance': data_store.get('feature_importance', {})
    })

@app.route('/api/class-imbalance')
def get_class_imbalance():
    """تقرير عدم توازن الفئات"""
    if not data_store['loaded']:
        return jsonify({'error': 'لم يتم رفع بيانات'}), 400
    
    return jsonify({
        'success': True,
        'report': data_store.get('class_imbalance_report', {})
    })

@app.route('/api/high-risk-customers')
def get_high_risk_customers():
    """العملاء عاليي الخطورة"""
    if not data_store['loaded']:
        return jsonify({'error': 'لم يتم رفع بيانات'}), 400
    
    return jsonify({
        'success': True,
        'customers': data_store.get('high_risk_customers', [])
    })

@app.route('/api/geographic')
def get_geographic():
    """التحليل الجغرافي"""
    if not data_store['loaded']:
        return jsonify({'error': 'لم يتم رفع بيانات'}), 400
    
    return jsonify({
        'success': True,
        'data': data_store.get('geographic_analysis', {})
    })

@app.route('/api/insights')
def get_insights():
    """الرؤى التحليلية"""
    if not data_store['loaded']:
        return jsonify({'error': 'لم يتم رفع بيانات'}), 400
    
    cached = get_cache('insights')
    if cached:
        return jsonify({'success': True, 'insights': cached})
    
    insights = generate_insights(data_store['df'], data_store['stats'])
    set_cache('insights', insights)
    
    return jsonify({
        'success': True,
        'insights': insights
    })

@app.route('/api/export-report')
def export_report():
    """تصدير التقرير الشامل"""
    if not data_store['loaded']:
        return jsonify({'error': 'لم يتم رفع بيانات'}), 400
    
    report = generate_export_report(data_store['df'], data_store['stats'])
    
    return jsonify({
        'success': True,
        'report': report
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """التنبؤ بحالة مشترك جديد"""
    if not data_store['loaded']:
        return jsonify({'error': 'لم يتم رفع بيانات'}), 400
    
    try:
        data = request.json
        
        
        risk_score = 0
        factors = []
        
        tenure = data.get('tenure_months', 0)
        charges = data.get('monthly_charges', 0)
        consumption = data.get('consumption_kwh', 0)
        arrears = data.get('payment_arrears', 0)
        complaints = data.get('complaints_count', 0)
        meter_issues = data.get('meter_reading_issues', 0)
        
        
        if consumption > 0:
            ratio = charges / consumption
            if ratio < 0.3:
                risk_score += 25
                factors.append(f'نسبة فاتورة منخفضة جداً: {ratio:.2f}')
            elif ratio > 0.8:
                risk_score += 15
                factors.append(f'نسبة فاتورة مرتفعة: {ratio:.2f}')
        
        if arrears >= 3:
            risk_score += arrears * 5
            factors.append(f'متأخرات: {arrears} شهر')
        
        if meter_issues >= 2:
            risk_score += meter_issues * 8
            factors.append(f'مشاكل عداد: {meter_issues}')
        
        if complaints >= 3:
            risk_score += complaints * 4
            factors.append(f'شكاوى: {complaints}')
        
        if tenure < 6 and consumption > 1000:
            risk_score += 20
            factors.append('مشترك جديد باستهلاك مرتفع')
        
        risk_score = min(risk_score, 100)
        
        
        if risk_score >= 70:
            level = 'critical'
            status = 'احتمال سرقة مرتفع جداً'
        elif risk_score >= 50:
            level = 'high'
            status = 'احتمال سرقة مرتفع'
        elif risk_score >= 30:
            level = 'medium'
            status = 'يحتاج مراجعة'
        else:
            level = 'low'
            status = 'طبيعي'
        
        
        prompt = f"""حلل حالة مشترك كهرباء:
- فترة الاشتراك: {tenure} شهر
- الفاتورة: {charges} جنيه
- الاستهلاك: {consumption} كيلووات
- المتأخرات: {arrears} شهر
- الشكاوى: {complaints}
- مشاكل العداد: {meter_issues}
- درجة الخطر: {risk_score}%

اكتب تحليلاً موجزاً في فقرتين عن الحالة والتوصيات."""

        try:
            ai_analysis = analyze_with_ai(prompt, max_tokens=300)
        except:
            ai_analysis = ''
        
        
        if risk_score >= 70:
            recommendations = [
                'إرسال فريق تفتيش فوري',
                'مراجعة سجل قراءات العداد',
                'فحص التوصيلات للكشف عن تلاعب'
            ]
        elif risk_score >= 50:
            recommendations = [
                'جدولة زيارة تفتيشية',
                'التحقق من دقة القراءات',
                'متابعة نمط الاستهلاك'
            ]
        else:
            recommendations = [
                'الحساب ضمن النطاق الطبيعي',
                'متابعة روتينية'
            ]
        
        return jsonify({
            'success': True,
            'risk_score': risk_score,
            'level': level,
            'status': status,
            'factors': factors,
            'ai_analysis': ai_analysis,
            'recommendations': recommendations
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/subscribers')
def get_subscribers():
    """قائمة المشتركين مع التصفية"""
    if not data_store['loaded']:
        return jsonify({'error': 'لم يتم رفع بيانات'}), 400
    
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 15, type=int)
    risk_filter = request.args.get('risk', 'all')
    
    df = data_store['df'].copy()
    
    if risk_filter == 'fraud':
        df = df[df['fraud_flag'] == 1]
    elif risk_filter == 'normal':
        df = df[df['fraud_flag'] == 0]
    
    total = len(df)
    start = (page - 1) * per_page
    end = start + per_page
    
    subscribers = df.iloc[start:end].to_dict('records')
    
    return jsonify({
        'success': True,
        'subscribers': subscribers,
        'total': total,
        'page': page,
        'total_pages': (total + per_page - 1) // per_page
    })

@app.route('/api/charts/distribution')
def get_distribution():
    if not data_store['loaded']:
        return jsonify({'error': 'لم يتم رفع بيانات'}), 400
    
    df = data_store['df']
    fraud_count = int(df['fraud_flag'].sum())
    normal_count = len(df) - fraud_count
    
    return jsonify({
        'success': True,
        'data': {
            'labels': ['مشتركين عاديين', 'حالات سرقة'],
            'values': [normal_count, fraud_count]
        }
    })

@app.route('/api/charts/geographic')
def get_geographic_chart():
    if not data_store['loaded']:
        return jsonify({'error': 'لم يتم رفع بيانات'}), 400
    
    geo = data_store['geographic_analysis']
    
    return jsonify({
        'success': True,
        'data': {
            'governorates': list(geo.keys()),
            'fraud_rates': [v['fraud_rate'] for v in geo.values()],
            'totals': [v['total_subscribers'] for v in geo.values()],
            'risk_levels': [v['risk_level'] for v in geo.values()]
        }
    })

@app.route('/api/charts/consumption')
def get_consumption_chart():
    if not data_store['loaded']:
        return jsonify({'error': 'لم يتم رفع بيانات'}), 400
    
    df = data_store['df']
    
    normal = df[df['fraud_flag'] == 0]['consumption_kwh'].tolist()
    fraud = df[df['fraud_flag'] == 1]['consumption_kwh'].tolist()
    
    return jsonify({
        'success': True,
        'data': {
            'normal': normal,
            'fraud': fraud
        }
    })

@app.route('/api/charts/scatter')
def get_scatter():
    if not data_store['loaded']:
        return jsonify({'error': 'لم يتم رفع بيانات'}), 400
    
    df = data_store['df']
    
    normal_df = df[df['fraud_flag'] == 0]
    fraud_df = df[df['fraud_flag'] == 1]
    
    return jsonify({
        'success': True,
        'data': {
            'normal': {
                'x': normal_df['consumption_kwh'].tolist(),
                'y': normal_df['monthly_charges'].tolist()
            },
            'fraud': {
                'x': fraud_df['consumption_kwh'].tolist(),
                'y': fraud_df['monthly_charges'].tolist()
            }
        }
    })

@app.route('/api/charts/model-comparison')
def get_model_comparison_chart():
    if not data_store['loaded']:
        return jsonify({'error': 'لم يتم رفع بيانات'}), 400
    
    comparison = data_store.get('model_comparison', {})
    
    return jsonify({
        'success': True,
        'data': comparison
    })

@app.route('/api/charts/feature-importance')
def get_feature_importance_chart():
    if not data_store['loaded']:
        return jsonify({'error': 'لم يتم رفع بيانات'}), 400
    
    importance = data_store.get('feature_importance', {})
    best_model = data_store.get('best_model', '')
    
    return jsonify({
        'success': True,
        'data': importance.get(best_model, {}),
        'model': best_model
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_chart():
    """تحليل رسم بياني بالـ AI"""
    if not data_store['loaded']:
        return jsonify({'error': 'لم يتم رفع بيانات'}), 400
    
    try:
        chart_type = request.json.get('type', '')
        df = data_store['df']
        stats = data_store['stats']
        
        prompts = {
            'distribution': f"""حلل توزيع المشتركين: إجمالي {stats['total']:,}، سرقة {stats['fraud_count']:,} بنسبة {stats['fraud_rate']:.1f}%.
اكتب تحليلاً في 3 جمل عن دلالة هذا التوزيع والتوصيات.""",
            
            'geographic': f"""حلل التوزيع الجغرافي للسرقة في المحافظات المصرية.
البيانات: {json.dumps({k: v['fraud_rate'] for k, v in list(data_store['geographic_analysis'].items())[:5]}, ensure_ascii=False)}
اكتب تحليلاً في 3 جمل عن الأنماط الجغرافية والمناطق الأكثر خطورة.""",
            
            'consumption': f"""حلل أنماط الاستهلاك بين المشتركين العاديين وحالات السرقة.
متوسط استهلاك العاديين: {df[df['fraud_flag']==0]['consumption_kwh'].mean():.0f} كيلووات
متوسط استهلاك السرقة: {df[df['fraud_flag']==1]['consumption_kwh'].mean():.0f} كيلووات
اكتب تحليلاً في 3 جمل.""",
            
            'model': f"""حلل أداء نماذج التعلم الآلي في كشف السرقة:
{json.dumps(data_store.get('model_comparison', {}), ensure_ascii=False)}
اكتب تحليلاً في 3 جمل عن أفضل نموذج وأسباب تفوقه."""
        }
        
        prompt = prompts.get(chart_type, 'حلل البيانات المتاحة.')
        analysis = analyze_with_ai(prompt, max_tokens=300)
        
        return jsonify({
            'success': True,
            'analysis': analysis
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_loaded': data_store['loaded'],
        'version': '4.0.0',
        'ai_enabled': hf_client is not None,
        'models_trained': len(data_store.get('ml_models', {})) > 0
    })

if __name__ == '__main__':
    print("\n" + "="*70)
    print("   ELECTRICITY FRAUD DETECTION SYSTEM")
    print("   نظام كشف سرقة الكهرباء بالذكاء الاصطناعي")
    print("   Version 4.0.0 | DEPI Graduation Project")
    print("="*70)
    print(f"   AI Status: {'Enabled' if hf_client else 'Disabled'}")
    print(f"   Server: http://localhost:5000")
    print("="*70 + "\n")
    app.run(debug=True, port=5000, host='0.0.0.0')
