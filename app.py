import streamlit as st
import joblib
import pandas as pd
import numpy as np

# إعداد الصفحة
st.set_page_config(page_title="Fraud Detection System", layout="centered")


@st.cache_resource # عشان يحملهم مرة واحدة بس والبرنامج يبقى سريع
def load_assets():
    model = joblib.load('fraud_model.pkl')
    scaler = joblib.load('scaler.pkl')
    imputer = joblib.load('imputer.pkl')
    all_columns = joblib.load('columns.pkl')
    top_features = joblib.load('top_features.pkl')
    return model, scaler, imputer, all_columns, top_features

model, scaler, imputer, all_columns, top_features = load_assets()

# واجهة المستخدم
st.title("🛡️ نظام كشف الاحتيال المالي")
st.markdown("قم بإدخال بيانات المعاملة للتحقق من سلامتها")

# عمل فورم لإدخال أهم 10 قيم
with st.form("prediction_form"):
    st.subheader("بيانات المعاملة الأساسية")
    
    user_inputs = {}
    # تقسيم المدخلات على عمودين عشان الشكل يبقى أنظف
    col1, col2 = st.columns(2)
    
    for i, feature in enumerate(top_features):
        with col1 if i % 2 == 0 else col2:
            user_inputs[feature] = st.number_input(f"Enter {feature}", value=0.0)
            
    submit = st.form_submit_button("تحليل المعاملة")

if submit:
    # 1. إنشاء DataFrame بكل الأعمدة (كلها أصفار في الأول)
    input_df = pd.DataFrame(0, index=[0], columns=all_columns)
    
    # 2. تحديث الأعمدة اللي المستخدم دخلها بس
    for feat, val in user_inputs.items():
        input_df[feat] = val
        
    # 3. المعالجة (Imputing & Scaling)
    input_imputed = imputer.transform(input_df)
    input_scaled = scaler.transform(input_imputed)
    
    # 4. التوقع
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1] # نسبة الشك في الاحتيال
    
    st.divider()
    if prediction[0] == 1:
        st.error(f"⚠️ تحذير: هذه المعاملة بنسبة كبيرة (احتيال)!")
        st.write(f"نسبة الشك: {probability*100:.2f}%")
    else:
        st.success(f"✅ المعاملة سليمة وآمنة.")
        st.write(f"نسبة الثقة: {(1-probability)*100:.2f}%")