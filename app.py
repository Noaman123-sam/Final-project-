import streamlit as st
import joblib
import pandas as pd
import numpy as np

# إعداد الصفحة
st.set_page_config(page_title="Fraud Detection System", layout="centered")

@st.cache_resource
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

with st.form("prediction_form"):
    st.subheader("بيانات المعاملة الأساسية")
    
    user_inputs = {}
    
    col1, col2 = st.columns(2)
    
    for i, feature in enumerate(top_features):
        with col1 if i % 2 == 0 else col2:
            user_inputs[feature] = st.number_input(
                f"Enter {feature}",
                value=0.0
            )
            
    submit = st.form_submit_button("تحليل المعاملة")

if submit:
    # إنشاء DataFrame بكل الأعمدة
    input_df = pd.DataFrame(0, index=[0], columns=all_columns)
    
    # تحديث الأعمدة اللي المستخدم دخلها
    for feat, val in user_inputs.items():
        input_df[feat] = val
        
    # المعالجة
    input_imputed = imputer.transform(input_df)
    input_scaled = scaler.transform(input_imputed)
    
    # التوقع
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]
    
    st.divider()

    if prediction[0] == 1:
        st.error("⚠️ تحذير: هذه المعاملة بنسبة كبيرة احتيال!")
        st.write(f"نسبة الشك: {probability*100:.2f}%")
    else:
        st.success("✅ المعاملة سليمة وآمنة.")
        st.write(f"نسبة الثقة: {(1-probability)*100:.2f}%")
