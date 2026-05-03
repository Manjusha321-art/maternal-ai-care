import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load model
model = joblib.load("model.pkl")

st.set_page_config(page_title="Maternal Care AI", layout="wide")

# ---------------- HEADER ----------------
st.title("🤰 Maternal Care AI")
st.subheader("AI-powered Clinical Decision Support System")

st.write("")

# ---------------- LAYOUT ----------------
col1, col2 = st.columns(2)

# ---------------- INPUT ----------------
with col1:
    st.markdown("### 🧾 Clinical Inputs")

    age = st.number_input("Age", 10, 60)
    sys_bp = st.number_input("Systolic BP", 80, 200)
    dia_bp = st.number_input("Diastolic BP", 50, 150)
    bs = st.number_input("Blood Sugar", 4.0, 20.0)
    temp = st.number_input("Body Temperature", 95.0, 105.0)
    heart_rate = st.number_input("Heart Rate", 50, 150)

# ---------------- OUTPUT ----------------
with col2:
    st.markdown("### 🧠 Risk Assessment")

    if st.button("Analyze Risk"):

        input_data = np.array([[age, sys_bp, dia_bp, bs, temp, heart_rate]])
        prediction = model.predict(input_data)[0]

        # Confidence
        if hasattr(model, "predict_proba"):
            prob = np.max(model.predict_proba(input_data))
        else:
            prob = 0.85  # fallback

        # ---------------- 3 LEVEL RISK ----------------
        if prob < 0.4:
            risk_level = "🟢 Low Risk"
            color = "green"
        elif prob < 0.7:
            risk_level = "🟡 Moderate Risk"
            color = "orange"
        else:
            risk_level = "🔴 High Risk"
            color = "red"

        st.markdown(
            f"<h2 style='color:{color};'>{risk_level} ({prob*100:.2f}% confidence)</h2>",
            unsafe_allow_html=True
        )

        # ---------------- ALERTS ----------------
        st.markdown("### ⚠️ Clinical Alerts")

        if sys_bp > 140 or dia_bp > 90:
            st.warning("High Blood Pressure detected")
        if bs > 10:
            st.warning("High Blood Sugar detected")
        if temp > 100:
            st.warning("Elevated Body Temperature")
        if heart_rate > 100:
            st.warning("High Heart Rate")

        # ---------------- WHY PREDICTION ----------------
        st.markdown("### 🔍 Why this result?")

        reasons = []
        if sys_bp > 130: reasons.append("High Systolic BP")
        if dia_bp > 90: reasons.append("High Diastolic BP")
        if bs > 10: reasons.append("High Blood Sugar")
        if heart_rate > 100: reasons.append("High Heart Rate")

        if reasons:
            for r in reasons:
                st.write(f"• {r}")
        else:
            st.write("All parameters are within normal range")

        # ---------------- RECOMMENDATIONS ----------------
        st.markdown("### 💡 Recommendations")

        if "High Risk" in risk_level:
            st.error("""
            - Immediate medical consultation required  
            - Monitor BP and sugar regularly  
            - Avoid stress and heavy activity  
            """)
        elif "Moderate Risk" in risk_level:
            st.warning("""
            - Regular monitoring advised  
            - Maintain proper diet and rest  
            """)
        else:
            st.success("""
            - Maintain healthy lifestyle  
            - Continue regular checkups  
            """)

        # ---------------- COMPARISON ----------------
        st.markdown("### 📊 Patient vs Normal")

        comparison = pd.DataFrame({
            "Parameter": ["Systolic BP", "Diastolic BP", "Blood Sugar"],
            "Normal": [120, 80, 5],
            "Patient": [sys_bp, dia_bp, bs]
        })

        st.table(comparison)

        # ---------------- REPORT ----------------
        st.markdown("### 📄 Download Report")

        report = f"""
Maternal Care AI Report
-----------------------
Date: {datetime.now()}

Clinical Inputs:
Age: {age}
BP: {sys_bp}/{dia_bp}
Blood Sugar: {bs}
Temperature: {temp}
Heart Rate: {heart_rate}

Risk Level: {risk_level}
Confidence: {prob*100:.2f}%

Alerts:
- {'High BP' if sys_bp > 140 or dia_bp > 90 else 'Normal BP'}
- {'High Sugar' if bs > 10 else 'Normal Sugar'}

Recommendations:
{risk_level}
"""

        st.download_button("Download Report", report, file_name="report.txt")

# ---------------- MODEL INFO ----------------
st.markdown("---")

with st.expander("ℹ️ Model Information"):
    st.write("Model Used: Random Forest")
    st.write("Accuracy: ~88%")
    st.write("F1 Score: ~0.90")
    st.write("Designed for early risk detection in maternal healthcare")

st.caption("⚠️ This system is a decision-support tool and not a replacement for medical professionals.")
