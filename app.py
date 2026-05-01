import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Maternal Care AI", layout="wide")

# ----------- CLEAN MEDICAL UI -----------
st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #f7fbff, #eef7f9);
}

/* Header */
.header {
    background: #ffffff;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}

/* Cards */
.card {
    background: #ffffff;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}

/* Button */
.stButton>button {
    background: #2E86C1;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
}

/* Subtle divider */
hr {
    border: none;
    height: 1px;
    background: #ddd;
}
</style>
""", unsafe_allow_html=True)

# ----------- HEADER -----------
st.markdown("""
<div class="header">
    <h1>🤰 Maternal Care AI</h1>
    <p>Smart & Reliable Pregnancy Risk Prediction</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# Image + Info
col_img, col_text = st.columns([1,2])

with col_img:
    st.image("https://images.unsplash.com/photo-1584515933487-779824d29309", use_container_width=True)

with col_text:
    st.markdown("""
    ### Why this system?
    - Helps in early detection of pregnancy risks  
    - Supports doctors in decision-making  
    - Designed for real-world healthcare usage  

    This system uses Machine Learning to analyze patient vitals and predict risk level.
    """)

st.write("")

# Load model
model = joblib.load("model.pkl")

# Layout
col1, col2 = st.columns(2)

# ----------- INPUT -----------
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📝 Patient Details")

    age = st.number_input("Age", 10, 60)
    sys_bp = st.number_input("Systolic BP", 80, 200)
    dia_bp = st.number_input("Diastolic BP", 50, 150)
    bs = st.number_input("Blood Sugar", 0.0)
    temp = st.number_input("Body Temperature", 90.0)
    heart_rate = st.number_input("Heart Rate", 40, 180)

    predict = st.button("🔍 Analyze Risk")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------- OUTPUT -----------
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Prediction Result")

    if predict:
        input_data = np.array([[age, sys_bp, dia_bp, bs, temp, heart_rate]])
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        # Result
        if prediction == 1:
            st.error("🔴 High Risk Pregnancy")
        else:
            st.success("🟢 Low Risk Pregnancy")

        # Confidence
        st.write(f"Confidence: {prob*100:.2f}%")
        st.progress(int(prob * 100))

        # Chart
        chart_data = pd.DataFrame({
            "Risk Type": ["Low Risk", "High Risk"],
            "Probability": [1 - prob, prob]
        })
        st.bar_chart(chart_data.set_index("Risk Type"))

        # Report
        report = f"""
Maternal Care Report
--------------------
Date: {datetime.now()}

Age: {age}
BP: {sys_bp}/{dia_bp}
Blood Sugar: {bs}
Temperature: {temp}
Heart Rate: {heart_rate}

Prediction: {"High Risk" if prediction else "Low Risk"}
Confidence: {prob*100:.2f}%
"""

        st.download_button("📄 Download Report", report)

    else:
        st.info("Enter details and click Analyze Risk.")

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.write("")
st.markdown("---")
st.caption("Maternal Care AI • Healthcare Decision Support System")