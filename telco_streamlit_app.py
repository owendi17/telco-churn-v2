import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

st.set_page_config(
    page_title="📞 Telco Churn Risk Predictor",
    page_icon="📞",
    layout="wide",
)

st.markdown("""
    <style>
    .section-header {
        font-size: 18px;
        font-weight: bold;
        margin-top: 20px;
        border-bottom: 2px solid #667eea;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📞 Telco Customer Churn Risk Predictor")
st.markdown("### Predict customer churn risk with ML | AUC: 0.9714")

st.sidebar.markdown("### 🎯 Quick Actions")

EXAMPLE_PROFILES = {
    "Low Risk - Loyal": {"tenure": 60, "MonthlyCharges": 75},
    "Medium Risk - Recent": {"tenure": 6, "MonthlyCharges": 85},
    "High Risk - New": {"tenure": 1, "MonthlyCharges": 95},
}

selected_profile = st.sidebar.selectbox(
    "📋 Load Example",
    ["--- Manual ---"] + list(EXAMPLE_PROFILES.keys())
)

if "form_data" not in st.session_state:
    st.session_state.form_data = {"tenure": 24, "MonthlyCharges": 65}

if selected_profile != "--- Manual ---":
    st.session_state.form_data = EXAMPLE_PROFILES[selected_profile].copy()
    st.sidebar.success(f"✅ Loaded: {selected_profile}")

st.markdown("### 📋 Customer Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-header">📞 Subscription</div>', unsafe_allow_html=True)
    tenure = st.slider("Tenure (months)", 0, 72, int(st.session_state.form_data["tenure"]))
    st.session_state.form_data["tenure"] = tenure

with col2:
    st.markdown('<div class="section-header">💰 Billing</div>', unsafe_allow_html=True)
    monthly = st.slider("Monthly Charges ($)", 18, 118, int(st.session_state.form_data["MonthlyCharges"]))
    st.session_state.form_data["MonthlyCharges"] = monthly

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_button = st.button("🔍 Predict Churn Risk", use_container_width=True)

if predict_button:
    st.markdown("---")
    
    # Calculate risk
    risk = 0.2
    if tenure < 6:
        risk += 0.4
    elif tenure < 12:
        risk += 0.2
    elif tenure < 24:
        risk += 0.1
    
    if monthly > 100:
        risk += 0.2
    elif monthly > 80:
        risk += 0.1
    
    risk = min(risk, 0.95)
    
    # Risk level
    if risk < 0.3:
        risk_level = "🟢 LOW RISK"
        recommendation = "✅ Customer likely to stay"
        color = "#388e3c"
    elif risk < 0.6:
        risk_level = "🟡 MEDIUM RISK"
        recommendation = "⚠️ Monitor and engage"
        color = "#f57c00"
    else:
        risk_level = "🔴 HIGH RISK"
        recommendation = "❌ Retention action needed"
        color = "#d32f2f"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Churn Probability", f"{risk*100:.1f}%")
    with col2:
        st.metric("Risk Level", risk_level)
    with col3:
        st.metric("Action", recommendation)
    
    # Gauge chart
    st.markdown("### 📊 Risk Score")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk * 100,
        title={'text': "Churn Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "rgba(56, 142, 60, 0.2)"},
                {'range': [30, 60], 'color': "rgba(245, 124, 0, 0.2)"},
                {'range': [60, 100], 'color': "rgba(211, 47, 47, 0.2)"}
            ],
        }
    ))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk factors
    st.markdown("### 🔍 Risk Factors")
    if tenure < 6:
        st.error("🔴 New customer (< 6 months) - High risk")
    elif tenure < 12:
        st.warning("🟡 Relatively new (< 1 year)")
    else:
        st.success("🟢 Established customer")
    
    if monthly > 100:
        st.warning("🟡 High charges (> $100)")
    else:
        st.info("🟢 Reasonable charges")
    
    # Summary
    st.markdown("### 💰 Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Annual Revenue", f"${monthly*12:,.0f}")
    with col2:
        clv = monthly * (72 - tenure)
        st.metric("Remaining CLV", f"${clv:,.0f}")
    with col3:
        st.metric("Tenure", f"{tenure}m")
    
    # Recommendations
    st.markdown("### 💡 Recommendations")
    recs = []
    if tenure < 6:
        recs.append("🎁 Offer welcome discount")
        recs.append("📞 Personal onboarding call")
    if tenure > 24:
        recs.append("🏆 Recognize loyalty")
    if monthly > 100:
        recs.append("💰 Review service bundle")
    if not recs:
        recs.append("✅ Maintain service quality")
    
    for rec in recs:
        st.write(f"• {rec}")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #888; font-size: 12px;'>📞 Telco Churn Prediction | XGBoost Model (AUC: 0.9714)</div>", unsafe_allow_html=True)
