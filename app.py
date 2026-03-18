"""
Clinical Risk Stratification Engine
Phase 5 — Streamlit Deployment
Author: Shirish Man Shakya
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clinical Risk Stratification Engine",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS — Clean Light Clinical Theme ───────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background-color: #f5f6fa;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e5ed;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    .app-header {
        background: #ffffff;
        border: 1px solid #e2e5ed;
        border-radius: 10px;
        padding: 1.4rem 2rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }

    .header-badge {
        background: #eef2ff;
        border: 1px solid #c7d2fe;
        border-radius: 8px;
        padding: 0.6rem 0.75rem;
        font-size: 1.4rem;
        line-height: 1;
    }

    .app-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1e2a3a;
        margin: 0 0 0.2rem 0;
        letter-spacing: -0.3px;
    }

    .app-subtitle {
        font-size: 0.8rem;
        color: #6b7a99;
        margin: 0;
        font-weight: 400;
    }

    .header-divider {
        height: 36px;
        width: 1px;
        background: #e2e5ed;
        margin: 0 0.3rem;
    }

    .header-stat {
        text-align: center;
        padding: 0 0.5rem;
    }

    .header-stat-value {
        font-family: 'DM Mono', monospace;
        font-size: 1.05rem;
        font-weight: 500;
        color: #3b5bdb;
        display: block;
    }

    .header-stat-label {
        font-size: 0.67rem;
        color: #6b7a99;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    .section-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.67rem;
        font-weight: 500;
        color: #3b5bdb;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        border-bottom: 1.5px solid #eef2ff;
        padding-bottom: 0.4rem;
        margin: 1.2rem 0 0.8rem 0;
    }

    .sidebar-section {
        font-family: 'DM Mono', monospace;
        font-size: 0.64rem;
        font-weight: 500;
        color: #3b5bdb;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        border-bottom: 1px solid #eef2ff;
        padding-bottom: 0.35rem;
        margin: 1.1rem 0 0.7rem 0;
    }

    .risk-card {
        border-radius: 10px;
        padding: 1.75rem 1.5rem;
        text-align: center;
        margin: 0.5rem 0 1rem 0;
        border: 1.5px solid;
    }

    .risk-low    { background: #f0fdf4; border-color: #86efac; }
    .risk-medium { background: #fffbeb; border-color: #fcd34d; }
    .risk-high   { background: #fef2f2; border-color: #fca5a5; }

    .risk-score {
        font-family: 'DM Mono', monospace;
        font-size: 3.2rem;
        font-weight: 500;
        line-height: 1;
        letter-spacing: -1px;
    }

    .risk-label {
        font-size: 0.72rem;
        font-weight: 600;
        margin-top: 0.6rem;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .risk-description {
        font-size: 0.82rem;
        margin-top: 0.85rem;
        line-height: 1.6;
    }

    .risk-low .risk-score,
    .risk-low .risk-label       { color: #16a34a; }
    .risk-low .risk-description { color: #166534; }

    .risk-medium .risk-score,
    .risk-medium .risk-label       { color: #d97706; }
    .risk-medium .risk-description { color: #92400e; }

    .risk-high .risk-score,
    .risk-high .risk-label       { color: #dc2626; }
    .risk-high .risk-description { color: #991b1b; }

    .info-box {
        background: #f8f9ff;
        border: 1px solid #c7d2fe;
        border-left: 3px solid #3b5bdb;
        border-radius: 0 6px 6px 0;
        padding: 0.9rem 1rem;
        margin: 0.8rem 0;
        font-size: 0.82rem;
        color: #374151;
        line-height: 1.65;
    }

    .model-footer {
        background: #f8f9ff;
        border: 1px solid #e2e5ed;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin-top: 1.2rem;
        font-size: 0.78rem;
        color: #6b7a99;
        line-height: 1.7;
    }

    .stButton > button {
        background: #3b5bdb;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 0.65rem 1.5rem;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.88rem;
        font-weight: 600;
        width: 100%;
        transition: background 0.15s ease;
    }

    .stButton > button:hover  { background: #2f4ac4; color: #ffffff; }
    .stButton > button:active { background: #2541b2; }

    .stSelectbox label,
    .stSlider label,
    .stRadio label {
        font-size: 0.82rem !important;
        color: #374151 !important;
        font-weight: 400 !important;
    }

    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load Model Artifacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open('models/xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    with open('models/threshold.pkl', 'rb') as f:
        threshold = pickle.load(f)
    return model, scaler, feature_names, threshold

try:
    model, scaler, feature_names, threshold = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    load_error = str(e)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="header-badge">🏥</div>
    <div>
        <p class="app-title">Clinical Risk Stratification Engine</p>
        <p class="app-subtitle">
            30-Day Hospital Readmission Risk &nbsp;·&nbsp;
            XGBoost + SHAP Explainability &nbsp;·&nbsp;
            Diabetes 130-US Hospitals (UCI)
        </p>
    </div>
    <div class="header-divider"></div>
    <div class="header-stat">
        <span class="header-stat-value">0.665</span>
        <span class="header-stat-label">ROC AUC</span>
    </div>
    <div class="header-stat">
        <span class="header-stat-value">0.554</span>
        <span class="header-stat-label">Recall</span>
    </div>
    <div class="header-stat">
        <span class="header-stat-value">66,860</span>
        <span class="header-stat-label">Patients</span>
    </div>
    <div class="header-stat">
        <span class="header-stat-value">95</span>
        <span class="header-stat-label">Features</span>
    </div>
</div>
""", unsafe_allow_html=True)

if not artifacts_loaded:
    st.error(f"""
    **Model files not found.**
    Ensure these files exist in a models/ folder next to app.py:
    xgb_model.pkl · scaler.pkl · feature_names.pkl · threshold.pkl
    Error: {load_error}
    """)
    st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Patient Details")
    st.markdown("---")

    st.markdown('<p class="sidebar-section">Demographics</p>',
                unsafe_allow_html=True)

    age = st.selectbox("Age Group", [
        '[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
        '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'
    ], index=6)
    age_map = {
        '[0-10)': 5,  '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45,'[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
        '[80-90)': 85,'[90-100)': 95
    }
    age_val    = age_map[age]
    gender     = st.selectbox("Gender", ["Female", "Male"])
    gender_val = 0 if gender == "Female" else 1
    race       = st.selectbox("Race", [
        "Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"
    ])

    st.markdown('<p class="sidebar-section">Admission Details</p>',
                unsafe_allow_html=True)

    admission_type = st.selectbox("Admission Type", [
        "Emergency", "Urgent", "Elective",
        "Newborn", "Not Available", "NULL", "Trauma Center"
    ], index=0)
    admission_type_map = {
        "Emergency": 1, "Urgent": 2, "Elective": 3, "Newborn": 4,
        "Not Available": 5, "NULL": 6, "Trauma Center": 7
    }
    admission_type_val = admission_type_map[admission_type]

    discharge_disp = st.selectbox("Discharge Disposition", [
        "Discharged to Home", "Transferred to Short Term Hospital",
        "Transferred to SNF", "Discharged to Home with Health Service",
        "NULL / Not Mapped", "Rehab / Outpatient", "Not Mapped"
    ], index=0)
    discharge_map = {
        "Discharged to Home": 1,
        "Transferred to Short Term Hospital": 2,
        "Transferred to SNF": 3,
        "Discharged to Home with Health Service": 6,
        "NULL / Not Mapped": 18,
        "Rehab / Outpatient": 22,
        "Not Mapped": 25
    }
    discharge_val = discharge_map[discharge_disp]

    admission_source = st.selectbox("Admission Source", [
        "Physician Referral", "Clinic Referral", "HMO Referral",
        "Emergency Room", "NULL", "Not Mapped"
    ], index=3)
    source_map = {
        "Physician Referral": 1, "Clinic Referral": 2,
        "HMO Referral": 3, "Emergency Room": 7,
        "NULL": 17, "Not Mapped": 20
    }
    admission_source_val = source_map[admission_source]

    st.markdown('<p class="sidebar-section">Clinical Metrics</p>',
                unsafe_allow_html=True)

    time_in_hospital   = st.slider("Days in Hospital", 1, 14, 4)
    num_lab_procedures = st.slider("Lab Procedures", 1, 132, 43)
    num_procedures     = st.slider("Procedures Performed", 0, 6, 1)
    num_medications    = st.slider("Number of Medications", 1, 81, 15)
    number_diagnoses   = st.slider("Number of Diagnoses", 1, 16, 7)

    st.markdown('<p class="sidebar-section">Prior Utilisation</p>',
                unsafe_allow_html=True)

    number_outpatient = st.slider("Prior Outpatient Visits", 0, 10, 0)
    number_emergency  = st.slider("Prior Emergency Visits", 0, 10, 0)
    number_inpatient  = st.slider("Prior Inpatient Admissions", 0, 10, 0)

    st.markdown('<p class="sidebar-section">Diagnoses</p>',
                unsafe_allow_html=True)

    diag_cats = [
        "Circulatory", "Respiratory", "Digestive", "Diabetes",
        "Injury", "Musculoskeletal", "Genitourinary", "Neoplasms", "Other"
    ]
    diag_1 = st.selectbox("Primary Diagnosis",   diag_cats, index=0)
    diag_2 = st.selectbox("Secondary Diagnosis", diag_cats, index=3)
    diag_3 = st.selectbox("Tertiary Diagnosis",  diag_cats, index=3)

    st.markdown('<p class="sidebar-section">Medications</p>',
                unsafe_allow_html=True)

    change          = st.radio("Medication Changed?",
                               ["No", "Yes"], horizontal=True)
    change_val      = 1 if change == "Yes" else 0
    diabetesMed     = st.radio("On Diabetes Medication?",
                               ["No", "Yes"], horizontal=True, index=1)
    diabetesMed_val = 1 if diabetesMed == "Yes" else 0
    on_insulin_in   = st.radio("On Insulin?",
                               ["No", "Yes"], horizontal=True, index=1)
    on_insulin_val  = 1 if on_insulin_in == "Yes" else 0

    med_count_active  = st.slider("Active Diabetes Medications", 0, 6, 1)
    med_changes_count = st.slider("Medication Dosage Changes", 0, 4, 0)

    st.markdown("---")
    predict_button = st.button("Assess Risk", use_container_width=True)


# ── Feature Vector Builder ────────────────────────────────────────────────────
def build_feature_vector(feature_names):
    row = {feat: 0 for feat in feature_names}
    row.update({
        'gender': gender_val, 'age': age_val,
        'time_in_hospital': time_in_hospital,
        'num_lab_procedures': num_lab_procedures,
        'num_procedures': num_procedures,
        'num_medications': num_medications,
        'number_outpatient': number_outpatient,
        'number_emergency': number_emergency,
        'number_inpatient': number_inpatient,
        'number_diagnoses': number_diagnoses,
        'change': change_val, 'diabetesMed': diabetesMed_val,
        'med_count_active': med_count_active,
        'med_changes_count': med_changes_count,
        'on_insulin': on_insulin_val,
        'prior_utilisation_score': (
            number_inpatient * 3 +
            number_emergency * 2 +
            number_outpatient * 1
        ),
        'is_complex_patient': int(num_medications >= 15 and change_val == 1)
    })
    for col_prefix, val in [
        ('race', race), ('diag_1', diag_1),
        ('diag_2', diag_2), ('diag_3', diag_3)
    ]:
        k = f'{col_prefix}_{val}'
        if k in row:
            row[k] = 1

    for col, val in [
        ('admission_type_id', admission_type_val),
        ('discharge_disposition_id', discharge_val),
        ('admission_source_id', admission_source_val)
    ]:
        k = f'{col}_{val}'
        if k in row:
            row[k] = 1

    return pd.DataFrame([row])[feature_names]


# ── Main Layout ───────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1.6], gap="large")

with col1:
    st.markdown('<p class="section-label">Risk Assessment</p>',
                unsafe_allow_html=True)

    if not predict_button:
        st.markdown("""
        <div class="info-box">
            Complete the patient form in the sidebar and click
            <strong>Assess Risk</strong> to generate a 30-day
            readmission prediction with SHAP-based explanation.<br><br>
            The model was trained on 53,488 diabetic patient records
            and uses 95 clinical features to estimate readmission
            probability.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="section-label">Model Performance</p>',
                    unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        with m1:
            st.metric("ROC AUC", "0.665")
            st.metric("Recall",  "0.554")
        with m2:
            st.metric("PR AUC", "0.182")
            st.metric("F1 Score", "0.233")

        st.markdown("""
        <div class="model-footer">
            <strong>Algorithm:</strong> XGBoost + Optuna (50 trials, 5-fold CV)<br>
            <strong>Class imbalance:</strong> 10:1 via scale_pos_weight<br>
            <strong>Benchmark:</strong> ROC AUC 0.665 is consistent with
            published academic results (0.63–0.70) on this dataset.<br><br>
            Many influential factors — social support, medication
            compliance, follow-up access — are not captured in the
            available features.
        </div>
        """, unsafe_allow_html=True)

    else:
        try:
            X_input = build_feature_vector(feature_names)
            scaler.transform(X_input)
            prob = model.predict_proba(X_input)[0][1]

            if prob < 0.15:
                tier, tier_class = "LOW RISK", "risk-low"
                tier_desc = ("Low probability of 30-day readmission. "
                             "Standard discharge protocol is appropriate.")
            elif prob < 0.30:
                tier, tier_class = "MEDIUM RISK", "risk-medium"
                tier_desc = ("Elevated readmission risk. Consider enhanced "
                             "discharge planning and a follow-up call "
                             "within 7 days.")
            else:
                tier, tier_class = "HIGH RISK", "risk-high"
                tier_desc = ("High probability of 30-day readmission. "
                             "Priority post-discharge intervention is "
                             "strongly recommended.")

            st.markdown(f"""
            <div class="risk-card {tier_class}">
                <div class="risk-score">{prob:.1%}</div>
                <div class="risk-label">{tier}</div>
                <div class="risk-description">{tier_desc}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<p class="section-label">Patient Summary</p>',
                        unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Age", f"{age_val} yrs")
                st.metric("Days in Hospital", time_in_hospital)
            with c2:
                st.metric("Prior Inpatient", number_inpatient)
                st.metric("Medications", num_medications)
            with c3:
                util = number_inpatient*3 + number_emergency*2 + number_outpatient
                st.metric("Utilisation Score", util)
                st.metric("Complex Patient",
                          "Yes" if (num_medications >= 15 and change_val) else "No")

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

with col2:
    st.markdown('<p class="section-label">SHAP Explanation — Top Risk Drivers</p>',
                unsafe_allow_html=True)

    if not predict_button:
        st.markdown("""
        <div class="info-box">
            After assessment, this panel displays a SHAP waterfall chart
            explaining which patient features contributed most to the
            risk prediction — and in which direction.<br><br>
            <strong>Red bars</strong> push risk higher.
            <strong>Blue bars</strong> push risk lower.
            The baseline is the average prediction across all training
            patients.
        </div>
        """, unsafe_allow_html=True)
    else:
        try:
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_input)

            fig, ax = plt.subplots(figsize=(9, 7))
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#ffffff')

            shap.waterfall_plot(
                shap.Explanation(
                    values        = shap_values[0],
                    base_values   = explainer.expected_value,
                    data          = X_input.iloc[0],
                    feature_names = feature_names
                ),
                max_display=15,
                show=False
            )

            ax = plt.gca()
            ax.set_facecolor('#ffffff')
            fig.patch.set_facecolor('#ffffff')

            for text in ax.get_xticklabels() + ax.get_yticklabels():
                text.set_color('#374151')
                text.set_fontsize(9)

            plt.title('SHAP Waterfall — Individual Patient Risk Explanation',
                      fontsize=11, color='#1e2a3a',
                      fontweight='bold', pad=12)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            shap_series = pd.Series(
                shap_values[0], index=feature_names
            ).abs().sort_values(ascending=False)

            top_feature = shap_series.index[0]
            top_val     = shap_values[0][
                list(feature_names).index(top_feature)
            ]
            direction = "increasing" if top_val > 0 else "decreasing"

            st.markdown(f"""
            <div class="info-box">
                <strong>Primary risk driver:</strong>
                <code>{top_feature}</code> is {direction} this
                patient's readmission risk most significantly.<br><br>
                Each bar shows how much a feature pushes the prediction
                above or below the average baseline of
                <strong>{explainer.expected_value:.1%}</strong>.
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"SHAP error: {e}")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#9ca3af; font-size:0.74rem;
            font-family:'DM Mono', monospace; padding: 0.5rem 0 1rem;">
    Clinical Risk Stratification Engine &nbsp;·&nbsp; Shirish Man Shakya
    &nbsp;·&nbsp; Master of Data Science & Innovation, UTS Sydney<br>
    XGBoost · SHAP · Streamlit · Diabetes 130-US Hospitals (UCI ML Repository)<br>
    <em>For portfolio and educational purposes only. Not for clinical use.</em>
</div>
""", unsafe_allow_html=True)