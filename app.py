
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =============================
# Page configuration
# =============================
st.set_page_config(
    page_title="Fraud Detection Risk Scoring",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# Styling
# =============================
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
            color: #e5e7eb;
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        .hero {
            background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(30,41,59,0.95));
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 20px;
            padding: 22px 24px;
            box-shadow: 0 12px 28px rgba(0,0,0,0.25);
            margin-bottom: 16px;
        }
        .hero h1 {
            margin: 0;
            font-size: 2.0rem;
            color: #ffffff;
        }
        .hero p {
            margin-top: 8px;
            color: #cbd5e1;
            font-size: 0.98rem;
        }
        .card {
            background: rgba(15,23,42,0.92);
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 16px;
            padding: 16px 18px;
            box-shadow: 0 8px 22px rgba(0,0,0,0.18);
        }
        .metric-title {
            color: #94a3b8;
            font-size: 0.85rem;
            margin-bottom: 3px;
        }
        .metric-value {
            color: #ffffff;
            font-size: 1.35rem;
            font-weight: 800;
        }
        .low {
            background: linear-gradient(135deg, #052e16, #14532d);
            color: #bbf7d0;
            padding: 14px 16px;
            border-radius: 16px;
            border: 1px solid #166534;
            font-size: 1.25rem;
            font-weight: 800;
            text-align: center;
        }
        .medium {
            background: linear-gradient(135deg, #3f2f05, #713f12);
            color: #fde68a;
            padding: 14px 16px;
            border-radius: 16px;
            border: 1px solid #a16207;
            font-size: 1.25rem;
            font-weight: 800;
            text-align: center;
        }
        .high {
            background: linear-gradient(135deg, #450a0a, #7f1d1d);
            color: #fecaca;
            padding: 14px 16px;
            border-radius: 16px;
            border: 1px solid #dc2626;
            font-size: 1.25rem;
            font-weight: 800;
            text-align: center;
        }
        .small-note {
            color: #94a3b8;
            font-size: 0.90rem;
        }
        .section-label {
            color: #e2e8f0;
            font-weight: 700;
            font-size: 1.05rem;
            margin-bottom: 0.25rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================
# Load artifacts
# =============================
@st.cache_resource
def load_artifacts():
    model = joblib.load("fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Could not load model artifacts: {e}")
    st.stop()

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Time_scaled", "Amount_scaled"]

def risk_label(prob: float):
    if prob < 0.30:
        return "Low Risk", "low"
    elif prob < 0.70:
        return "Medium Risk", "medium"
    return "High Risk", "high"

def build_input(v_dict, time_val, amount_val):
    scaled = scaler.transform(pd.DataFrame([[time_val, amount_val]], columns=["Time", "Amount"]))
    time_scaled = float(scaled[0][0])
    amount_scaled = float(scaled[0][1])

    row = [v_dict[f"V{i}"] for i in range(1, 29)] + [time_scaled, amount_scaled]
    return pd.DataFrame([row], columns=FEATURES)

# =============================
# Header
# =============================
st.markdown(
    """
    <div class="hero">
        <h1>Fraud Detection Risk Scoring Dashboard</h1>
        <p>
            ABA Final Project • Predict suspicious transactions using a trained machine learning model.
            The output shows fraud probability and a business-friendly risk level.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================
# Top metrics
# =============================
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown('<div class="card"><div class="metric-title">Model Type</div><div class="metric-value">Random Forest</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="card"><div class="metric-title">Task</div><div class="metric-value">Fraud Classification</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="card"><div class="metric-title">Risk Output</div><div class="metric-value">Score + Label</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown('<div class="card"><div class="metric-title">Use Case</div><div class="metric-value">Banking Screening</div></div>', unsafe_allow_html=True)

st.write("")

# =============================
# Sidebar
# =============================
with st.sidebar:
    st.header("Control Panel")
    threshold = st.slider("Flagging threshold", 0.10, 0.90, 0.30, 0.05)
    st.caption("Lower threshold = more fraud caught, but more false alarms.")
    st.markdown("---")
    st.subheader("Project Notes")
    st.write(
        "This app uses the Kaggle credit card fraud dataset. "
        "Features V1–V28 are anonymized, so this is a technical risk-scoring demo."
    )

# =============================
# Main layout
# =============================
tab1, tab2, tab3 = st.tabs(["Transaction Scanner", "Project Summary", "Run / Deploy"])

with tab1:
    left, right = st.columns([1.3, 1])

    with left:
        st.markdown('<div class="section-label">Enter Transaction Details</div>', unsafe_allow_html=True)

        with st.form("fraud_form"):
            c1, c2 = st.columns(2)

            with c1:
                time_val = st.number_input("Time", value=0.0)
                amount_val = st.number_input("Amount", value=0.0)

                st.markdown("**V1 to V14**")
                v_dict = {}
                for i in range(1, 15):
                    v_dict[f"V{i}"] = st.number_input(f"V{i}", value=0.0, format="%.6f")

            with c2:
                st.markdown("**V15 to V28**")
                for i in range(15, 29):
                    v_dict[f"V{i}"] = st.number_input(f"V{i}", value=0.0, format="%.6f")

            submitted = st.form_submit_button("Predict Fraud Risk")

    with right:
        st.markdown('<div class="section-label">Prediction Output</div>', unsafe_allow_html=True)

        if submitted:
            try:
                input_df = build_input(v_dict, time_val, amount_val)
                prob = float(model.predict_proba(input_df)[0][1])
                label, cls = risk_label(prob)

                if cls == "low":
                    st.markdown(f'<div class="low">{label}</div>', unsafe_allow_html=True)
                elif cls == "medium":
                    st.markdown(f'<div class="medium">{label}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="high">{label}</div>', unsafe_allow_html=True)

                st.metric("Fraud Probability", f"{prob:.4f}")
                st.progress(prob)

                st.write("**Decision Guidance**")
                if prob < threshold:
                    st.success("Transaction can usually be approved.")
                elif prob < 0.70:
                    st.warning("Transaction should be reviewed manually.")
                else:
                    st.error("Transaction should be flagged or blocked.")

                st.caption(f"Current threshold: {threshold:.2f}")

                with st.expander("Prepared model input"):
                    st.dataframe(input_df, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.info("Fill the transaction values and click **Predict Fraud Risk**.")

    st.markdown("---")
    st.markdown("### Business Interpretation")
    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown('<div class="card"><b>Low Risk</b><br><span class="small-note">Transaction appears normal and may be approved automatically.</span></div>', unsafe_allow_html=True)
    with b2:
        st.markdown('<div class="card"><b>Medium Risk</b><br><span class="small-note">Transaction should be reviewed by a bank officer or rule engine.</span></div>', unsafe_allow_html=True)
    with b3:
        st.markdown('<div class="card"><b>High Risk</b><br><span class="small-note">Transaction should be blocked or sent for urgent manual verification.</span></div>', unsafe_allow_html=True)

with tab2:
    st.subheader("Project Summary")
    st.write(
        "This ABA project solves a real banking problem: detecting suspicious transactions early while reducing false positives."
    )
    st.write("**Business problem:** Fraud causes financial loss and weakens customer trust.")
    st.write("**Analytics objective:** Assign a fraud risk score to each transaction.")
    st.write("**Method:** Supervised machine learning classification using a Random Forest model.")
    st.write("**Outcome:** A dashboard that supports approve / review / block decisions.")

    st.subheader("Key Questions")
    st.write(
        "1. Which transaction patterns are associated with fraud?\n"
        "2. Can machine learning detect fraud better than fixed rules?\n"
        "3. What threshold gives a practical balance between precision and recall?"
    )

with tab3:
    st.subheader("How to Run Locally")
    st.code("streamlit run app.py", language="bash")

    st.subheader("Files Required")
    st.code(
        "app.py\nfraud_model.pkl\nscaler.pkl\nrequirements.txt",
        language="text"
    )

    st.subheader("requirements.txt")
    st.code(
        "streamlit\npandas\nnumpy\nscikit-learn\njoblib",
        language="text"
    )

    st.subheader("Deployment Flow")
    st.write("1. Push files to GitHub")
    st.write("2. Connect GitHub repo to Streamlit Community Cloud")
    st.write("3. Select `app.py` as the main file")
    st.write("4. Deploy")
