import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="Email & Security Alert Classifier",
    page_icon="🛡️",
    layout="centered",
)

st.title("🛡️ Email & Security Alert Classifier")
st.caption(
    "Uses `cybersectony/phishing-email-detection-distilbert_v2.4.1` to detect phishing and suspicious alerts."
)

MODEL_ID = "cybersectony/phishing-email-detection-distilbert_v2.4.1"


@st.cache_resource(show_spinner="Loading classification model...")
def load_classifier():
    return pipeline("text-classification", model=MODEL_ID)


# Sample presets for quick testing
sample_alerts = {
    "Suspicious Phishing Alert": (
        "URGENT SECURITY ALERT: Unusual sign-in detected on your corporate account.\n"
        "Verify your credentials within 2 hours at http://login-verify-security.com to avoid suspension."
    ),
    "Legitimate IT Notification": (
        "Hi team, scheduled maintenance for the internal VPN gateway will take place this Sunday "
        "from 02:00 to 04:00 HKT. No action is required on your part."
    ),
    "Custom Input": "",
}

selected_sample = st.selectbox("Choose a sample or write your own:", list(sample_alerts.keys()))

alert_text = st.text_area(
    "Email or Security Alert Text:",
    value=sample_alerts[selected_sample],
    height=150,
    placeholder="Paste an email body or security alert here...",
)

if st.button("Classify Text", type="primary"):
    if not alert_text.strip():
        st.warning("Please enter an email or security alert before classifying.")
    else:
        try:
            classifier = load_classifier()
            with st.spinner("Analyzing text..."):
                result = classifier(alert_text)[0]

            label = result["label"]
            score = result["score"]

            st.subheader("Classification Result")
            col1, col2 = st.columns(2)
            col1.metric("Predicted Label", label)
            col2.metric("Confidence Score", f"{score:.2%}")

            # Visual status banner based on label
            if "phish" in label.lower() or label == "LABEL_1":
                st.error("⚠️ High risk: This message exhibits characteristics of phishing or a malicious alert.")
            else:
                st.success("✅ Low risk: This message appears legitimate.")

        except Exception as e:
            st.error(f"Error running classification: {e}")
