# Version 1
import streamlit as st

st.title("ISOM5160")
st.write("ISOM5160 - Python Programming for Business Applications")

import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Sentiment Analysis App", page_icon="💬", layout="centered")

st.title("💬 Sentiment Analysis")
st.caption("Estimates tone, not factual accuracy.")

# Sidebar configuration
with st.sidebar:
    st.header("Model Settings")
    model_id = st.selectbox(
        "Hugging Face Model",
        options=[
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        ],
        index=0,
        help="Meta-Llama-3-8B-Instruct requires Hugging Face authentication and significant VRAM/RAM.",
    )
    hf_token = st.text_input(
        "Hugging Face Token (optional)",
        type="password",
        help="Required for gated models like Meta-Llama-3 if not logged in via CLI.",
    )


@st.cache_resource(show_spinner="Loading model into memory...")
def load_pipeline(selected_model: str, token: str | None):
    kwargs = {"device_map": "auto"}
    if token:
        kwargs["token"] = token
    return pipeline("sentiment-analysis", model=selected_model, **kwargs)


default_text = (
    "Deep Learning (DL) represents a highly promising approach to developing\n"
    "applications in Artificial Intelligence (AI)."
)

text = st.text_area("Enter text to analyze:", value=default_text, height=140)

if st.button("Analyze Sentiment", type="primary"):
    if not text.strip():
        st.warning("Please enter some text before running the analysis.")
    else:
        try:
            sentiment_pipeline = load_pipeline(model_id, hf_token or None)
            with st.spinner("Analyzing sentiment..."):
                result = sentiment_pipeline(text)

            label = result[0]["label"]
            score = result[0]["score"]

            st.subheader("Result")
            col1, col2 = st.columns(2)
            col1.metric("Sentiment", label)
            col2.metric("Confidence Score", f"{score:.4f}")

        except Exception as e:
            st.error(f"Error running pipeline: {e}")
