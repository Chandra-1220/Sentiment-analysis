import streamlit as st
from transformers import pipeline

# Load sentiment analysis model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

sentiment = load_model()

# Streamlit UI
st.title("😊 Sentiment Analysis Bot")

text = st.text_area("Enter your text")

if st.button("Analyze Sentiment"):
    if text.strip():
        result = sentiment(text)[0]

        st.subheader("Result")
        st.write("**Sentiment:**", result["label"])
        st.write("**Confidence Score:**", f"{result['score']:.2%}")
    else:
        st.warning("Please enter some text.")
