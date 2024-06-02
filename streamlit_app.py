import streamlit as st

from streamlit_utils import classify_text
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# --- Page configuration ---
st.set_page_config(page_title="Emotion Classification", page_icon="🌟")

# --- Title ---
st.title("🌟 Emotion Classification")
st.divider()

# Custom CSS to improve the app's appearance
st.markdown(
    """
    <style>
    .emotion-btn {
        padding: 20px;
        text-align: center;
        width: 100px;
        height: 115px;
        font-size: 20px;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar with instructions and about section ---
st.sidebar.title("❓ How to Use")
st.sidebar.info(
    """
    There are two options for text classification:
    
    1. 📝 **Enter your text**.
    2. 📁 **Upload a text file**.
    
    After providing the text input:
    - 🖱️ **Click the "Classify" button**.
    """
)
st.sidebar.divider()
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    """
    This app uses a pre-trained BERT model to classify text into six categories:
    1. 😢 **Sadness**
    2. 😊 **Joy**
    3. ❤️ **Love**
    4. 😡 **Anger**
    5. 😨 **Fear**
    6. 😲 **Surprise**
    
    The app automatically summarizes texts longer than 512 tokens before classification.
    """
)


@st.cache_resource
def load_model_and_tokenizer():
    """
    Loading pre-trained model, corresponding tokenizer and summarizer
    """
    # Loading tokenizer from Hugging Face Hub account
    tokenizer = AutoTokenizer.from_pretrained("Muzdeka/emotion-classificator")
    model = AutoModelForSequenceClassification.from_pretrained(
        "Muzdeka/emotion-classificator"
    )

    # Loading summarizer
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return tokenizer, model, summarizer


categories = ["😢 Sadness", "😊 Joy", "❤️ Love", "😡 Anger", "😨 Fear", "😲 Surprise"]

uploaded_file = False
tokenizer, model, summarizer = load_model_and_tokenizer()

# Containers for conditional display
text_area_container = st.empty()
file_uploader_container = st.empty()

user_input = text_area_container.text_area("📝 Enter your text:", height=300)

if not user_input:
    uploaded_file = file_uploader_container.file_uploader(
        "📁 Or upload a text file", type=["txt"]
    )
    if uploaded_file:
        user_input = uploaded_file.read().decode("utf-8")
        text_area_container.empty()
        uploaded_file = True
else:
    file_uploader_container.empty()


if user_input:
    if st.button("Classify"):
        st.divider()
        # Classify text
        probabilities = (
            classify_text(tokenizer, model, summarizer, user_input).squeeze().tolist()
        )  # Convert probabilities to list
        predicted_index = probabilities.index(
            max(probabilities)
        )  # Get the predicted class index

        # Create buttons for emotions
        cols = st.columns(len(categories))
        for i, category in enumerate(categories):
            prob_percentage = probabilities[i] * 100
            # Determine the background color based on classification result
            bg_color = f"linear-gradient(to top, #4BBB4F {prob_percentage}%, #ddd {prob_percentage}%)"
            with cols[i]:
                st.markdown(
                    f"""
                    <div class='emotion-btn' style='background: {bg_color};'>
                        {category}<br>{prob_percentage:.2f}%
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
