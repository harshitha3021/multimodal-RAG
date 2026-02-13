import streamlit as st
from groq import Client
from sklearn.feature_extraction.text import TfidfVectorizer
import os

st.set_page_config(page_title="Multimodal RAG App", layout="wide")
st.title("Multimodal RAG App")

# -------------------- API KEYS --------------------
groq_api_key = st.text_input("Enter Groq API Key", type="password")
jina_api_key = st.text_input("Enter Jina API Key", type="password")

if not groq_api_key or not jina_api_key:
    st.warning("Please enter both API keys to continue.")
    st.stop()

# -------------------- Initialize Groq Client --------------------
try:
    groq_client = Client(api_key=groq_api_key)
    st.success("Groq client initialized ✅")
except Exception as e:
    st.error(f"Error initializing Groq client: {e}")
    st.stop()

# -------------------- File Upload --------------------
st.header("Upload Files")
uploaded_text = st.file_uploader("Upload a text file", type=["txt"])
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

text_content = ""
if uploaded_text:
    try:
        text_content = uploaded_text.read().decode("utf-8")
        st.text_area("Text File Content", text_content, height=200)
    except Exception as e:
        st.error(f"Error reading text file: {e}")

# Display uploaded image
if uploaded_image:
    from PIL import Image
    try:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    except Exception as e:
        st.error(f"Error reading image file: {e}")

# -------------------- User Question --------------------
question = st.text_input("Enter your question for the model:")

# -------------------- Dummy Embedding & Retrieval --------------------
def generate_answer_groq(context: str, question: str) -> str:
    """
    Example function to generate answer using Groq client.
    Replace with actual Groq API call for LLM if available.
    """
    if not context or not question:
        return "Please upload files and enter a question."
    try:
        # Minimal placeholder logic (replace with real API calls)
        combined = f"Context: {context}\nQuestion: {question}"
        # Example: pretend Groq returns last 100 chars reversed as "answer"
        answer = combined[-100:][::-1]
        return answer
    except Exception as e:
        return f"Error generating answer: {e}"

# -------------------- Generate Answer --------------------
if st.button("Generate Answer"):
    if not uploaded_text and not uploaded_image:
        st.warning("Upload at least one file first.")
    elif not question:
        st.warning("Enter a question to get an answer.")
    else:
        context_text = text_content if text_content else ""
        # For image, you could extract embeddings or captions here if needed
        if uploaded_image:
            context_text += " [Image uploaded]"  # placeholder

        answer = generate_answer_groq(context_text, question)
        st.subheader("Answer:")
        st.write(answer)
