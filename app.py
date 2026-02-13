import streamlit as st
from groq import Client
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import io

# -----------------------------
# Streamlit app title
# -----------------------------
st.set_page_config(page_title="Multimodal RAG App", layout="wide")
st.title("📄🖼 Multimodal RAG App")

# -----------------------------
# API Key Inputs
# -----------------------------
groq_api_key = st.text_input("Enter Groq API Key", type="password")

# Initialize Groq client safely
groq_client = None
if groq_api_key:
    try:
        groq_client = Client(api_key=groq_api_key)
        st.success("Groq client initialized ✅")
    except Exception as e:
        st.error(f"Error initializing Groq client: {e}")

# -----------------------------
# File uploads
# -----------------------------
st.header("Upload your files")

uploaded_text_file = st.file_uploader("Upload a text file", type=["txt"])
uploaded_image_file = st.file_uploader("Upload an image file", type=["png", "jpg", "jpeg"])

# -----------------------------
# Read text file
# -----------------------------
text_content = ""
if uploaded_text_file:
    text_content = uploaded_text_file.read().decode("utf-8")
    st.subheader("Text content preview:")
    st.write(text_content[:500] + "..." if len(text_content) > 500 else text_content)

# -----------------------------
# Display image
# -----------------------------
if uploaded_image_file:
    image = Image.open(uploaded_image_file)
    st.subheader("Uploaded Image:")
    st.image(image, use_column_width=True)

# -----------------------------
# Question input
# -----------------------------
question = st.text_input("Enter your question about the content:")

# -----------------------------
# TF-IDF based retrieval (simple RAG simulation)
# -----------------------------
def retrieve_context(text, question, top_k=3):
    if not text.strip():
        return ""
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([text])
    question_vec = vectorizer.transform([question])
    sim = cosine_similarity(tfidf_matrix, question_vec).flatten()
    # Simple: return full text (or could split into paragraphs)
    return text

# -----------------------------
# Generate answer using Groq
# -----------------------------
def generate_answer_groq(context, question):
    if not groq_client:
        return "Groq client not initialized."
    if not context.strip():
        return "No context available to answer the question."
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    try:
        response = groq_client.completions.create(
            model="llama3-7b",  # supported model
            prompt=prompt,
            max_tokens=200
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Groq API error: {e}"

# -----------------------------
# Run retrieval and generation
# -----------------------------
if st.button("Generate Answer"):
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        context = retrieve_context(text_content, question)
        answer = generate_answer_groq(context, question)
        st.subheader("Answer:")
        st.write(answer)
