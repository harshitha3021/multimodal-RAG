import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import os

st.set_page_config(page_title="Local Multimodal RAG App", layout="wide")
st.title("Local Multimodal RAG App")

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

if uploaded_image:
    try:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.info("Note: Image processing is simulated. You can add captions or embeddings later.")
    except Exception as e:
        st.error(f"Error reading image file: {e}")

# -------------------- User Question --------------------
question = st.text_input("Enter your question:")

# -------------------- Local TF-IDF Retrieval --------------------
def generate_answer_local(context: str, question: str) -> str:
    """
    Simple TF-IDF based answer retrieval from uploaded text.
    """
    if not context or not question:
        return "Please upload files and enter a question."

    try:
        # Split text into sentences
        sentences = context.split("\n")
        # TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences + [question])
        # Cosine similarity of question to sentences
        similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        best_idx = similarity.argmax()
        answer = sentences[best_idx]
        return answer
    except Exception as e:
        return f"Error generating answer: {e}"

# -------------------- Generate Answer --------------------
if st.button("Generate Answer"):
    if not uploaded_text and not uploaded_image:
        st.warning("Upload at least a text file first.")
    elif not question:
        st.warning("Enter a question to get an answer.")
    else:
        context_text = text_content if text_content else ""
        if uploaded_image:
            context_text += " [Image uploaded]"  # placeholder for image context
        answer = generate_answer_local(context_text, question)
        st.subheader("Answer:")
        st.write(answer)

