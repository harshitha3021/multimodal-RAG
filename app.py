import streamlit as st
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import io

st.set_page_config(page_title="Multimodal RAG App", layout="wide")
st.title("📚 Multimodal RAG with Groq")

# Ask for Groq API key
groq_api_key = st.text_input("Enter your Groq API Key", type="password")

# Initialize Groq client if key is entered
client = None
if groq_api_key:
    try:
        client = Groq(api_key=groq_api_key)
    except Exception as e:
        st.error(f"Error initializing Groq client: {e}")

# File uploads
st.header("Upload your files")
uploaded_text_files = st.file_uploader("Upload text files", type=["txt"], accept_multiple_files=True)
uploaded_images = st.file_uploader("Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# Function to read text from uploaded files
def read_text_files(files):
    texts = []
    for file in files:
        try:
            content = file.read().decode("utf-8")
            texts.append(content)
        except Exception as e:
            st.warning(f"Could not read {file.name}: {e}")
    return texts

# Extract text content
text_data = read_text_files(uploaded_text_files)

# Show uploaded images
if uploaded_images:
    st.subheader("Uploaded Images")
    for img_file in uploaded_images:
        img = Image.open(img_file)
        st.image(img, caption=img_file.name)

# Generate embeddings locally using TF-IDF (simple for demo)
vectorizer = None
text_embeddings = None
if text_data:
    vectorizer = TfidfVectorizer()
    text_embeddings = vectorizer.fit_transform(text_data)

# Function to get most relevant text chunks
def get_relevant_context(query, embeddings, texts, top_k=3):
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, embeddings)[0]
    top_indices = sims.argsort()[-top_k:][::-1]
    return [texts[i] for i in top_indices]

# Ask user question
st.header("Ask a Question")
question = st.text_input("Enter your question:")

# Generate answer using Groq
if st.button("Generate Answer") and question and text_embeddings is not None and client:
    try:
        # Get relevant text context
        context = get_relevant_context(question, text_embeddings, text_data, top_k=3)
        final_context = "\n\n".join(context)

        # Call Groq LLM
        response = client.chat(
            model="llama3-7b",  # Supported model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Context:\n{final_context}\n\nQuestion: {question}"}
            ]
        )

        answer = response.choices[0].message.content
        st.subheader("Answer:")
        st.write(answer)

    except Exception as e:
        st.error(f"Error generating answer: {e}")

