import streamlit as st
import numpy as np
import faiss
import requests
from pypdf import PdfReader
from PIL import Image
import easyocr
from groq import Groq

# ----------------------------------------
# PAGE CONFIG
# ----------------------------------------
st.set_page_config(page_title="Multimodal AI RAG Assistant", layout="wide")
st.title("📚 Multimodal AI RAG Assistant")

# ----------------------------------------
# API KEYS (SIDEBAR)
# ----------------------------------------
st.sidebar.header("🔑 API Keys")

groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
jina_api_key = st.sidebar.text_input("Enter Jina API Key", type="password")

# ----------------------------------------
# FILE TEXT EXTRACTION
# ----------------------------------------

def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def extract_text_from_txt(file):
    return file.read().decode("utf-8")


def extract_text_from_image(file):
    image = Image.open(file)
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(np.array(image), detail=0)
    return " ".join(result)

# ----------------------------------------
# JINA EMBEDDINGS
# ----------------------------------------

def get_jina_embeddings(texts, api_key):
    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "jina-embeddings-v2-base-en",
        "input": texts
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 200:
        st.error("Jina API error")
        return None

    embeddings = [item["embedding"] for item in response.json()["data"]]
    return np.array(embeddings).astype("float32")

# ----------------------------------------
# VECTOR STORE (FAISS)
# ----------------------------------------

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# ----------------------------------------
# GROQ LLM
# ----------------------------------------

def generate_answer_groq(context, question, api_key):
    client = Groq(api_key=api_key)

    prompt = f"""
    Answer the question based on the context below.

    Context:
    {context}

    Question:
    {question}

    Answer clearly:
    """

    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
    )

    return completion.choices[0].message.content

# ----------------------------------------
# FILE UPLOAD
# ----------------------------------------

uploaded_files = st.file_uploader(
    "Upload Text, PDF, or Image",
    type=["txt", "pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

context_text = ""

if uploaded_files:
    for file in uploaded_files:
        ext = file.name.split(".")[-1].lower()

        if ext == "txt":
            context_text += extract_text_from_txt(file) + "\n"
            st.success(f"{file.name} loaded.")

        elif ext == "pdf":
            context_text += extract_text_from_pdf(file) + "\n"
            st.success(f"{file.name} processed.")

        elif ext in ["png", "jpg", "jpeg"]:
            st.image(file, caption=file.name, use_column_width=True)
            context_text += extract_text_from_image(file) + "\n"
            st.success(f"{file.name} OCR processed.")

# ----------------------------------------
# QUESTION SECTION
# ----------------------------------------

st.markdown("---")
question = st.text_input("🔎 Ask a question about the uploaded content")

if st.button("Generate Answer"):

    if not groq_api_key or not jina_api_key:
        st.warning("Please enter both API keys.")
    elif not context_text:
        st.warning("Please upload at least one file.")
    elif not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Creating embeddings..."):

            # Split context into chunks
            chunks = context_text.split("\n")
            chunks = [c for c in chunks if c.strip() != ""]

            embeddings = get_jina_embeddings(chunks, jina_api_key)

            if embeddings is not None:

                index = create_faiss_index(embeddings)

                # Embed question
                question_embedding = get_jina_embeddings([question], jina_api_key)

                D, I = index.search(question_embedding, k=3)

                retrieved_chunks = [chunks[i] for i in I[0]]

                final_context = "\n".join(retrieved_chunks)

                with st.spinner("Generating answer with Groq..."):
                    answer = generate_answer_groq(final_context, question, groq_api_key)

                st.subheader("📌 AI Answer:")
                st.write(answer)

st.markdown("---")
st.caption("Full AI Multimodal RAG | Groq + Jina + FAISS")
