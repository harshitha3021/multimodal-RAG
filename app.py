import streamlit as st
from pypdf import PdfReader
from PIL import Image
import easyocr
import numpy as np

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(page_title="Multimodal RAG Assistant", layout="wide")

st.title("📚 Multimodal RAG Assistant")
st.markdown("Upload a Text file, PDF, or Image and ask questions.")

# --------------------------------
# FUNCTIONS
# --------------------------------

def extract_text_from_pdf(uploaded_file):
    pdf = PdfReader(uploaded_file)
    text = ""
    for page in pdf.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def extract_text_from_txt(uploaded_file):
    return uploaded_file.read().decode("utf-8")


def extract_text_from_image(uploaded_file):
    image = Image.open(uploaded_file)
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(np.array(image), detail=0)
    return " ".join(result)


def simple_rag_answer(context, question):
    sentences = context.split(".")
    keywords = question.lower().split()

    relevant_sentences = []
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in keywords):
            relevant_sentences.append(sentence.strip())

    if relevant_sentences:
        return ". ".join(relevant_sentences[:5])
    else:
        return "No relevant information found."


# --------------------------------
# FILE UPLOADER (MULTIPLE TYPES)
# --------------------------------

uploaded_files = st.file_uploader(
    "Upload Text (.txt), PDF, or Image",
    type=["txt", "pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

context_text = ""

if uploaded_files:
    for file in uploaded_files:
        file_type = file.name.split(".")[-1].lower()

        if file_type == "txt":
            context_text += extract_text_from_txt(file) + "\n"
            st.success(f"Text file '{file.name}' loaded.")

        elif file_type == "pdf":
            context_text += extract_text_from_pdf(file) + "\n"
            st.success(f"PDF '{file.name}' processed.")

        elif file_type in ["png", "jpg", "jpeg"]:
            st.image(file, caption=file.name, use_column_width=True)
            context_text += extract_text_from_image(file) + "\n"
            st.success(f"Image '{file.name}' processed with OCR.")

# --------------------------------
# QUESTION SECTION
# --------------------------------

st.markdown("---")
question = st.text_input("🔎 Ask a question about the uploaded content")

if st.button("Generate Answer"):
    if not context_text:
        st.warning("Please upload at least one file first.")
    elif not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            answer = simple_rag_answer(context_text, question)

        st.subheader("📌 Answer:")
        st.write(answer)

st.markdown("---")
st.caption("Multimodal RAG Demo | Supports Text + PDF + Image")
