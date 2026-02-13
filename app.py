import streamlit as st
from pypdf import PdfReader
from PIL import Image
import easyocr
import numpy as np

st.set_page_config(page_title="Multimodal RAG Assistant", layout="wide")

st.title("📚 Multimodal RAG Assistant")
st.markdown("Upload Text, PDF, or Image and ask questions.")

def extract_text_from_pdf(uploaded_file):
    pdf = PdfReader(uploaded_file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_image(uploaded_file):
    image = Image.open(uploaded_file)
    reader = easyocr.Reader(['en'])
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
        return "No relevant information found in the document."

input_type = st.sidebar.radio(
    "Choose Input Type",
    ["Text", "PDF", "Image"]
)

context_text = ""

if input_type == "Text":
    context_text = st.text_area("Enter your text here")

elif input_type == "PDF":
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_pdf:
        with st.spinner("Extracting text from PDF..."):
            context_text = extract_text_from_pdf(uploaded_pdf)
        st.success("PDF text extracted successfully!")

elif input_type == "Image":
    uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Extracting text from Image (OCR)..."):
            context_text = extract_text_from_image(uploaded_image)
        st.success("Text extracted from image!")

st.markdown("---")
question = st.text_input("🔎 Ask a question about the uploaded content")

if st.button("Generate Answer"):
    if not context_text:
        st.warning("Please upload or enter content first.")
    elif not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            answer = simple_rag_answer(context_text, question)
        st.subheader("📌 Answer:")
        st.write(answer)

st.markdown("---")
st.caption("Simple Multimodal RAG Demo | Streamlit Cloud Compatible")
