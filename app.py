import streamlit as st
import os

# Streamlit page config
st.set_page_config(page_title="Multimodal RAG Assistant", layout="wide")

st.title("📚 Multimodal RAG Assistant")

# Input selection
input_type = st.radio("Choose input type:", ["Text", "Image"])

if input_type == "Text":
    user_query = st.text_area("Enter your query:")
else:
    user_query = None

# API keys
groq_api_key = st.text_input("Enter your Groq API Key:", type="password")
jina_api_key = st.text_input("Enter your Jina API Key:", type="password")

# Upload files
uploaded_text_file = st.file_uploader("Upload TXT or PDF", type=["txt", "pdf"])
uploaded_image_file = st.file_uploader("Upload PNG or JPG", type=["png", "jpg", "jpeg"])

# Other configuration options
llm_model = st.selectbox("LLM Model", ["llama-3.1-8b-instant"])
chunk_size = st.slider("Chunk Size", 300, 1000, 300)
top_k = st.slider("Top-K Retrieval", 1, 10, 3)
retrieval_scope = st.radio("Retrieval Scope", ["all", "text", "image"])

# Query button
if st.button("Ask a question about your data"):
    if not groq_api_key or not jina_api_key:
        st.error("Please enter both API keys!")
    elif not user_query and not uploaded_text_file and not uploaded_image_file:
        st.error("Please enter a query or upload a file!")
    else:
        st.info("Processing your query...")

        # Initialize clients without 'proxies'
        groq_client = GroqClient(api_key=groq_api_key)
        jina_client = JinaClient(api_key=jina_api_key)

        # Prepare embeddings from uploaded files
        embeddings = []
        if uploaded_text_file:
            embeddings += get_jina_embeddings(uploaded_text_file.read(), file_type=uploaded_text_file.type)
        if uploaded_image_file:
            embeddings += get_jina_embeddings(uploaded_image_file.read(), file_type=uploaded_image_file.type)

        # Add query text embeddings if text input
        if user_query:
            embeddings += get_jina_embeddings(user_query, file_type="text")

        # Query Groq
        results = groq_client.query(
            embeddings=embeddings,
            model=llm_model,
            chunk_size=chunk_size,
            top_k=top_k,
            scope=retrieval_scope
        )

        st.subheader("Results:")
        for i, r in enumerate(results):
            st.markdown(f"**Result {i+1}:** {r['text']}")

