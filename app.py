import streamlit as st
import os

st.title("Multimodal RAG App")

# ------------------------------
# STEP 1: Enter API Keys
# ------------------------------
groq_api_key = st.text_input("Enter Groq API Key", type="password")
jina_api_key = st.text_input("Enter Jina API Key", type="password")

if not groq_api_key or not jina_api_key:
    st.warning("Please enter both API keys to continue.")
    st.stop()

# ------------------------------
# STEP 2: File Upload (Text & Image)
# ------------------------------
st.header("Upload your files")
uploaded_text = st.file_uploader("Upload a text file", type=["txt"])
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if not uploaded_text and not uploaded_image:
    st.info("Please upload at least one text or image file.")
    st.stop()

text_content = ""
if uploaded_text:
    text_content = uploaded_text.read().decode("utf-8")

# ------------------------------
# STEP 3: Initialize Groq Client
# ------------------------------
groq_client = None
try:
    from groq import Client

    # Groq SDK requires api_key in constructor OR environment variable
    groq_client = Client(api_key=groq_api_key)
    st.success("Groq client initialized ✅")
except Exception as e:
    st.error(f"Error initializing Groq client: {e}")
    st.stop()

# ------------------------------
# STEP 4: Initialize Jina Client (example)
# ------------------------------
jina_client = None
try:
    from jina import Client as JinaClient
    jina_client = JinaClient(api_key=jina_api_key)
    st.success("Jina client initialized ✅")
except Exception as e:
    st.warning(f"Could not initialize Jina client: {e}")
    st.info("Jina queries will be disabled.")

# ------------------------------
# STEP 5: Query Input
# ------------------------------
question = st.text_input("Ask a question about your uploaded file(s):")

# ------------------------------
# STEP 6: Generate Answer
# ------------------------------
def generate_answer_groq(client, context, question):
    """Example Groq query function"""
    try:
        response = client.run(
            model="llama3-8b-8192",  # Use a currently supported model
            prompt=f"Context:\n{context}\n\nQuestion: {question}",
            max_tokens=200
        )
        return response
    except Exception as e:
        return f"Error querying Groq: {e}"

if st.button("Get Answer") and question:
    final_context = text_content if text_content else "No text uploaded"

    if uploaded_image:
        final_context += "\n[Image uploaded]"  # image can be processed later

    answer = generate_answer_groq(groq_client, final_context, question)
    st.subheader("Answer from Groq")
    st.write(answer)
