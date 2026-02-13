import streamlit as st
from pathlib import Path

# ------------------------------
# STEP 1: Enter API Keys
# ------------------------------
st.title("Multimodal RAG App")
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

# Read text content
text_content = ""
if uploaded_text:
    text_content = uploaded_text.read().decode("utf-8")

# ------------------------------
# STEP 3: Initialize Groq Client
# ------------------------------
groq_client = None
try:
    from groq import Client
    groq_client = Client()               # initialize without arguments
    groq_client.set_api_key(groq_api_key)  # set API key separately
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
            model="llama3-8b",  # make sure this model is active in your account
            prompt=f"Context:\n{context}\n\nQuestion: {question}",
            max_tokens=200
        )
        return response
    except Exception as e:
        return f"Error querying Groq: {e}"

if st.button("Get Answer") and question:
    final_context = text_content if text_content else "No text uploaded"
    
    # Show uploaded image info if available
    if uploaded_image:
        final_context += "\n[Image uploaded]"  # for demonstration, image can be processed separately

    answer = generate_answer_groq(groq_client, final_context, question)
    st.subheader("Answer from Groq")
    st.write(answer)
