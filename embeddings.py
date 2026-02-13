from jina import Client
import os

def get_jina_embeddings(texts, jina_api_key):
    """
    Convert text into embeddings using Jina RAG client.
    """
    client = Client(
        host=os.getenv("JINA_HOST", "grpc://your-jina-host"),
        api_key=jina_api_key
    )

    # Wrap texts into a format expected by Jina
    inputs = [{"text": t} for t in texts]

    # Send request to Jina
    response = client.post("/index", inputs=inputs)
    
    # Extract embeddings from response
    embeddings = [doc.embedding for doc in response.docs]
    return embeddings
