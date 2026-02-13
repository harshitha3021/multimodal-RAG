# rag/groq_client.py

class GroqClient:
    def __init__(self, api_key=None):
        self.api_key = api_key  # Accepts api_key
        print(f"GroqClient initialized with API key: {api_key}")

    def process(self, text):
        return {
            "input": text,
            "output": f"Processed '{text}' with GroqClient (API key: {self.api_key})"
        }
