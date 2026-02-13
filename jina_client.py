# rag/jina_client.py

class JinaClient:
    def __init__(self):
        pass

    def search(self, query: str):
        return {
            "query": query,
            "results": [
                {"id": 1, "text": "Result 1 for your query"},
                {"id": 2, "text": "Result 2 for your query"},
                {"id": 3, "text": "Result 3 for your query"}
            ]
        }
