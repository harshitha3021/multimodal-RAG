import faiss
import numpy as np


class FAISSRetriever:
    def __init__(self, embeddings, metadata):
        """
        embeddings: np.ndarray (N, D)
        metadata: list of dicts
        """

        if len(embeddings) == 0:
            raise ValueError("Embeddings cannot be empty.")

        self.dimension = embeddings.shape[1]

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Use Inner Product (for cosine similarity)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

        self.embeddings = embeddings
        self.metadata = metadata

    def search(self, query_embedding, top_k=5, filter_type=None):
        """
        query_embedding: np.ndarray (1, D)
        """

        if query_embedding is None or len(query_embedding) == 0:
            return []

        # Normalize query
        faiss.normalize_L2(query_embedding)

        # Search more if filtering is applied
        search_k = top_k * 3 if filter_type else top_k

        scores, indices = self.index.search(query_embedding, search_k)

        results = []

        for idx in indices[0]:
            if idx == -1:
                continue

            meta = self.metadata[idx]

            if filter_type and meta.get("type") != filter_type:
                continue

            results.append(idx)

            if len(results) >= top_k:
                break

        return results
