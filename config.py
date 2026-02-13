# ----------------------------
# Models
# ----------------------------
GROQ_MODEL: str = "llama-3.1-8b-instant"
VISION_MODEL: str = "meta-llama/llama-4-scout-17b-16e-instruct"
JINA_MODEL: str = "jina-embeddings-v4"

# ----------------------------
# API URLs
# ----------------------------
JINA_EMBEDDING_URL: str = "https://api.jina.ai/v1/embeddings"

# ----------------------------
# Defaults / Hyperparameters
# ----------------------------
DEFAULT_CHUNK_SIZE: int = 400
DEFAULT_OVERLAP: int = 80
DEFAULT_TOP_K: int = 5
EMBEDDING_BATCH_SIZE: int = 32
MAX_EMBEDDING_RETRIES: int = 3
MAX_VISION_RETRIES: int = 3
MAX_LLM_TOKENS: int = 512
TEMPERATURE: float = 0

# ----------------------------
# Limits / Safety
# ----------------------------
MAX_IMAGE_SIZE_MB: int = 5  # max size for vision uploads
