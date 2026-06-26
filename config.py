import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Server
    PORT = int(os.getenv("PORT", 6001))
    HOST = "0.0.0.0"

    # Ollama Configuration
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
    OLLAMA_TIMEOUT = 300
    OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    # API Gateway
    GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8000")

    # Qdrant Configuration
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

    # RAG Configuration
    RAG_ENABLED = os.getenv("RAG_ENABLED", "true").lower() == "true"
    RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", 500))
    RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", 100))
    RAG_TOP_K = int(os.getenv("RAG_TOP_K", 3))

    # Collections
    CHARACTER_COLLECTION = "characters"
    FEWSHOTS_COLLECTION = "fewshots"
    CONVERSATION_COLLECTION = "conversation_summaries"

config = Config()
