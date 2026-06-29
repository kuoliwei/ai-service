from src.config.config_loader import config as config_loader

# 使用 config_loader 作為主要配置源
class Config:
    # Server
    PORT = int(config_loader.get("server.port", 6001))
    HOST = "0.0.0.0"

    # Ollama Configuration
    OLLAMA_URL = config_loader.get("ollama.url", "http://localhost:11434")
    OLLAMA_MODEL = config_loader.get("ollama.model", "llama2")
    OLLAMA_TIMEOUT = config_loader.get("ollama.timeout", 300)
    OLLAMA_EMBED_MODEL = config_loader.get("ollama.embedModel", "nomic-embed-text")

    # API Gateway
    GATEWAY_URL = config_loader.get("gateway.url", "http://localhost:8000")

    # Qdrant Configuration
    QDRANT_URL = config_loader.get("qdrant.url", "http://localhost:6333")
    QDRANT_API_KEY = config_loader.get("qdrant.apiKey", None)

    # RAG Configuration
    RAG_ENABLED = config_loader.get("rag.enabled", True)
    RAG_CHUNK_SIZE = config_loader.get("rag.chunkSize", 500)
    RAG_CHUNK_OVERLAP = config_loader.get("rag.chunkOverlap", 100)

    # Collections
    CHARACTER_COLLECTION = "characters"
    FEWSHOTS_COLLECTION = "fewshots"

config = Config()
