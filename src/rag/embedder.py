"""
Embedding Service - 使用 Ollama 進行向量化
"""

import requests
from typing import List
from config import config


class OllamaEmbedder:
    """使用 Ollama 的嵌入模型進行向量化"""

    def __init__(self, model: str = None):
        self.model = model or config.OLLAMA_EMBED_MODEL
        self.base_url = config.OLLAMA_URL
        self.timeout = config.OLLAMA_TIMEOUT

    def embed_text(self, text: str) -> List[float]:
        """
        將單個文本向量化

        參數：
            text: 要向量化的文本

        返回：
            浮點數列表（向量）
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to embed text: {e}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        將多個文本向量化

        參數：
            texts: 文本列表

        返回：
            向量列表
        """
        embeddings = []
        for text in texts:
            embedding = self.embed_text(text)
            embeddings.append(embedding)
        return embeddings

    def get_embedding_dimension(self) -> int:
        """獲取嵌入向量的維度"""
        sample_embedding = self.embed_text("test")
        return len(sample_embedding)


# 全局實例
embedder = OllamaEmbedder()
