"""
RAG Repository - 資料層，負責與向量資料庫溝通
"""

from typing import List, Dict, Optional
import uuid
from qdrant_client.models import Filter, FieldCondition, MatchValue
from src.rag.vector_store import vector_store
from src.rag.data_loader import data_loader
from src.config.config_loader import config


class RAGRepository:
    """RAG 資料層操作"""

    def __init__(self):
        self.vector_store = vector_store
        self.data_loader = data_loader
        self.max_chunk_size = config.get("rag.chunkSize", 500)
        self.chunk_overlap = config.get("rag.chunkOverlap", 100)

    def _chunk_text(self, text: str) -> List[str]:
        """
        按換行符切片文本

        參數：
            text: 要切片的文本

        返回：
            切片列表
        """
        if not text:
            return []

        paragraphs = text.split('\n')
        chunks = []

        for p in paragraphs:
            p = p.strip()
            if not p:
                continue

            # 如果段落長度小於等於最大切片大小，直接加入
            if len(p) <= self.max_chunk_size:
                chunks.append(p)
            else:
                # 使用重疊方式繼續切割
                for i in range(0, len(p), self.max_chunk_size - self.chunk_overlap):
                    chunks.append(p[i:i + self.max_chunk_size])

        return chunks

    # ===== Collection 管理 =====

    def create_collection(self, collection_name: str) -> bool:
        """建立向量集合"""
        return self.vector_store.create_collection(collection_name)

    def list_collections(self) -> List[str]:
        """列出所有集合"""
        return self.vector_store.list_collections()

    def delete_collection(self, collection_name: str) -> bool:
        """刪除集合"""
        return self.vector_store.delete_collection(collection_name)

    # ===== 資料載入 =====

    def add_character_background(
        self,
        conversation_id: str,
        character_id: str,
        background_text: str
    ) -> bool:
        """
        添加角色背景到向量資料庫（自動切片）

        參數：
            conversation_id: 聊天室 ID
            character_id: 角色 ID
            background_text: 背景文本

        返回：
            是否成功
        """
        # 按換行符切片
        chunks = self._chunk_text(background_text)

        if not chunks:
            return True

        # 為每個切片建立文檔
        documents = []
        for idx, chunk in enumerate(chunks):
            doc = {
                "id": str(uuid.uuid4()),
                "text": chunk,
                "conversation_id": conversation_id,
                "character_id": character_id,
                "type": "background",
                "chunk_index": idx
            }
            documents.append(doc)

        return self.vector_store.upsert_documents(
            "characters",
            documents,
            metadata_fields=["conversation_id", "character_id", "type", "chunk_index"]
        )

    def add_fewshots(
        self,
        conversation_id: str,
        character_id: str,
        fewshots: List[str]
    ) -> bool:
        """
        添加 Few-Shot 範例到向量資料庫（陣列中每一筆為一片）

        參數：
            conversation_id: 聊天室 ID
            character_id: 角色 ID
            fewshots: Few-Shot 文本列表

        返回：
            是否成功
        """
        documents = []
        for idx, fewshot_text in enumerate(fewshots):
            # 過濾掉空的 few-shot
            if not fewshot_text or not fewshot_text.strip():
                continue

            doc = {
                "id": str(uuid.uuid4()),
                "text": fewshot_text.strip(),
                "conversation_id": conversation_id,
                "character_id": character_id,
                "type": "few_shot",
                "index": idx
            }
            documents.append(doc)

        if not documents:
            return True

        return self.vector_store.upsert_documents(
            "fewshots",
            documents,
            metadata_fields=["conversation_id", "character_id", "type", "index"]
        )

    # ===== 資料搜尋 =====

    def search_character_background(
        self,
        conversation_id: str,
        query: str
    ) -> List[Dict]:
        """
        搜尋最相關背景資訊

        參數：
            conversation_id: 聊天室 ID
            query: 搜尋文本

        返回：
            最相關背景資訊搜尋結果
        """
        limit = config.get("rag.search.backgroundLimit")
        if limit is None:
            raise ValueError("rag.search.backgroundLimit must be set in config")

        return self.vector_store.search(
            collection_name="characters",
            query=query,
            limit=limit,
            filters={"conversation_id": conversation_id}
        )

    def search_fewshots(
        self,
        conversation_id: str,
        query: str
    ) -> List[Dict]:
        """
        搜尋最相關對話範例

        參數：
            conversation_id: 聊天室 ID
            query: 搜尋文本

        返回：
            最相關對話範例搜尋結果
        """
        limit = config.get("rag.search.fewshotsLimit")
        if limit is None:
            raise ValueError("rag.search.fewshotsLimit must be set in config")

        return self.vector_store.search(
            collection_name="fewshots",
            query=query,
            limit=limit,
            filters={"conversation_id": conversation_id}
        )

    def search_summaries(
        self,
        conversation_id: str,
        query: str
    ) -> List[Dict]:
        """
        搜尋最相關歷史摘要

        參數：
            conversation_id: 聊天室 ID
            query: 搜尋文本

        返回：
            最相關歷史摘要結果
        """
        limit = config.get("rag.search.summariesLimit")
        if limit is None:
            raise ValueError("rag.search.summariesLimit must be set in config")

        return self.vector_store.search(
            collection_name="summaries",
            query=query,
            limit=limit,
            filters={"conversation_id": conversation_id}
        )

    def add_summary(
        self,
        conversation_id: str,
        summary: str
    ) -> bool:
        """
        添加對話摘要到向量資料庫

        參數：
            conversation_id: 聊天室 ID
            summary: 摘要文本

        返回：
            是否成功
        """
        # 為摘要建立文檔
        doc = {
            "id": str(uuid.uuid4()),
            "text": summary,
            "conversation_id": conversation_id,
            "type": "summary",
            "timestamp": __import__('time').time()
        }

        return self.vector_store.upsert_documents(
            "summaries",
            [doc],
            metadata_fields=["conversation_id", "type", "timestamp"]
        )

    # ===== 資料驗證 =====

    def get_conversation_data(self, conversation_id: str) -> Dict:
        """
        查詢某個聊天室的所有 RAG 資料（用於驗證初始化結果）

        參數：
            conversation_id: 聊天室 ID

        返回：
            {
              "characters": [{"text": "...", "type": "background", "chunk_index": 0}, ...],
              "fewshots": [{"text": "...", "type": "few_shot", "index": 0}, ...]
            }
        """
        try:
            # 建立過濾條件
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="conversation_id",
                        match=MatchValue(value=conversation_id)
                    )
                ]
            )

            # 查詢 characters collection
            characters_result = self.vector_store.client.scroll(
                collection_name="characters",
                limit=1000,
                scroll_filter=filter_condition
            )
            characters = [point.payload for point in characters_result[0]] if characters_result[0] else []

            # 查詢 fewshots collection
            fewshots_result = self.vector_store.client.scroll(
                collection_name="fewshots",
                limit=1000,
                scroll_filter=filter_condition
            )
            fewshots = [point.payload for point in fewshots_result[0]] if fewshots_result[0] else []

            return {
                "characters": characters,
                "fewshots": fewshots
            }
        except Exception as e:
            print(f"✗ Failed to get conversation data: {e}")
            return {"characters": [], "fewshots": []}

    # ===== 資料刪除 =====

    def delete_conversation_data(self, conversation_id: str) -> bool:
        """
        刪除某個聊天室的所有 RAG 資料

        參數：
            conversation_id: 聊天室 ID

        返回：
            是否成功
        """
        try:
            # 建立過濾條件
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="conversation_id",
                        match=MatchValue(value=conversation_id)
                    )
                ]
            )

            # 刪除角色背景
            self.vector_store.client.delete(
                collection_name="characters",
                points_selector=filter_condition
            )

            # 刪除 few-shots
            self.vector_store.client.delete(
                collection_name="fewshots",
                points_selector=filter_condition
            )

            print(f"✓ Deleted conversation data for {conversation_id}")
            return True
        except Exception as e:
            print(f"✗ Failed to delete conversation data: {e}")
            return False


# 全局實例
rag_repository = RAGRepository()
