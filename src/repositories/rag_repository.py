"""
RAG Repository - 資料層，負責與向量資料庫溝通
"""

from typing import List, Dict, Optional
import uuid
from qdrant_client.models import Filter, FieldCondition, MatchValue
from src.rag.vector_store import vector_store
from src.config.config_loader import config


class RAGRepository:
    """RAG 資料層操作"""

    def __init__(self):
        self.vector_store = vector_store
        # 🆕 【無預設值】切片參數必須由 config 提供，缺鍵直接拋錯
        self.max_chunk_size = config.get("rag.chunkSize")
        if self.max_chunk_size is None:
            raise ValueError("rag.chunkSize must be set in config")
        self.chunk_overlap = config.get("rag.chunkOverlap")
        if self.chunk_overlap is None:
            raise ValueError("rag.chunkOverlap must be set in config")

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

    def check_connection(self):
        """
        檢查 RAG 數據庫連接狀態

        拋出：
            Exception 如果連接失敗
        """
        print(f"📡 [rag_repository] 檢查 Qdrant 連接...")
        # 🆕 【被動報錯】vector_store 會拋異常，直接傳播上去
        self.vector_store.check_connection()

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
        # 🆕 type 明確標示為 character_background（與 protagonist_background 區分）
        documents = []
        for idx, chunk in enumerate(chunks):
            doc = {
                "id": str(uuid.uuid4()),
                "text": chunk,
                "conversation_id": conversation_id,
                "character_id": character_id,
                "type": "character_background",
                "chunk_index": idx
            }
            documents.append(doc)

        return self.vector_store.upsert_documents(
            "characters",
            documents,
            metadata_fields=["conversation_id", "character_id", "type", "chunk_index"]
        )

    def replace_protagonist_background(
        self,
        conversation_id: str,
        background_text: str
    ) -> bool:
        """
        更新主角（主人公）背景到向量資料庫（先刪舊切片，再存新切片）

        與角色背景同一個 characters collection，以 type='protagonist_background' 區分。
        使用者在聊天室編輯並儲存主角人設時呼叫。

        參數：
            conversation_id: 聊天室 ID
            background_text: 主角背景文本（可為空——空則只刪不存，等同清除）

        拋出：
            Exception 如果 Qdrant 操作失敗——【被動報錯】不捕獲
        """
        # 1. 先刪除舊的主角背景切片（重複儲存不累積髒資料）
        filter_condition = Filter(
            must=[
                FieldCondition(key="conversation_id", match=MatchValue(value=conversation_id)),
                FieldCondition(key="type", match=MatchValue(value="protagonist_background"))
            ]
        )
        self.vector_store.client.delete(
            collection_name="characters",
            points_selector=filter_condition
        )
        print(f"✓ [rag_repository] 舊主角背景切片已清除: conversationId={conversation_id}")

        # 2. 切片並存入新的背景（文本為空則到此為止）
        chunks = self._chunk_text(background_text)
        if not chunks:
            print(f"✓ [rag_repository] 主角背景為空，僅清除舊資料")
            return True

        documents = []
        for idx, chunk in enumerate(chunks):
            doc = {
                "id": str(uuid.uuid4()),
                "text": chunk,
                "conversation_id": conversation_id,
                "type": "protagonist_background",
                "chunk_index": idx
            }
            documents.append(doc)

        self.vector_store.upsert_documents(
            "characters",
            documents,
            metadata_fields=["conversation_id", "type", "chunk_index"]
        )
        print(f"✓ [rag_repository] 主角背景已存入: {len(documents)} 片")
        return True

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

        # 🆕 加上 type 過濾：characters collection 現在同時存放角色背景與主角背景
        return self.vector_store.search(
            collection_name="characters",
            query=query,
            limit=limit,
            filters={"conversation_id": conversation_id, "type": "character_background"}
        )

    def search_character_personality(
        self,
        conversation_id: str
    ) -> List[Dict]:
        """
        搜尋最能代表角色性格的背景切片

        🆕 與其他檢索不同：不使用當前對話當索引，而是用 config 中固定的
        「性格」語意關鍵詞（personalityQuery）——性格錨定不隨劇情漂移，
        每輪都穩定命中【性格特徵】【語言風格】等切片

        參數：
            conversation_id: 聊天室 ID

        返回：
            最相關性格描述切片
        """
        limit = config.get("rag.search.personalityLimit")
        if limit is None:
            raise ValueError("rag.search.personalityLimit must be set in config")

        query = config.get("rag.search.personalityQuery")
        if not query:
            raise ValueError("rag.search.personalityQuery must be set in config")

        return self.vector_store.search(
            collection_name="characters",
            query=query,
            limit=limit,
            filters={"conversation_id": conversation_id, "type": "character_background"}
        )

    def search_protagonist_background(
        self,
        conversation_id: str,
        query: str
    ) -> List[Dict]:
        """
        搜尋最相關主角（主人公）背景資訊

        參數：
            conversation_id: 聊天室 ID
            query: 搜尋文本

        返回：
            最相關主角背景搜尋結果
        """
        limit = config.get("rag.search.protagonistLimit")
        if limit is None:
            raise ValueError("rag.search.protagonistLimit must be set in config")

        return self.vector_store.search(
            collection_name="characters",
            query=query,
            limit=limit,
            filters={"conversation_id": conversation_id, "type": "protagonist_background"}
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
    ) -> str:
        """
        添加對話摘要到向量資料庫

        參數：
            conversation_id: 聊天室 ID
            summary: 摘要文本

        返回：
            summary_id（Qdrant point id，UUID 字串）——
            🆕 回傳給 chat-service 記錄到訊息的 summaryId 欄位，供日後精準刪除

        拋出：
            Exception 如果存入失敗（upsert_documents 會拋出具體錯誤）
        """
        # 生成摘要 ID（同時作為 Qdrant point id）
        summary_id = str(uuid.uuid4())

        # 為摘要建立文檔
        doc = {
            "id": summary_id,
            "text": summary,
            "conversation_id": conversation_id,
            "type": "summary",
            "timestamp": __import__('time').time()
        }

        # 🆕 upsert_documents 失敗會直接拋錯，成功才會走到 return
        self.vector_store.upsert_documents(
            "summaries",
            [doc],
            metadata_fields=["conversation_id", "type", "timestamp"]
        )

        print(f"✓ [rag_repository] 摘要已存入: summary_id={summary_id}")
        return summary_id

    def get_latest_summary(self, conversation_id: str) -> Optional[str]:
        """
        取得該聊天室「時間最新」的一筆摘要（依 timestamp 最大者）

        🆕 與 search_summaries（按當前對話相關度檢索）不同：
        這裡不做語義檢索，而是撈出全部摘要、挑時間最新的一筆——
        用來保證 AI 永遠看得到「上一段劇情剛發生什麼」，維持連續性。

        參數：
            conversation_id: 聊天室 ID

        返回：
            最新摘要的文本；若無任何摘要則回 None

        拋出：
            Exception 如果 Qdrant 不可用（scroll 會拋錯）——【被動報錯】不捕獲
        """
        filter_condition = Filter(
            must=[
                FieldCondition(key="conversation_id", match=MatchValue(value=conversation_id))
            ]
        )
        # 摘要每室數量不多，一次 scroll 全撈；with_payload 才拿得到 text/timestamp
        points, _ = self.vector_store.client.scroll(
            collection_name="summaries",
            scroll_filter=filter_condition,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )
        if not points:
            return None

        # 挑 timestamp 最大（最新）的一筆
        latest = max(points, key=lambda p: p.payload.get("timestamp", 0))
        return latest.payload.get("text")

    def delete_summaries(self, summary_ids: List[str]):
        """
        按 summary_id 精準刪除摘要（用於訊息回溯刪除時的記憶清理）

        參數：
            summary_ids: 摘要 ID 列表（= Qdrant point id）

        拋出：
            Exception 如果刪除失敗（Qdrant 不可用等）——【被動報錯】不捕獲
        """
        self.vector_store.delete_points("summaries", summary_ids)

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

        拋出：
            Exception 如果刪除失敗（Qdrant 不可用等）
        """
        # 🆕 【被動報錯】不捕獲異常，Qdrant 不可用時直接拋出
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

        # 刪除歷史摘要
        self.vector_store.client.delete(
            collection_name="summaries",
            points_selector=filter_condition
        )

        print(f"✓ Deleted conversation data for {conversation_id}")
        return True


# 全局實例
rag_repository = RAGRepository()
