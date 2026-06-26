"""
RAG Retriever - 統一的檢索邏輯層
"""

from typing import List, Dict, Optional
from config import config
from .vector_store import vector_store


class RAGRetriever:
    """RAG 檢索統一接口"""

    def __init__(self, vector_store=None):
        self.vector_store = vector_store or vector_store
        self.top_k = config.RAG_TOP_K

    def retrieve_character_context(
        self,
        conversation_id: str,
        character_id: str,
        query: str = None,
        limit: int = None
    ) -> Optional[Dict]:
        """
        檢索角色背景

        參數：
            conversation_id: 聊天室 ID
            character_id: 角色 ID
            query: 搜尋查詢（用於語義搜尋）
            limit: 返回數量

        返回：
            角色背景信息
        """
        limit = limit or self.top_k

        # 如果沒有查詢，直接用 character_id 搜尋
        if not query:
            query = character_id

        results = self.vector_store.search(
            collection_name=config.CHARACTER_COLLECTION,
            query=query,
            limit=limit,
            filters={"conversation_id": conversation_id}
        )

        if not results:
            return None

        # 返回最相關的角色背景
        return results[0]

    def retrieve_fewshots(
        self,
        conversation_id: str,
        character_id: str,
        query: str,
        limit: int = None
    ) -> List[Dict]:
        """
        檢索角色的 Few-Shot 範例

        參數：
            conversation_id: 聊天室 ID
            character_id: 角色 ID
            query: 當前用戶訊息（用於語義相關性）
            limit: 返回數量

        返回：
            Few-Shot 範例列表
        """
        limit = limit or self.top_k

        results = self.vector_store.search(
            collection_name=config.FEWSHOTS_COLLECTION,
            query=query,
            limit=limit,
            filters={"conversation_id": conversation_id}
        )

        return results

    def retrieve_conversation_context(
        self,
        conversation_id: str,
        query: str,
        limit: int = None
    ) -> List[Dict]:
        """
        檢索對話歷史摘要

        參數：
            conversation_id: 聊天室 ID
            query: 當前訊息（用於檢索相關對話）
            limit: 返回數量

        返回：
            對話摘要列表
        """
        limit = limit or self.top_k

        results = self.vector_store.search(
            collection_name=config.CONVERSATION_COLLECTION,
            query=query,
            limit=limit,
            filters={"conversation_id": conversation_id}
        )

        return results

    def build_rag_context(
        self,
        conversation_id: str,
        character_id: str,
        user_message: str
    ) -> Dict[str, any]:
        """
        構建完整的 RAG 上下文

        參數：
            conversation_id: 聊天室 ID
            character_id: 角色 ID
            user_message: 當前用戶訊息

        返回：
            包含角色背景、few-shots 和對話歷史的上下文字典
        """
        context = {
            "character_background": None,
            "fewshots": [],
            "conversation_history": []
        }

        # 檢索角色背景
        character = self.retrieve_character_context(conversation_id, character_id, user_message)
        if character:
            context["character_background"] = character.get("text")

        # 檢索 Few-Shots
        fewshots = self.retrieve_fewshots(conversation_id, character_id, user_message)
        if fewshots:
            context["fewshots"] = [fs.get("text") for fs in fewshots]

        # 檢索對話歷史摘要
        conversation = self.retrieve_conversation_context(conversation_id, user_message)
        if conversation:
            context["conversation_history"] = [c.get("text") for c in conversation]

        return context


# 全局實例
rag_retriever = RAGRetriever()
