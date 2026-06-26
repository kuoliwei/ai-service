"""
RAG Service - 業務邏輯層，協調 RAG 操作
"""

from typing import List, Dict, Optional
from src.repositories.rag_repository import rag_repository
from src.rag.retriever import rag_retriever
from config import config


class RAGService:
    """RAG 服務業務邏輯"""

    def __init__(self):
        self.repository = rag_repository
        self.retriever = rag_retriever

    # ===== 聊天室初始化 =====

    def initialize_conversation(
        self,
        conversation_id: str,
        character_id: str,
        background: str,
        fewshots: List[str] = None
    ) -> Dict:
        """
        初始化聊天室的 RAG 資料

        參數：
            conversation_id: 聊天室 ID
            character_id: 角色 ID
            background: 角色背景文本
            fewshots: Few-Shot 範例列表（可選）

        返回：
            初始化結果
        """
        try:
            # 1. 添加角色背景
            if background:
                bg_success = self.repository.add_character_background(
                    conversation_id, character_id, background
                )
                if not bg_success:
                    raise Exception("Failed to add character background")

            # 2. 添加 Few-Shots
            if fewshots:
                fs_success = self.repository.add_fewshots(
                    conversation_id, character_id, fewshots
                )
                if not fs_success:
                    raise Exception("Failed to add few-shots")

            return {
                "status": "success",
                "conversation_id": conversation_id,
                "character_id": character_id,
                "message": "Conversation RAG data initialized"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    # ===== 聊天室清理 =====

    def cleanup_conversation(self, conversation_id: str) -> Dict:
        """
        清理聊天室的 RAG 資料（聊天室刪除時調用）

        參數：
            conversation_id: 聊天室 ID

        返回：
            清理結果
        """
        try:
            success = self.repository.delete_conversation_data(conversation_id)
            if not success:
                raise Exception("Failed to delete conversation data")

            return {
                "status": "success",
                "message": f"Conversation {conversation_id} RAG data cleaned up"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    # ===== 對話摘要管理 =====

    def add_conversation_summary(
        self,
        conversation_id: str,
        summary_text: str,
        summary_id: str = None
    ) -> Dict:
        """
        添加對話摘要

        參數：
            conversation_id: 聊天室 ID
            summary_text: 摘要內容
            summary_id: 摘要 ID（可選）

        返回：
            添加結果
        """
        try:
            success = self.repository.add_conversation_summary(
                conversation_id, summary_text, summary_id
            )
            if not success:
                raise Exception("Failed to add conversation summary")

            return {
                "status": "success",
                "message": "Conversation summary added"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    # ===== 檢索上下文 =====

    def get_rag_context(
        self,
        conversation_id: str,
        character_id: str,
        user_message: str
    ) -> Dict:
        """
        為對話檢索 RAG 上下文

        參數：
            conversation_id: 聊天室 ID
            character_id: 角色 ID
            user_message: 用戶訊息

        返回：
            包含背景、few-shots、摘要的上下文
        """
        try:
            # 1. 搜尋角色背景
            background = self.repository.search_character_background(
                conversation_id, user_message, limit=1
            )
            background_text = background[0]["text"] if background else None

            # 2. 搜尋 Few-Shots
            fewshots = self.repository.search_fewshots(
                conversation_id, user_message
            )
            fewshots_text = [fs["text"] for fs in fewshots] if fewshots else []

            # 3. 搜尋對話摘要
            summaries = self.repository.search_conversation_summaries(
                conversation_id, user_message
            )
            summaries_text = [s["text"] for s in summaries] if summaries else []

            return {
                "status": "success",
                "character_background": background_text,
                "fewshots": fewshots_text,
                "conversation_summaries": summaries_text
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "character_background": None,
                "fewshots": [],
                "conversation_summaries": []
            }

    # ===== 系統管理 =====

    def get_system_status(self) -> Dict:
        """
        獲取 RAG 系統狀態

        返回：
            系統狀態信息
        """
        try:
            collections = self.repository.list_collections()
            return {
                "status": "ok",
                "rag_enabled": config.RAG_ENABLED,
                "collections": collections,
                "config": {
                    "qdrant_url": config.QDRANT_URL,
                    "ollama_url": config.OLLAMA_URL,
                    "embed_model": config.OLLAMA_EMBED_MODEL,
                    "top_k": config.RAG_TOP_K
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }


# 全局實例
rag_service = RAGService()
