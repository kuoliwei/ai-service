"""
RAG Controller - API 層，負責接收請求並調用 RAG Service
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from src.services.rag_service import rag_service

router = APIRouter(prefix="/api/v1/rag", tags=["RAG"])


# ===== Request Models =====

class InitializeConversationRequest(BaseModel):
    """初始化聊天室 RAG 資料的請求"""
    conversation_id: str
    character_id: str
    background: str
    fewshots: Optional[List[str]] = None


class AddConversationSummaryRequest(BaseModel):
    """添加對話摘要的請求"""
    conversation_id: str
    summary_text: str
    summary_id: Optional[str] = None


# ===== Endpoints =====

@router.post("/conversations/initialize")
async def initialize_conversation(request: InitializeConversationRequest):
    """
    初始化聊天室的 RAG 資料

    當 chat-service 建立新聊天室時呼叫此端點
    """
    try:
        result = rag_service.initialize_conversation(
            conversation_id=request.conversation_id,
            character_id=request.character_id,
            background=request.background,
            fewshots=request.fewshots
        )

        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/conversations/{conversation_id}")
async def cleanup_conversation(conversation_id: str):
    """
    清理聊天室的 RAG 資料

    當 chat-service 刪除聊天室時呼叫此端點
    """
    try:
        result = rag_service.cleanup_conversation(conversation_id)

        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conversations/{conversation_id}/summary")
async def add_conversation_summary(
    conversation_id: str,
    request: AddConversationSummaryRequest
):
    """
    添加對話摘要

    定期添加對話摘要作為長期記憶
    """
    try:
        result = rag_service.add_conversation_summary(
            conversation_id=conversation_id,
            summary_text=request.summary_text,
            summary_id=request.summary_id
        )

        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{conversation_id}/context")
async def get_rag_context(
    conversation_id: str,
    character_id: str,
    user_message: str
):
    """
    為聊天室檢索 RAG 上下文

    在生成回應前呼叫此端點取得相關背景、範例和摘要
    """
    try:
        result = rag_service.get_rag_context(
            conversation_id=conversation_id,
            character_id=character_id,
            user_message=user_message
        )

        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_status():
    """
    獲取 RAG 系統狀態
    """
    try:
        result = rag_service.get_system_status()

        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
