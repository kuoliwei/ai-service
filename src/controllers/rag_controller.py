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


class GenerateSummaryRequest(BaseModel):
    """生成對話摘要的請求"""
    conversation_id: str
    prompt: str


class AddSummaryRequest(BaseModel):
    """將摘要存入向量資料庫的請求"""
    conversation_id: str
    summary: str




# ===== Endpoints =====

@router.post("/conversations/initialize", status_code=202)
async def initialize_conversation(request: InitializeConversationRequest):
    """
    啟動聊天室 RAG 初始化（非同步）

    當 chat-service 建立新聊天室時呼叫此端點
    立即回傳 202 Accepted，背景執行初始化
    """
    try:
        print(f"📥 [rag_controller] 初始化請求: conversationId={request.conversation_id}")
        result = rag_service.initialize_conversation(
            conversation_id=request.conversation_id,
            character_id=request.character_id,
            background=request.background,
            fewshots=request.fewshots
        )

        # 202 Accepted：已接受請求，背景處理中
        print(f"✅ [rag_controller] 初始化請求已接受，背景處理中")
        return result
    except Exception as e:
        print(f"❌ [rag_controller] 異常: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
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


@router.get("/conversations/{conversation_id}/context")
async def get_rag_context(
    conversation_id: str,
    user_message: str
):
    """
    為聊天室檢索 RAG 上下文

    在生成回應前呼叫此端點取得相關背景和範例
    """
    try:
        result = rag_service.get_rag_context(
            conversation_id=conversation_id,
            user_message=user_message
        )

        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{conversation_id}/status")
async def get_initialization_status(conversation_id: str):
    """
    查詢聊天室 RAG 初始化狀態

    chat-service 輪詢此端點來確認初始化進度
    """
    try:
        print(f"📡 [rag_controller] 查詢初始化狀態: conversationId={conversation_id}")
        result = rag_service.get_initialization_status(conversation_id)

        # 🆕 如果記錄不存在 → 404 Not Found
        if result.get('status') == 'unknown':
            raise HTTPException(
                status_code=404,
                detail=f"Conversation {conversation_id} not found or already deleted"
            )

        print(f"✅ [rag_controller] 狀態查詢: {result['status']}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [rag_controller] 異常: {type(e).__name__}: {str(e)}")
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


@router.post("/summaries")
async def add_summary(request: AddSummaryRequest):
    """
    將摘要存入向量資料庫

    chat-service 在生成摘要後呼叫此端點
    """
    try:
        print(f"📥 [rag_controller] 存入摘要: conversationId={request.conversation_id}")
        result = rag_service.add_summary(
            conversation_id=request.conversation_id,
            summary=request.summary
        )

        if result["status"] == "error":
            print(f"❌ [rag_controller] rag_service 返回錯誤: {result['message']}")
            raise HTTPException(status_code=500, detail=result["message"])

        print(f"✅ [rag_controller] 摘要存儲成功")
        return result
    except Exception as e:
        print(f"❌ [rag_controller] 異常: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
