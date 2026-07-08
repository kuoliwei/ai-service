"""
Chat Controller - HTTP 層，負責接收請求並調用 chat service
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from src.services.chat_service import chat_service

router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])


# ===== Request Models =====

class MessageHistory(BaseModel):
    """聊天歷史訊息"""
    role: str
    text: str


class CharacterInfo(BaseModel):
    """角色基本資訊"""
    name: str
    gender: str
    tags: List[str] = []


class GenerateResponseRequest(BaseModel):
    """生成回應的請求"""
    conversation_id: str
    character_info: CharacterInfo
    conversation_history: List[MessageHistory]
    # 🆕 主角（主人公）名稱：使用者扮演的角色名，可選（未設定則 prompt 不注入主角段落）
    protagonist_name: Optional[str] = None


class GenerateSummaryRequest(BaseModel):
    """生成摘要的請求"""
    conversation_id: str
    prompt: str


# ===== Endpoints =====

@router.post("/generate")
async def generate_response(request: GenerateResponseRequest):
    """
    生成 AI 回應

    接收聊天室資訊和對話歷史，返回 AI 生成的回應
    """
    print(f"📥 [chat_controller] 收到請求: {request}")
    try:
        # 驗證輸入
        if not request.conversation_id:
            raise HTTPException(status_code=400, detail="conversation_id 不能為空")

        if not request.character_info.name:
            raise HTTPException(status_code=400, detail="character_info.name 不能為空")

        if not request.conversation_history:
            raise HTTPException(status_code=400, detail="conversation_history 不能為空")

        # 轉換為 dict（供 service 使用）
        character_info_dict = {
            "name": request.character_info.name,
            "gender": request.character_info.gender,
            "tags": request.character_info.tags
        }

        conversation_history_list = [
            {"role": msg.role, "text": msg.text}
            for msg in request.conversation_history
        ]

        # 呼叫 service 生成回應
        result = chat_service.generate_response(
            conversation_id=request.conversation_id,
            character_info=character_info_dict,
            conversation_history=conversation_history_list,
            protagonist_name=request.protagonist_name
        )

        # 檢查結果
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [chat_controller] 異常: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summary")
async def generate_summary(request: GenerateSummaryRequest):
    """
    生成對話摘要

    接收摘要提示詞，返回 LLM 生成的摘要
    """
    print(f"📥 [chat_controller] 生成摘要: conversationId={request.conversation_id}")
    try:
        # 驗證輸入
        if not request.conversation_id:
            raise HTTPException(status_code=400, detail="conversation_id 不能為空")

        if not request.prompt:
            raise HTTPException(status_code=400, detail="prompt 不能為空")

        # 呼叫 service 生成摘要
        result = chat_service.generate_summary(
            conversation_id=request.conversation_id,
            prompt=request.prompt
        )

        # 檢查結果
        if result["status"] == "error":
            print(f"❌ [chat_controller] chat_service 返回錯誤: {result['message']}")
            raise HTTPException(status_code=500, detail=result["message"])

        print(f"✅ [chat_controller] 摘要生成成功")
        return result

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [chat_controller] 異常: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
