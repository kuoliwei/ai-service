"""
Controller 層：HTTP 請求處理
負責解析請求、調用 Service、返回響應
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.services.reply_service import reply_service


router = APIRouter(prefix="/api/v1", tags=["reply"])


# 請求/響應模型
class GenerateReplyRequest(BaseModel):
    user_id: str
    character_id: str
    user_message: str
    character_context: str = ""


class GenerateReplyResponse(BaseModel):
    status: str
    data: dict


@router.post("/generate", response_model=GenerateReplyResponse)
async def generate_reply(request: GenerateReplyRequest):
    """
    生成 AI 回複端點

    請求體：
    {
        "user_id": "usr_xxx",
        "character_id": "char_xxx",
        "user_message": "你好",
        "character_context": "角色背景"
    }
    """
    try:
        print(f"📥 [ReplyController] 收到請求: user_id={request.user_id}, character_id={request.character_id}")

        # 調用 Service
        result = reply_service.generate_reply(
            user_id=request.user_id,
            character_id=request.character_id,
            user_message=request.user_message,
            character_context=request.character_context
        )

        print(f"✅ [ReplyController] 生成成功")
        return GenerateReplyResponse(
            status="success",
            data=result
        )

    except ValueError as e:
        print(f"❌ [ReplyController] 驗證錯誤: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        print(f"❌ [ReplyController] 伺服器錯誤: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
