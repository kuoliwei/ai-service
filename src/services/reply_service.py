"""
Service 層：業務邏輯
負責調用 Ollama、處理 Prompt、協調數據流
"""

from src.repositories.conversation_repository import conversation_repository


class ReplyService:
    def __init__(self):
        self.repository = conversation_repository

    def generate_reply(self, user_id: str, character_id: str, user_message: str, character_context: str = ""):
        """
        生成 AI 回複

        參數：
            user_id: 使用者 ID
            character_id: 角色 ID
            user_message: 使用者訊息
            character_context: 角色背景信息

        返回：
            生成的回複文本
        """
        if not user_message:
            raise ValueError("user_message cannot be empty")

        if not user_id or not character_id:
            raise ValueError("user_id and character_id are required")

        # 1. 獲取對話歷史
        history = self.repository.get_conversation_history(user_id, character_id)

        # 2. 組建 Prompt（待實裝，暫時返回示例）
        prompt = self._build_prompt(character_context, history, user_message)
        print(f"📝 [ReplyService] 生成 Prompt:\n{prompt}")

        # 3. 調用 Ollama（待實裝，暫時返回示例回複）
        reply = f"[示例回複] 收到訊息: {user_message}"

        # 4. 保存訊息到數據庫
        # self.repository.save_message(user_id, character_id, "user", user_message)
        # self.repository.save_message(user_id, character_id, "assistant", reply)

        return {
            "reply": reply,
            "prompt": prompt
        }

    def _build_prompt(self, character_context: str, history: list, user_message: str) -> str:
        """組建 Prompt"""
        prompt = ""

        # 角色背景
        if character_context:
            prompt += f"[角色背景]\n{character_context}\n\n"

        # 對話歷史
        if history:
            prompt += "[對話歷史]\n"
            for msg in history:
                prompt += f"{msg['role']}: {msg['text']}\n"
            prompt += "\n"

        # 當前訊息
        prompt += f"[用戶訊息]\n{user_message}\n\n[回複]"

        return prompt


reply_service = ReplyService()
