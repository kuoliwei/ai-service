"""
Repository 層：純數據操作
暫時為空，未來連接數據庫
"""

class ConversationRepository:
    def __init__(self):
        pass

    def get_conversation_history(self, user_id: str, character_id: str):
        """取得對話歷史"""
        # TODO: 從數據庫查詢
        return []

    def save_message(self, user_id: str, character_id: str, role: str, text: str):
        """保存訊息到數據庫"""
        # TODO: 保存到數據庫
        return {"id": "msg_xxx", "role": role, "text": text}

conversation_repository = ConversationRepository()
