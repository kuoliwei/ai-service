"""
Prompt Service - Prompt 工程層
負責組建、優化提示詞，整合 RAG 檢索結果
"""

from config import config
from src.rag.retriever import rag_retriever


class PromptService:
    """Prompt 工程服務"""

    def __init__(self, retriever=None):
        self.retriever = retriever or rag_retriever
        self.rag_enabled = config.RAG_ENABLED

    def build_prompt(
        self,
        character_id: str,
        user_id: str,
        user_message: str,
        character_context: str = None,
        conversation_history: list = None,
        use_rag: bool = None
    ) -> dict:
        """
        組建 Prompt，支持 RAG 檢索

        參數：
            character_id: 角色 ID（用於 RAG 檢索）
            user_id: 用戶 ID（用於 RAG 檢索）
            user_message: 用戶訊息
            character_context: 角色背景信息（如果提供則優先使用）
            conversation_history: 對話歷史
            use_rag: 是否使用 RAG（預設為配置值）

        返回：
            包含 prompt 和 metadata 的字典
        """
        use_rag = use_rag if use_rag is not None else self.rag_enabled

        # 如果啟用 RAG，檢索相關信息
        rag_context = {}
        if use_rag:
            try:
                rag_context = self.retriever.build_rag_context(
                    character_id, user_id, user_message
                )
            except Exception as e:
                print(f"⚠️  RAG retrieval failed: {e}")
                rag_context = {
                    "character_background": None,
                    "fewshots": [],
                    "conversation_history": []
                }

        # 確定最終的角色背景（優先使用 RAG 或傳入的背景）
        final_character_context = character_context or rag_context.get("character_background")

        prompt = ""

        # 1. 角色背景（System 提示詞）
        if final_character_context:
            prompt += self._build_system_prompt(final_character_context)
            prompt += "\n\n"

        # 2. Few-Shot 範例
        if use_rag and rag_context.get("fewshots"):
            prompt += self._build_fewshots(rag_context["fewshots"])
            prompt += "\n\n"

        # 3. 對話歷史（RAG 檢索或傳入）
        rag_conv_history = rag_context.get("conversation_history") if use_rag else None
        final_history = conversation_history or (rag_conv_history if rag_conv_history else None)

        if final_history:
            prompt += self._build_conversation_history(final_history)
            prompt += "\n\n"

        # 4. 當前訊息（用戶輸入）
        prompt += self._build_user_message(user_message)

        return {
            "prompt": prompt,
            "rag_context": rag_context if use_rag else None,
            "character_context_source": "rag" if (use_rag and rag_context.get("character_background")) else "provided"
        }

    def _build_system_prompt(self, character_context: str) -> str:
        """組建系統提示詞（角色背景）"""
        return f"""[角色設定]
{character_context}

請根據以上角色設定，自然地回應用戶的訊息。"""

    def _build_fewshots(self, fewshots: list) -> str:
        """組建 Few-Shot 範例"""
        if not fewshots:
            return ""

        fewshot_text = "[範例]\n"
        for idx, fs in enumerate(fewshots, 1):
            fewshot_text += f"範例 {idx}:\n{fs}\n\n"

        return fewshot_text.strip()

    def _build_conversation_history(self, history) -> str:
        """組建對話歷史"""
        if not history:
            return ""

        # 支持兩種格式
        if isinstance(history, list) and len(history) > 0:
            if isinstance(history[0], dict):
                # 字典格式
                history_text = "[對話歷史]\n"
                for msg in history:
                    role = "角色" if msg.get("role") == "assistant" else "用戶"
                    text = msg.get("text", "")
                    history_text += f"{role}: {text}\n"
            else:
                # 字符串列表格式
                history_text = "[對話歷史]\n"
                for msg in history:
                    history_text += f"{msg}\n"
        else:
            return ""

        return history_text.strip()

    def _build_user_message(self, user_message: str) -> str:
        """組建用戶訊息"""
        return f"""[用戶訊息]
用戶: {user_message}

[回複]
角色:"""

    def validate_prompt(self, prompt: str) -> bool:
        """驗證 Prompt 是否有效"""
        return len(prompt.strip()) > 0


# 全局實例
prompt_service = PromptService()
