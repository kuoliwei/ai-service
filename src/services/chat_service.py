"""
Chat Service - 聊天業務邏輯層
負責處理聊天請求、組裝 prompt、呼叫引擎生成回應
"""

import json
import os
from pathlib import Path
from src.prompt.prompt_builder import prompt_builder
from src.services.rag_service import rag_service
from src.config.config_loader import config


class ChatService:
    """聊天服務"""

    def __init__(self):
        """初始化服務，載入 behavior_specs"""
        self.behavior_specs = self._load_behavior_specs()

    def _load_behavior_specs(self) -> dict:
        """
        載入 behavior_specs JSON 檔案

        返回：
            dict，behavior_specs 資料
        """
        try:
            # 找到 behavior_specs 檔案路徑
            spec_dir = Path(__file__).parent.parent.parent / "behavior_specs"
            spec_file = spec_dir / "persona_roleplay_behavior_specs_v1_ChineseVersion.json"

            if not spec_file.exists():
                raise FileNotFoundError(f"找不到 behavior_specs 檔案: {spec_file}")

            with open(spec_file, 'r', encoding='utf-8') as f:
                specs = json.load(f)

            print(f"✅ [chat_service] 已載入 behavior_specs: {spec_file}")
            return specs

        except Exception as e:
            print(f"❌ [chat_service] 載入 behavior_specs 失敗: {e}")
            raise

    def generate_response(
        self,
        conversation_id: str,
        character_info: dict,
        conversation_history: list
    ) -> dict:
        """
        生成 AI 回應

        參數：
            conversation_id: str，聊天室 ID
            character_info: dict，角色資訊 {name, gender, tags}
            conversation_history: list，對話歷史 [{role, text}, ...]

        返回：
            dict，{status, message}
        """
        try:
            print(f"\n🤖 [chat_service] 開始生成回應: conversationId={conversation_id}")

            # 1. 檢索 RAG 上下文
            rag_context = {}
            if conversation_history:
                # 取最後一條訊息作為查詢文本
                last_message = conversation_history[-1].get("text", "")
                print(f"📚 [chat_service] 檢索 RAG 上下文，查詢文本: {last_message[:50]}...")
                rag_context = rag_service.get_rag_context(
                    conversation_id=conversation_id,
                    user_message=last_message
                )

                # 詳細 log RAG 檢索結果
                print(f"✅ [chat_service] RAG 檢索完成")
                if rag_context.get("character_background"):
                    print(f"📖 [chat_service] 檢索到背景:\n{rag_context['character_background']}\n")
                else:
                    print(f"⚠️  [chat_service] 未檢索到背景")

                fewshots = rag_context.get("fewshots", [])
                print(f"💬 [chat_service] 檢索到 {len(fewshots)} 個 few-shot 範例")
                for idx, fewshot in enumerate(fewshots, 1):
                    print(f"  範例 {idx}:\n{fewshot}\n")

            # 2. 組裝 system prompt
            print(f"📝 [chat_service] 組裝 system prompt...")
            system_prompt = prompt_builder.build_system_prompt(
                behavior_specs=self.behavior_specs,
                character_info=character_info,
                conversation_history=conversation_history,
                rag_context=rag_context
            )

            # 3. 準備訊息清單（system prompt + 對話歷史）
            messages = [
                {"role": "system", "content": system_prompt}
            ]

            # 加入對話歷史
            for msg in conversation_history:
                messages.append({
                    "role": msg.get("role"),
                    "content": msg.get("text")
                })

            print(f"📊 [chat_service] 訊息清單: {len(messages)} 筆 (1 個 system + {len(conversation_history)} 個對話)")

            # 4. 呼叫 Ollama 生成回應
            print(f"🧠 [chat_service] 呼叫 Ollama 引擎...")
            import ollama

            ollama_model = config.get("ollama.model", "llama2")
            ollama_temperature = config.get("ollama.temperature", 0.7)

            response = ollama.chat(
                model=ollama_model,
                messages=messages,
                options={"temperature": ollama_temperature}
            )

            ai_response = response['message']['content']

            print(f"✅ [chat_service] 回應生成成功")

            return {
                "status": "success",
                "message": ai_response
            }

        except Exception as e:
            print(f"❌ [chat_service] 生成回應失敗: {e}")
            return {
                "status": "error",
                "message": str(e)
            }


# 全局實例
chat_service = ChatService()
