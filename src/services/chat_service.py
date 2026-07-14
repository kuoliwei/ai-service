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
            spec_file = spec_dir / "persona_roleplay_behavior_specs_v1_EnglishVersion.json"

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
        conversation_history: list,
        protagonist_name: str = None
    ) -> dict:
        """
        生成 AI 回應

        參數：
            conversation_id: str，聊天室 ID
            character_info: dict，角色資訊 {name, gender, tags}
            conversation_history: list，對話歷史 [{role, text}, ...]
            protagonist_name: str，主角（主人公）名稱（可選，未設定則不注入主角段落）

        返回：
            dict，{status, message}
        """
        try:
            import time
            start_time = time.time()
            print(f"\n🤖 [chat_service] 開始生成回應: conversationId={conversation_id}, 時間={start_time}")

            # 🆕 【實驗性註解】暫時停用健檢觸發，觀察無健檢時錯誤是否仍能正確傳播
            # # 🆕 【同步檢查】檢查 RAG 資料庫健康狀態
            # print(f"🏥 [chat_service] 檢查 RAG 資料庫健康狀態...")
            # try:
            #     rag_service.check_rag_health()
            #     print(f"✅ [chat_service] RAG 資料庫健康，繼續生成回應")
            # except Exception as e:
            #     # 【被動報錯】如果 RAG 不可用，立即拋錯，不繼續執行
            #     raise Exception(f"RAG health check failed: {str(e)}")

            # 1. 檢索 RAG 上下文
            rag_context = {}
            if conversation_history:
                # 取最後一條訊息作為查詢文本
                last_message = conversation_history[-1].get("text", "")
                rag_start = time.time()
                print(f"📚 [chat_service] 檢索 RAG 上下文，查詢文本: {last_message[:50]}...")
                rag_context = rag_service.get_rag_context(
                    conversation_id=conversation_id,
                    user_message=last_message
                )
                rag_duration = time.time() - rag_start
                print(f"✅ [chat_service] RAG 檢索完成 (耗時 {rag_duration:.2f}s)")
                backgrounds = rag_context.get("character_background", [])
                print(f"📖 [chat_service] 檢索到 {len(backgrounds)} 條最相關背景資訊")
                for idx, bg in enumerate(backgrounds, 1):
                    print(f"  背景 {idx}: {bg}")

                fewshots = rag_context.get("fewshots", [])
                print(f"💬 [chat_service] 檢索到 {len(fewshots)} 個最相關對話範例")
                for idx, fewshot in enumerate(fewshots, 1):
                    print(f"  最相關範例 {idx}:\n{fewshot}\n")

                summaries = rag_context.get("summaries", [])
                print(f"📜 [chat_service] 檢索到 {len(summaries)} 個最相關歷史摘要")
                for idx, summary in enumerate(summaries, 1):
                    print(f"  最相關摘要 {idx}:\n{summary}\n")

                # 🆕 主角背景檢索結果（列表）
                protagonist_bgs = rag_context.get("protagonist_background", [])
                print(f"👤 [chat_service] 檢索到 {len(protagonist_bgs)} 條最相關主角背景")
                for idx, pb in enumerate(protagonist_bgs, 1):
                    print(f"  主角背景 {idx}: {pb}")

                # 🆕 角色性格描述（固定索引檢索）
                personality = rag_context.get("character_personality", [])
                print(f"🎭 [chat_service] 檢索到 {len(personality)} 條角色性格描述（固定索引）")
                for idx, p in enumerate(personality, 1):
                    print(f"  性格描述 {idx}: {p}")

            # 2. 組裝 system prompt
            print(f"📝 [chat_service] 組裝 system prompt...")
            system_prompt = prompt_builder.build_system_prompt(
                behavior_specs=self.behavior_specs,
                character_info=character_info,
                conversation_history=conversation_history,
                rag_context=rag_context,
                protagonist_name=protagonist_name
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

            # 列印「語言模型實際收到的完整輸入」——依 Ollama messages 順序、每筆全文
            # 這是模型唯一的輸入來源：第一筆為 system prompt 全文，其後為未摘要對話（user/assistant 交錯）
            print(f"\n========== 語言模型實際收到的完整輸入 ==========\n")
            for m in messages:
                print(f"[{m['role']}]")
                print(f"{m['content']}\n")
            print(f"========== 輸入結束 ==========\n")

            # 4. 呼叫 Ollama 生成回應
            ollama_start = time.time()
            print(f"🧠 [chat_service] 呼叫 Ollama 引擎... (開始時間: {ollama_start})")
            import ollama

            # 🆕 【無預設值】Ollama 參數必須由 config 提供，缺鍵直接拋錯
            ollama_model = config.get("ollama.model")
            if ollama_model is None:
                raise ValueError("ollama.model must be set in config")
            ollama_temperature = config.get("ollama.temperature")
            if ollama_temperature is None:
                raise ValueError("ollama.temperature must be set in config")
            ollama_keep_alive = config.get("ollama.keepAlive")  # 模型常駐設定（-1 = 永不卸載）
            if ollama_keep_alive is None:
                raise ValueError("ollama.keepAlive must be set in config")

            response = ollama.chat(
                model=ollama_model,
                messages=messages,
                options={"temperature": ollama_temperature},
                keep_alive=ollama_keep_alive
            )

            ai_response = response['message']['content']
            ollama_duration = time.time() - ollama_start
            print(f"✅ [chat_service] 回應生成成功 (耗時 {ollama_duration:.2f}s)")

            total_duration = time.time() - start_time
            print(f"⏱️  [chat_service] 總耗時: {total_duration:.2f}s")
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

    def generate_summary(
        self,
        conversation_id: str,
        prompt: str
    ) -> dict:
        """
        生成對話摘要

        參數：
            conversation_id: str，聊天室 ID（用於日誌）
            prompt: str，摘要提示詞

        返回：
            dict，{status, data: {summary}}
        """
        try:
            print(f"\n📝 [chat_service] 生成摘要: conversationId={conversation_id}")
            print(f"   提示詞: {prompt[:100]}...")

            # 呼叫 Ollama 生成摘要
            import ollama

            # 🆕 【無預設值】Ollama 參數必須由 config 提供，缺鍵直接拋錯
            ollama_model = config.get("ollama.model")
            if ollama_model is None:
                raise ValueError("ollama.model must be set in config")
            ollama_temperature = config.get("ollama.temperature")
            if ollama_temperature is None:
                raise ValueError("ollama.temperature must be set in config")
            ollama_keep_alive = config.get("ollama.keepAlive")  # 模型常駐設定
            if ollama_keep_alive is None:
                raise ValueError("ollama.keepAlive must be set in config")

            response = ollama.chat(
                model=ollama_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": ollama_temperature},
                keep_alive=ollama_keep_alive
            )

            summary = response['message']['content']

            print(f"✅ [chat_service] 摘要生成成功")
            print(f"   摘要內容: {summary}\n")

            return {
                "status": "success",
                "data": {
                    "summary": summary
                }
            }

        except Exception as e:
            print(f"❌ [chat_service] 生成摘要失敗: {e}")
            return {
                "status": "error",
                "message": str(e)
            }


# 全局實例
chat_service = ChatService()
