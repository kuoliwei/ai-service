"""
RAG Service - 業務邏輯層，協調 RAG 操作
"""

import threading
from typing import List, Dict, Optional
from src.repositories.rag_repository import rag_repository
from config import config


class RAGService:
    """RAG 服務業務邏輯"""

    def __init__(self):
        self.repository = rag_repository
        # 🆕 初始化狀態追蹤：key = conversation_id, value = { status: 'pending' | 'ready' | 'failed', error?: str }
        self.initialization_jobs = {}

    # ===== 健康檢查 =====

    def check_rag_health(self):
        """
        檢查 RAG 資料庫健康狀態（同步檢查）

        拋出：
            Exception 如果 Qdrant 無法連接
        """
        self.repository.vector_store.check_connection()

    # ===== 聊天室初始化（非同步） =====

    def initialize_conversation(
        self,
        conversation_id: str,
        character_id: str,
        background: str,
        fewshots: List[str] = None
    ) -> Dict:
        """
        啟動聊天室 RAG 初始化（真正的背景執行）

        流程：
        1. 【新增】檢查 RAG 資料庫連接（同步，立即）
        2. 如果不可用，立即返回 failed（HTTP 503）
        3. 立即標記狀態為 'pending'
        4. 用 threading 啟動背景任務 _do_initialize_conversation_background()
        5. 立即回傳 202（不等待初始化完成）
        6. 背景線程執行：完成時標記為 'ready'，失敗時標記為 'failed'

        參數：
            conversation_id: 聊天室 ID
            character_id: 角色 ID
            background: 角色背景文本
            fewshots: Few-Shot 範例列表（可選）

        返回：
            { status: 'accepted' }（表示已接受，背景處理中）
            或 { status: 'failed' }（如果 RAG 資料庫不可用）
        """
        print(f"\n📥 [rag_service] 初始化請求: conversationId={conversation_id}")
        print(f"   ├─ characterId: {character_id}")
        print(f"   ├─ background: {len(background or '')} 字")
        print(f"   ├─ fewshots: {len(fewshots or [])} 個")

        # 🆕 【實驗性註解】暫時停用健檢觸發，觀察無健檢時錯誤是否仍能正確傳播
        # # 🆕 【健康檢查】先檢查 RAG 資料庫是否可用（同步、立即）
        # print(f"   ├─ 【健康檢查】檢查 RAG 資料庫連接...")
        # try:
        #     self.repository.check_connection()
        #     print(f"   ├─ ✅ RAG 資料庫連接正常，可以開始初始化")
        # except Exception as e:
        #     # 【被動報錯】如果 RAG 不可用，立即拋錯，不繼續執行
        #     print(f"   ❌ RAG 資料庫不可用: {str(e)}")
        #     self.initialization_jobs[conversation_id] = {
        #         "status": "failed",
        #         "error": str(e)
        #     }
        #     raise Exception(f"RAG health check failed: {str(e)}")

        # 標記狀態為 pending（正在處理）
        self.initialization_jobs[conversation_id] = { "status": "pending" }
        print(f"   ├─ job 狀態標記: pending")

        # 🆕 用 threading 啟動真正的背景線程
        background_thread = threading.Thread(
            target=self._do_initialize_conversation_background,
            args=(conversation_id, character_id, background, fewshots),
            daemon=True  # 設為 daemon，讓程式結束時自動終止
        )
        background_thread.start()
        print(f"   ├─ 背景線程已啟動: {background_thread.name}")

        # 立即回傳：已接受、背景處理中（此時初始化還在背景進行）
        print(f"   ✅ 立即回傳 202 Accepted\n")
        return {
            "status": "accepted",
            "conversation_id": conversation_id,
            "message": "Initialization started, check status for progress"
        }

    def _do_initialize_conversation_background(
        self,
        conversation_id: str,
        character_id: str,
        background: str,
        fewshots: List[str] = None
    ) -> None:
        """
        背景線程執行 RAG 初始化（由 initialize_conversation 的 threading.Thread 啟動）

        此方法在獨立的線程中運行，不會阻塞主請求線程

        參數同上

        返回：
            無（結果存於 self.initialization_jobs[conversation_id]）
        """
        import time
        start_time = time.time()
        try:
            print(f"\n🧠 [rag_service] 【背景線程】初始化開始: conversationId={conversation_id}, 時間={start_time}")

            # 1. 添加角色背景
            if background:
                t1 = time.time()
                print(f"   ├─ 📖 添加背景文本 ({len(background)} 字)... 時間={t1}")
                bg_success = self.repository.add_character_background(
                    conversation_id, character_id, background
                )
                t2 = time.time()
                print(f"   ├─ 背景文本操作耗時: {t2 - t1:.2f}秒")
                if not bg_success:
                    raise Exception("Failed to add character background")
                print(f"   ├─ ✅ 背景文本已存入 Qdrant")
            else:
                print(f"   ├─ ⊘ 無背景文本")

            # 2. 添加 Few-Shots
            if fewshots:
                t3 = time.time()
                print(f"   ├─ 💬 添加 Few-Shot 範例 ({len(fewshots)} 個)... 時間={t3}")
                fs_success = self.repository.add_fewshots(
                    conversation_id, character_id, fewshots
                )
                t4 = time.time()
                print(f"   ├─ Few-Shots 操作耗時: {t4 - t3:.2f}秒")
                if not fs_success:
                    raise Exception("Failed to add few-shots")
                print(f"   ├─ ✅ Few-Shots 已存入 Qdrant")
            else:
                print(f"   ├─ ⊘ 無 Few-Shots")

            # 成功：標記狀態為 ready
            end_time = time.time()
            total_time = end_time - start_time
            self.initialization_jobs[conversation_id] = { "status": "ready" }
            print(f"   ✅ 【背景線程】初始化完成，標記狀態: ready")
            print(f"   ⏱️  【總耗時】{total_time:.2f}秒")

            # 🆕 驗證 RAG 數據（列出已保存的資料）
            print(f"\n   📊 【背景線程】RAG 數據驗證 (conversationId={conversation_id}):")
            rag_data = self.repository.get_conversation_data(conversation_id)

            # 顯示 characters collection
            characters = rag_data.get("characters", [])
            print(f"   ├─ characters collection ({len(characters)} 筆):")
            for idx, char in enumerate(characters):
                text_preview = char.get("text", "")[:10]  # 取前 10 個字
                print(f"   │  ├─ 【聊天室 {conversation_id}】chunk_{char.get('chunk_index', idx)}: {text_preview}")

            # 顯示 fewshots collection
            fewshots = rag_data.get("fewshots", [])
            print(f"   ├─ fewshots collection ({len(fewshots)} 筆):")
            for idx, fewshot in enumerate(fewshots):
                text_preview = fewshot.get("text", "")[:10]  # 取前 10 個字
                print(f"   │  ├─ 【聊天室 {conversation_id}】example_{fewshot.get('index', idx)}: {text_preview}")

            print(f"   └─ ✅ 驗證完成\n")

        except Exception as e:
            # 失敗：標記狀態為 failed，記錄錯誤
            error_msg = str(e)
            print(f"   ❌ 【背景線程】初始化失敗: {error_msg}\n")
            self.initialization_jobs[conversation_id] = {
                "status": "failed",
                "error": error_msg
            }

    def get_initialization_status(self, conversation_id: str) -> Dict:
        """
        查詢聊天室 RAG 初始化狀態

        參數：
            conversation_id: 聊天室 ID

        返回：
            { status: 'pending' | 'ready' | 'failed', error?: str }
        """
        job = self.initialization_jobs.get(conversation_id)

        if not job:
            # 查詢不到 → 未初始化或已完成過（狀態被清除）
            print(f"📊 [rag_service] 查詢狀態: conversationId={conversation_id} → unknown (未找到記錄)")
            return {
                "status": "unknown",
                "message": "No initialization record found"
            }

        status = job.get("status")
        print(f"📊 [rag_service] 查詢狀態: conversationId={conversation_id} → {status}")
        if status == "failed":
            error = job.get("error", "unknown")
            print(f"   ├─ 錯誤: {error}")
        return job

    # ===== 聊天室清理 =====

    def cleanup_conversation(self, conversation_id: str) -> Dict:
        """
        清理聊天室的 RAG 資料（聊天室刪除時調用）

        參數：
            conversation_id: 聊天室 ID

        返回：
            清理結果

        拋出：
            Exception 如果清理失敗（Qdrant 不可用等）
        """
        # 🆕 【被動報錯】不捕獲異常，讓 repository 的異常直接傳播
        self.repository.delete_conversation_data(conversation_id)

        # 也要刪除內存中的 job 狀態記錄
        if conversation_id in self.initialization_jobs:
            del self.initialization_jobs[conversation_id]

        return {
            "status": "success",
            "message": f"Conversation {conversation_id} RAG data cleaned up"
        }

    # ===== 檢索上下文 =====

    def get_rag_context(
        self,
        conversation_id: str,
        user_message: str
    ) -> Dict:
        """
        為對話檢索 RAG 上下文

        參數：
            conversation_id: 聊天室 ID
            user_message: 查詢文本（通常是最後一條訊息）

        返回：
            包含背景、few-shots 和最相關歷史摘要的上下文

        拋出：
            如果 RAG 資料庫（Qdrant）不可用，直接拋異常（被動報錯）
        """
        # 🆕 【被動報錯】不捕獲異常，讓 Qdrant 不可用時直接拋出
        # 這樣 chat_service 才能知道 RAG 失敗，整個流程停止

        # 🆕 所有檢索結果一律回傳列表（數量完全由 config 的 limit 決定，不在代碼中截斷）

        # 1. 搜尋最相關背景資訊
        background = self.repository.search_character_background(
            conversation_id, user_message
        )
        background_text = [b["text"] for b in background] if background else []

        # 2. 搜尋最相關對話範例
        fewshots = self.repository.search_fewshots(
            conversation_id, user_message
        )
        fewshots_text = [fs["text"] for fs in fewshots] if fewshots else []

        # 3. 搜尋最相關歷史摘要
        summaries = self.repository.search_summaries(
            conversation_id, user_message
        )
        summaries_text = [s["text"] for s in summaries] if summaries else []

        # 4. 🆕 搜尋最相關主角（主人公）背景資訊（未設定時自然為空）
        protagonist = self.repository.search_protagonist_background(
            conversation_id, user_message
        )
        protagonist_text = [p["text"] for p in protagonist] if protagonist else []

        # 5. 🆕 搜尋角色性格描述（固定「性格」語意索引，不用當前對話——性格錨定不隨劇情漂移）
        personality = self.repository.search_character_personality(conversation_id)
        personality_text = [p["text"] for p in personality] if personality else []

        return {
            "status": "success",
            "character_background": background_text,
            "fewshots": fewshots_text,
            "summaries": summaries_text,
            "protagonist_background": protagonist_text,
            "character_personality": personality_text
        }

    # ===== 主角（主人公）人設管理 =====

    def update_protagonist_background(
        self,
        conversation_id: str,
        background: str
    ) -> Dict:
        """
        更新聊天室的主角背景（先刪舊切片再存新切片）

        使用者在聊天室編輯並儲存主角人設時，由 chat-service 呼叫

        參數：
            conversation_id: 聊天室 ID
            background: 主角背景文本（空字串 = 清除）

        返回：
            更新結果

        拋出：
            Exception 如果 Qdrant 操作失敗——【被動報錯】不捕獲
        """
        print(f"👤 [rag_service] 更新主角背景: conversationId={conversation_id}, {len(background or '')} 字")

        self.repository.replace_protagonist_background(conversation_id, background or "")

        return {
            "status": "success",
            "conversation_id": conversation_id,
            "message": "Protagonist background updated"
        }

    # ===== 摘要管理 =====

    def add_summary(
        self,
        conversation_id: str,
        summary: str
    ) -> Dict:
        """
        將對話摘要存入向量資料庫

        參數：
            conversation_id: 聊天室 ID
            summary: 摘要文本

        返回：
            存儲結果

        拋出：
            Exception 如果存儲失敗（RAG 不可用或其他錯誤）
        """
        print(f"💾 [rag_service] 存入摘要: conversationId={conversation_id}")
        print(f"   摘要內容: {summary}\n")

        # 🆕 【被動報錯】不捕獲異常，讓 repository 的異常直接傳播
        # repository 回傳 summary_id（Qdrant point id），一併回報給 chat-service
        summary_id = self.repository.add_summary(
            conversation_id, summary
        )

        return {
            "status": "success",
            "conversation_id": conversation_id,
            "summary_id": summary_id,
            "message": "Summary added to vector database"
        }

    def delete_summaries(
        self,
        conversation_id: str,
        summary_ids: List[str]
    ) -> Dict:
        """
        按 summary_id 精準刪除摘要（訊息回溯刪除時，清除涵蓋被刪訊息的記憶）

        參數：
            conversation_id: 聊天室 ID（僅用於日誌）
            summary_ids: 摘要 ID 列表（= Qdrant point id）

        返回：
            刪除結果

        拋出：
            Exception 如果刪除失敗——【被動報錯】不捕獲
        """
        print(f"🗑️ [rag_service] 刪除摘要: conversationId={conversation_id}, 共 {len(summary_ids)} 份")
        print(f"🐛 [DEBUG] ----- 將從 Qdrant summaries collection 刪除的 points -----")
        for idx, sid in enumerate(summary_ids, 1):
            print(f"🐛 [DEBUG]   {idx}. point_id={sid}")

        self.repository.delete_summaries(summary_ids)

        print(f"🐛 [DEBUG] ✅ Qdrant 刪除完成: {len(summary_ids)} 個 points 已移除")

        return {
            "status": "success",
            "conversation_id": conversation_id,
            "deleted_count": len(summary_ids),
            "message": "Summaries deleted from vector database"
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
