"""
Data Loader - 從各種來源載入資料到向量資料庫
"""

import json
import os
from typing import List, Dict
from pathlib import Path
from .vector_store import vector_store
from config import config


class RAGDataLoader:
    """RAG 資料載入器"""

    def __init__(self, vector_store_instance=None):
        self.vector_store = vector_store_instance or vector_store

    def load_character_profiles_from_json(
        self,
        directory: str = "protocols"
    ) -> bool:
        """
        從 JSON 文件載入角色背景到向量資料庫

        參數：
            directory: 包含角色信息的目錄路徑

        返回：
            是否成功
        """
        try:
            # 確保集合存在
            self.vector_store.create_collection(config.CHARACTER_COLLECTION)

            documents = []
            protocol_path = Path(directory)

            # 尋找所有協議文件
            for json_file in protocol_path.glob("*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # 提取協議內容作為字符串
                    protocol_text = json.dumps(data, ensure_ascii=False, indent=2)

                    doc = {
                        "id": f"{json_file.stem}",
                        "text": protocol_text,
                        "filename": json_file.name,
                        "protocol_id": data.get("protocol_id", "unknown")
                    }
                    documents.append(doc)
                    print(f"  ✓ Loaded: {json_file.name}")
                except Exception as e:
                    print(f"  ✗ Error loading {json_file.name}: {e}")

            if documents:
                self.vector_store.upsert_documents(
                    config.CHARACTER_COLLECTION,
                    documents,
                    metadata_fields=["filename", "protocol_id"]
                )
                return True
            return False

        except Exception as e:
            print(f"Failed to load character profiles: {e}")
            return False

    def load_fewshots(
        self,
        fewshots_data: List[Dict]
    ) -> bool:
        """
        載入 Few-Shot 範例

        參數：
            fewshots_data: Few-Shot 範例列表
                每個範例應包含: id, character_id, input, output

        返回：
            是否成功
        """
        try:
            self.vector_store.create_collection(config.FEWSHOTS_COLLECTION)

            documents = []
            for idx, fewshot in enumerate(fewshots_data):
                # 組合 input 和 output 作為可搜尋的文本
                text = f"Input: {fewshot.get('input', '')}\nOutput: {fewshot.get('output', '')}"

                doc = {
                    "id": fewshot.get("id", f"fewshot_{idx}"),
                    "text": text,
                    "character_id": fewshot.get("character_id"),
                    "input": fewshot.get("input"),
                    "output": fewshot.get("output")
                }
                documents.append(doc)

            self.vector_store.upsert_documents(
                config.FEWSHOTS_COLLECTION,
                documents,
                metadata_fields=["character_id", "input", "output"]
            )
            return True
        except Exception as e:
            print(f"Failed to load fewshots: {e}")
            return False

    def load_conversation_summary(
        self,
        user_id: str,
        character_id: str,
        summary_text: str,
        summary_id: str = None
    ) -> bool:
        """
        載入對話摘要

        參數：
            user_id: 用戶 ID
            character_id: 角色 ID
            summary_text: 摘要內容
            summary_id: 摘要 ID（可選）

        返回：
            是否成功
        """
        try:
            self.vector_store.create_collection(config.CONVERSATION_COLLECTION)

            doc = {
                "id": summary_id or f"{user_id}_{character_id}_{int(__import__('time').time())}",
                "text": summary_text,
                "user_id": user_id,
                "character_id": character_id
            }

            return self.vector_store.upsert_documents(
                config.CONVERSATION_COLLECTION,
                [doc],
                metadata_fields=["user_id", "character_id"]
            )
        except Exception as e:
            print(f"Failed to load conversation summary: {e}")
            return False

    def initialize_all_collections(self) -> bool:
        """初始化所有集合"""
        collections = [
            config.CHARACTER_COLLECTION,
            config.FEWSHOTS_COLLECTION,
            config.CONVERSATION_COLLECTION
        ]

        success = True
        for collection in collections:
            if not self.vector_store.create_collection(collection):
                success = False

        return success


# 全局實例
data_loader = RAGDataLoader()
