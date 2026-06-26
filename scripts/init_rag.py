"""
RAG 初始化腳本
檢查 RAG 系統連接狀態
"""

import sys
import os

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.vector_store import vector_store
from config import config


def main():
    print("=" * 60)
    print("  📋 RAG 系統狀態檢查")
    print("=" * 60)
    print()

    try:
        # 檢查 Qdrant 連接
        collections = vector_store.list_collections()
        print("✅ Qdrant 連接成功")
        print(f"  現有集合: {collections if collections else '(無)'}")

        print()
        print(f"  配置:")
        print(f"    Qdrant URL: {config.QDRANT_URL}")
        print(f"    Ollama URL: {config.OLLAMA_URL}")
        print(f"    Embedding Model: {config.OLLAMA_EMBED_MODEL}")

        print()
        print("=" * 60)
        print("  ✅ RAG 系統已就緒，等待手動操作")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"❌ 連接失敗: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
