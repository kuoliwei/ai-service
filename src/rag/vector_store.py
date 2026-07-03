"""
Vector Store - Qdrant 向量資料庫操作層
"""

from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from config import config
from .embedder import embedder


class QdrantVectorStore:
    """Qdrant 向量資料庫操作"""

    def __init__(self, url: str = None, api_key: str = None):
        self.url = url or config.QDRANT_URL
        self.api_key = api_key or config.QDRANT_API_KEY
        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key
        )
        self.embedder = embedder

    def create_collection(
        self,
        collection_name: str,
        vector_size: int = 384,
        distance: Distance = Distance.COSINE
    ) -> bool:
        """
        建立集合（若不存在）

        參數：
            collection_name: 集合名稱
            vector_size: 向量維度
            distance: 距離度量方式

        返回：
            是否成功建立
        """
        try:
            # 檢查集合是否已存在
            try:
                self.client.get_collection(collection_name)
                print(f"✓ Collection '{collection_name}' already exists")
                return True
            except:
                pass

            # 建立新集合
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                )
            )
            print(f"✓ Collection '{collection_name}' created successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to create collection: {e}")
            return False

    def upsert_documents(
        self,
        collection_name: str,
        documents: List[Dict],
        metadata_fields: List[str] = None
    ) -> bool:
        """
        插入或更新文檔

        參數：
            collection_name: 集合名稱
            documents: 文檔列表，每個文檔包含 'id', 'text' 等字段
            metadata_fields: 要保存的元數據字段

        返回：
            是否成功
        """
        import time
        upsert_start = time.time()
        try:
            print(f"\n📤 [vector_store] upsert_documents 開始: collection={collection_name}, 文檔數={len(documents)}, 時間={upsert_start}")
            embedding_time = 0
            points = []
            for idx, doc in enumerate(documents):
                # 向量化文本
                embed_start = time.time()
                embedding = self.embedder.embed_text(doc.get("text", ""))
                embed_end = time.time()
                embedding_time += (embed_end - embed_start)
                if idx == 0 or (idx + 1) % max(1, len(documents) // 3) == 0:
                    print(f"   ├─ embedding 進度: {idx + 1}/{len(documents)}, 單次耗時={embed_end - embed_start:.2f}秒")

                # 準備元數據（payload）
                payload = {"text": doc.get("text", "")}
                if metadata_fields:
                    for field in metadata_fields:
                        if field in doc:
                            payload[field] = doc[field]
                else:
                    # 保存所有非 'text' 的字段
                    for key, value in doc.items():
                        if key != "text" and key != "id":
                            payload[key] = value

                # 建立 Point
                point = PointStruct(
                    id=doc.get("id", idx),
                    vector=embedding,
                    payload=payload
                )
                points.append(point)

            # 批量上傳
            upsert_db_start = time.time()
            print(f"   ├─ 【embedding 總耗時】{embedding_time:.2f}秒")
            print(f"   ├─ 【即將連接 Qdrant】時間={upsert_db_start}")
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            upsert_db_end = time.time()
            print(f"   ├─ 【Qdrant upsert 耗時】{upsert_db_end - upsert_db_start:.2f}秒")
            total_upsert = time.time() - upsert_start
            print(f"✓ Upserted {len(points)} documents to '{collection_name}', 總耗時={total_upsert:.2f}秒")
            return True
        except Exception as e:
            error_time = time.time() - upsert_start
            print(f"✗ Failed to upsert documents: {e}, 耗時={error_time:.2f}秒")
            return False

    def search(
        self,
        collection_name: str,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.0,
        filters: Dict = None
    ) -> List[Dict]:
        """
        搜尋相似文檔

        參數：
            collection_name: 集合名稱
            query: 查詢文本
            limit: 返回結果數量
            score_threshold: 分數閾值
            filters: 過濾條件字典 (e.g., {"conversation_id": "conv_123"})

        返回：
            搜尋結果列表
        """
        try:
            # 向量化查詢
            query_embedding = self.embedder.embed_text(query)

            # 構建過濾條件
            query_filter = None
            if filters:
                must_conditions = []
                for key, value in filters.items():
                    must_conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
                if must_conditions:
                    query_filter = Filter(must=must_conditions)

            print(f"🔍 [vector_store] 搜尋: collection={collection_name}, query={query}, filters={filters}")

            # 搜尋 - 使用 query_points
            results = self.client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter
            ).points

            print(f"📊 [vector_store] 搜尋結果: {len(results)} 筆")

            # 格式化結果
            documents = []
            for result in results:
                doc = {
                    "id": result.id,
                    "score": result.score,
                    **result.payload
                }
                documents.append(doc)

            return documents
        except Exception as e:
            # 🆕 【被動報錯】不捕獲異常，直接拋出
            # 這樣 Qdrant 不可用時，整個流程會停止
            print(f"✗ Search failed: {e}")
            raise  # ← 重新拋出異常，讓上層知道出錯了

    def delete_collection(self, collection_name: str) -> bool:
        """刪除集合"""
        try:
            self.client.delete_collection(collection_name)
            print(f"✓ Collection '{collection_name}' deleted")
            return True
        except Exception as e:
            print(f"✗ Failed to delete collection: {e}")
            return False

    def check_connection(self):
        """
        檢查 Qdrant 連接狀態

        拋出：
            Exception 如果 Qdrant 無法連接
        """
        try:
            # 嘗試獲取集合列表（輕量級操作）
            self.client.get_collections()
            print(f"✅ [vector_store] Qdrant 連接正常")
        except Exception as e:
            # 🆕 【被動報錯】直接拋異常，而不是返回 False
            error_msg = f"Qdrant connection failed: {str(e)}"
            print(f"❌ [vector_store] Qdrant 連接失敗: {error_msg}")
            raise Exception(error_msg)

    def list_collections(self) -> List[str]:
        """列出所有集合"""
        try:
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            return collection_names
        except Exception as e:
            print(f"✗ Failed to list collections: {e}")
            return []


# 全局實例
vector_store = QdrantVectorStore()
