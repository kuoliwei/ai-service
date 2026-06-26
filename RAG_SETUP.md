# RAG 系統設置指南

## 概述

AI Service 現已集成 RAG（檢索增強生成）系統，使用 **LangChain + Qdrant + Ollama** 的技術棧。

### 核心功能

1. **角色背景檢索** - 從向量資料庫動態檢索角色背景信息
2. **Few-Shot 範例檢索** - 基於用戶訊息的語義相似性檢索相關範例
3. **對話歷史摘要** - 管理和檢索對話摘要作為長期上下文

---

## 架構

```
┌─────────────────────────────────────────┐
│          API Endpoints (FastAPI)        │
├─────────────────────────────────────────┤
│  /api/v1/generate (對話)                 │
│  /api/v1/rag/* (RAG 管理)                │
├─────────────────────────────────────────┤
│       PromptService (RAG 整合)           │
├─────────────────────────────────────────┤
│         RAGRetriever (檢索邏輯)          │
├─────────────────────────────────────────┤
│    QdrantVectorStore (向量資料庫)        │
├─────────────────────────────────────────┤
│    OllamaEmbedder (向量化)               │
├─────────────────────────────────────────┤
│  Qdrant 服務 + Ollama 服務               │
└─────────────────────────────────────────┘
```

---

## 設置步驟

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 配置環境變數

複製 `.env.example` 到 `.env`：

```bash
cp .env.example .env
```

編輯 `.env` 文件配置：

```env
# Ollama 配置
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama2
OLLAMA_EMBED_MODEL=nomic-embed-text

# Qdrant 配置
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=  # 如果 Qdrant 使用認證

# RAG 配置
RAG_ENABLED=true
RAG_TOP_K=3
```

### 3. 啟動外部服務

#### 啟動 Ollama

```bash
# 如果已安裝 Ollama（在另一個終端）
ollama serve

# 拉取 embedding 模型（如果還沒有）
ollama pull nomic-embed-text
```

#### 啟動 Qdrant

**選項 A：Docker（推薦）**

```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

**選項 B：本地安裝**

```bash
# 詳見 https://qdrant.tech/documentation/
```

### 4. 初始化 RAG 系統

運行初始化腳本：

```bash
python scripts/init_rag.py
```

或通過 API：

```bash
curl -X POST http://localhost:6001/api/v1/rag/initialize
```

### 5. 啟動 AI Service

```bash
python main.py
# 或
uvicorn main:app --reload
```

---

## API 端點

### RAG 管理端點

#### 初始化 RAG 系統
```
POST /api/v1/rag/initialize
```

#### 載入協議文件
```
POST /api/v1/rag/load-protocols
```

#### 載入 Few-Shot 範例
```
POST /api/v1/rag/load-fewshots
Content-Type: application/json

[
  {
    "id": "fewshot_1",
    "character_id": "char_001",
    "input": "早安",
    "output": "早安！希望你今天過得愉快。"
  },
  ...
]
```

#### 添加對話摘要
```
POST /api/v1/rag/add-conversation-summary
?user_id=user_001&character_id=char_001&summary_text=...
```

#### 列出集合
```
GET /api/v1/rag/collections
```

#### 刪除集合
```
DELETE /api/v1/rag/collections/{collection_name}
```

#### 獲取 RAG 系統狀態
```
GET /api/v1/rag/status
```

### 對話端點（已整合 RAG）

```
POST /api/v1/generate
Content-Type: application/json

{
  "character_id": "char_001",
  "user_id": "user_001",
  "message": "你好",
  "use_rag": true  // 啟用 RAG 檢索
}
```

---

## 使用範例

### 初始化流程

1. **確保服務運行**
   - Ollama: http://localhost:11434
   - Qdrant: http://localhost:6333
   - AI Service: http://localhost:6001

2. **初始化系統**
   ```bash
   curl -X POST http://localhost:6001/api/v1/rag/initialize
   ```

3. **檢查狀態**
   ```bash
   curl http://localhost:6001/api/v1/rag/status
   ```

### 對話流程

1. **帶 RAG 的對話生成**
   ```bash
   curl -X POST http://localhost:6001/api/v1/generate \
     -H "Content-Type: application/json" \
     -d '{
       "character_id": "persona_roleplay_v2",
       "user_id": "user_001",
       "message": "早安",
       "use_rag": true
     }'
   ```

---

## 配置說明

| 配置項 | 預設值 | 說明 |
|--------|--------|------|
| `RAG_ENABLED` | true | 是否啟用 RAG 系統 |
| `RAG_TOP_K` | 3 | 每次檢索返回的結果數量 |
| `RAG_CHUNK_SIZE` | 500 | 文本分塊大小 |
| `RAG_CHUNK_OVERLAP` | 100 | 分塊重疊大小 |
| `OLLAMA_EMBED_MODEL` | nomic-embed-text | Embedding 模型 |

---

## 故障排除

### 連接 Qdrant 失敗

```
Error: Failed to connect to Qdrant at http://localhost:6333
```

**解決方案**：
1. 確認 Qdrant 已啟動
2. 檢查 `QDRANT_URL` 配置
3. 檢查防火牆設置

### Embedding 失敗

```
Error: Failed to embed text
```

**解決方案**：
1. 確認 Ollama 已啟動
2. 確認模型已拉取：`ollama pull nomic-embed-text`
3. 檢查 `OLLAMA_URL` 和 `OLLAMA_EMBED_MODEL`

### 協議文件未載入

```
⚠️  未能載入任何協議文件
```

**解決方案**：
1. 檢查 `protocols/` 目錄是否存在
2. 確認目錄中有 `.json` 文件
3. 確認 JSON 格式有效

---

## 向量資料庫結構

### 集合定義

#### `characters`
- **用途**：儲存角色背景信息
- **字段**：id, text, filename, protocol_id

#### `fewshots`
- **用途**：儲存 Few-Shot 範例
- **字段**：id, text, character_id, input, output

#### `conversation_summaries`
- **用途**：儲存對話摘要
- **字段**：id, text, user_id, character_id

---

## 最佳實踐

1. **定期更新對話摘要**
   - 每隔 N 條對話，生成新的摘要並存入向量資料庫
   - 有助於長期上下文的維持

2. **優化 Few-Shots**
   - 為每個角色提供多樣化的 Few-Shot 範例
   - 定期評估和更新範例效果

3. **監控 RAG 性能**
   - 記錄檢索的相關性得分
   - 調整 `RAG_TOP_K` 和其他參數

4. **備份向量資料庫**
   - 定期備份 Qdrant 數據
   - 確保數據持久性

---

## 進階配置

### 自訂 Embedding 模型

編輯 `.env`：

```env
OLLAMA_EMBED_MODEL=all-minilm  # 更輕量的模型
```

支持的模型（需先 pull）：
- `nomic-embed-text` - 推薦，768 維
- `all-minilm` - 輕量，384 維
- `mistral-embed` - 高性能，1024 維

### 自訂檢索策略

在 `PromptService` 或 `RAGRetriever` 中修改：

```python
# 改變檢索的結果數量
rag_context = retriever.build_rag_context(
    character_id, user_id, user_message,
    top_k=5  # 返回 5 條結果
)
```

---

## 常見問題

**Q: RAG 會影響回應速度嗎？**

A: 會有輕微影響（通常 100-300ms）。可以通過調整 `RAG_TOP_K` 或優化 Qdrant 索引來改善。

**Q: 如何禁用 RAG？**

A: 設置 `RAG_ENABLED=false` 或在請求時指定 `use_rag=false`。

**Q: 支持多用戶多角色嗎？**

A: 完全支持。RAG 系統通過 `user_id` 和 `character_id` 隔離數據。

---

## 更多資源

- [LangChain 文檔](https://python.langchain.com/)
- [Qdrant 文檔](https://qdrant.tech/documentation/)
- [Ollama 文檔](https://ollama.ai/)
