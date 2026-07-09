"""
FastAPI 應用主檔
負責註冊路由、配置中間層
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.controllers import reply_controller, rag_controller, chat_controller
from src.rag.vector_store import vector_store
from config import config

# 建立 FastAPI 應用
app = FastAPI(
    title="AI Service",
    version="0.1.0",
    description="Persona Nexus AI 微服務"
)

# 啟動事件：初始化 RAG 集合
@app.on_event("startup")
async def startup_event():
    """在服務啟動時初始化 RAG 集合"""
    print("🚀 [app.py] 初始化 RAG 集合...")
    try:
        # nomic-embed-text 產生 768 維向量
        # 檢查返回值，失敗時拋出異常
        if not vector_store.create_collection(config.CHARACTER_COLLECTION, vector_size=768):
            raise Exception(f"Failed to create collection: {config.CHARACTER_COLLECTION}")

        if not vector_store.create_collection(config.FEWSHOTS_COLLECTION, vector_size=768):
            raise Exception(f"Failed to create collection: {config.FEWSHOTS_COLLECTION}")

        if not vector_store.create_collection("summaries", vector_size=768):
            raise Exception("Failed to create collection: summaries")

        print("✅ [app.py] RAG 集合初始化成功")
    except Exception as e:
        print(f"❌ [app.py] RAG 集合初始化失敗: {e}")

    # 🆕 背景預載 Ollama 模型（不阻塞啟動，載入期間其他端點照常可用）
    # keep_alive=-1 讓模型常駐記憶體，不會因閒置 5 分鐘被 Ollama 自動卸載
    import threading

    def _preload_model():
        import time
        from src.config.config_loader import config as cfg
        model = cfg.get("ollama.model")
        keep_alive = cfg.get("ollama.keepAlive")
        try:
            # 【無預設值】必須由 config 提供
            if model is None:
                raise ValueError("ollama.model must be set in config")
            if keep_alive is None:
                raise ValueError("ollama.keepAlive must be set in config")
            print(f"🔥 [app.py] 開始預載模型: {model}（keep_alive={keep_alive}）...")
            start = time.time()
            import ollama
            # 空 prompt 的 generate 請求 = Ollama 官方的「純載入」慣例，不會實際生成
            ollama.generate(model=model, prompt="", keep_alive=keep_alive)
            print(f"✅ [app.py] 模型預載完成: {model}（耗時 {time.time() - start:.1f}s，常駐記憶體）")
        except Exception as e:
            print(f"⚠️  [app.py] 模型預載失敗（首次聊天時將重新載入）: {e}")

    threading.Thread(target=_preload_model, daemon=True).start()

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],  # API Gateway
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 健康檢查端點
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "ai-service",
        "ollama_url": config.OLLAMA_URL,
        "model": config.OLLAMA_MODEL
    }

# 根路由
@app.get("/")
async def root():
    return {
        "service": "ai-service",
        "version": "0.1.0",
        "endpoints": {
            "health": "GET /health",
            "generate": "POST /api/v1/generate"
        }
    }

# 註冊 Router
app.include_router(reply_controller.router)
app.include_router(rag_controller.router)
app.include_router(chat_controller.router)

print("✅ FastAPI 應用已初始化")
