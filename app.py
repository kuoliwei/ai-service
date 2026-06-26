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
        vector_store.create_collection(config.CHARACTER_COLLECTION, vector_size=768)
        vector_store.create_collection(config.FEWSHOTS_COLLECTION, vector_size=768)
        vector_store.create_collection(config.CONVERSATION_COLLECTION, vector_size=768)
        print("✅ [app.py] RAG 集合初始化成功")
    except Exception as e:
        print(f"❌ [app.py] RAG 集合初始化失敗: {e}")

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
