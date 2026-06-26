"""
AI Service 啟動檔
使用方式: python main.py 或 uvicorn main:app --reload
"""

import uvicorn
from app import app
from config import config

if __name__ == "__main__":
    print("=" * 60)
    print("  🤖 AI Service 啟動中...")
    print(f"  主機: {config.HOST}")
    print(f"  連接埠: {config.PORT}")
    print(f"  Ollama URL: {config.OLLAMA_URL}")
    print(f"  Model: {config.OLLAMA_MODEL}")
    print("=" * 60)

    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT
    )
