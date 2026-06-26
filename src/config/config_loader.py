"""
Config Loader - 從 JSON 檔案讀取配置
"""

import json
from pathlib import Path


class ConfigLoader:
    """配置加載器"""

    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """從 JSON 檔案加載配置"""
        try:
            config_path = Path(__file__).parent / "config.json"

            if not config_path.exists():
                raise FileNotFoundError(f"找不到配置檔案: {config_path}")

            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = json.load(f)

            print(f"✅ [config_loader] 配置已加載: {config_path}")

        except Exception as e:
            print(f"❌ [config_loader] 加載配置失敗: {e}")
            raise

    def get(self, key_path: str, default=None):
        """
        獲取配置值

        參數：
            key_path: str，使用點號分隔的路徑 (e.g., "ollama.model")
            default: 預設值

        返回：
            配置值，或預設值
        """
        keys = key_path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default

        return value if value is not None else default


# 全局實例
config = ConfigLoader()
