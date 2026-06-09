# OPENCODE API配置
from typing import Optional

OPENCODE_BASE_URL = "http://localhost:4096"

# 可选大模型列表（页面「快捷分析」旁的下拉菜单从此读取）
OPENCODE_MODELS = [
    {
        "id": "big-pickle",
        "label": "默认模型",
        "model": "Big Pickle",
        "modelID": "big-pickle",
        "providerID": "opencode",
    },
    {
        "id": "glm-4.7-flash",
        "label": "glm-4.7-flash (本地 Ollama)",
        "model": "glm-4.7-flash:latest",
        "modelID": "glm-4.7-flash:latest",
        "providerID": "ollama",
    },
    {
        "id": "gpt-oss",
        "label": "gpt-oss (本地 Ollama)",
        "model": "gpt-oss:latest",
        "modelID": "gpt-oss:latest",
        "providerID": "ollama",
    },
    {
        "id": "qwen3.5-plus",
        "label": "qwen3.5-plus (第三方)",
        "model": "qwen3.5-plus",
        "modelID": "qwen3.5-plus",
        "providerID": "WenModel",
    },
]

# 默认模型（修改此处即可切换全局默认）
DEFAULT_MODEL_ID = "big-pickle"


def get_model_config(model_id: Optional[str] = None) -> dict:
    """根据 model id 返回 opencode 所需的 model 配置；未匹配时回退到默认模型。"""
    target_id = (model_id or DEFAULT_MODEL_ID).strip()
    for item in OPENCODE_MODELS:
        if item["id"] == target_id or item["modelID"] == target_id:
            return {
                "model": item["model"],
                "modelID": item["modelID"],
                "providerID": item["providerID"],
            }
    for item in OPENCODE_MODELS:
        if item["id"] == DEFAULT_MODEL_ID:
            return {
                "model": item["model"],
                "modelID": item["modelID"],
                "providerID": item["providerID"],
            }
    first = OPENCODE_MODELS[0]
    return {
        "model": first["model"],
        "modelID": first["modelID"],
        "providerID": first["providerID"],
    }


OPENCODE_MODEL = get_model_config(DEFAULT_MODEL_ID)
