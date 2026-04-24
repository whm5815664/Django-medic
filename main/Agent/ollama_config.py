# Ollama API配置
OLLAMA_BASE_URL = "http://localhost:11435"
OLLANA_MODEL_NAME = "deepseek-r1:1.5b"


# OPENCODE API配置
OPENCODE_BASE_URL = "http://localhost:4096"

# 本地ollama部署
#OPENCODE_MODEL = {'model': 'glm-4.7-flash:latest', 'modelID': 'glm-4.7-flash:latest', 'providerID': 'ollama'}
#OPENCODE_MODEL = {'model': 'gpt-oss:latest', 'modelID': 'gpt-oss:latest', 'providerID': 'ollama'}
#OPENCODE_MODEL = {'model':'qwen3.5:0.8b', 'modelID': 'qwen3.5:0.8b', 'providerID': 'ollama'}

# 免费api
#OPENCODE_MODEL = {'model': 'Big Pickle', 'modelID': 'big-pickle', 'providerID': 'opencode'}
#OPENCODE_MODEL = {'model':'MiniMax M2.5 Free', 'modelID': 'minimax-m2.5-free', 'providerID': 'opencode'}

# 第三方咸鱼api http://ai.wenmodel.com/console(https://m.tb.cn/h.ioaPOiq?tk=QGsd5gSbXZ)
OPENCODE_MODEL = {'model':'qwen3.5-plus', 'modelID': 'qwen3.5-plus', 'providerID': 'WenModel'}