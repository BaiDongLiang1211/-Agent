import os

# 1. 从环境变量中读取配置
# os.getenv("变量名") 会读取你在 .bashrc 中 export 的值
# 如果读取不到，第二个参数是默认值

# 读取你在 .bashrc 中配置的 LLM_API_KEY
os.environ["OPENAI_API_KEY"] = os.getenv("LLM_API_KEY") 

# 如果你使用的是 DeepSeek，基础路径通常如下，或者你也可以在 .bashrc 中 export 后读取
os.environ["OPENAI_API_BASE"] = os.getenv("LLM_BASE_URL")

# 2. 模型与数据配置
# 这里修改为 deepseek-chat
MODEL_NAME = "deepseek-chat"  
DATA_FILENAME = "titanic_cleaned.csv"

