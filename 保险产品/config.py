"""
配置文件 - 保险产品RAG系统
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 项目根目录
BASE_DIR = Path(__file__).parent

# PDF文件路径
PDF_FILES = {
    "aia_flexiachiever": BASE_DIR / "AIA FlexiAchieverSavingsPlan_tc-活享儲蓄計劃.pdf",
    "royal_fortune": BASE_DIR / "RoyalFortune_Product Brochure_EN.pdf"
}

# OpenAI配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

# 模型配置
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4-turbo-preview")

# 文本处理配置
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

# 检索配置
MAX_RETRIEVAL_RESULTS = int(os.getenv("MAX_RETRIEVAL_RESULTS", 5))

# 向量存储配置
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
VECTOR_STORE_DIR.mkdir(exist_ok=True)

# 输出配置
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# 日志配置
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = BASE_DIR / "insurance_rag.log"