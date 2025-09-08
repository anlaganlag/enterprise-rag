"""
简单测试OpenAI连接
"""
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 清理代理
for key in list(os.environ.keys()):
    if 'proxy' in key.lower():
        del os.environ[key]

# 测试1：直接使用openai库
print("测试1: OpenAI库")
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input="测试文本"
    )
    print("✓ OpenAI库工作正常")
except Exception as e:
    print(f"✗ OpenAI库错误: {e}")

# 测试2：LangChain OpenAI
print("\n测试2: LangChain OpenAI")
try:
    from langchain_openai import OpenAIEmbeddings
    
    # 方法1：只传API key
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    result = embeddings.embed_query("测试文本")
    print(f"✓ LangChain嵌入工作正常 (维度: {len(result)})")
except Exception as e:
    print(f"✗ LangChain错误: {e}")
    
    # 尝试备选方法
    print("\n尝试备选初始化...")
    try:
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()
        result = embeddings.embed_query("测试文本")
        print(f"✓ 备选方法成功 (维度: {len(result)})")
    except Exception as e2:
        print(f"✗ 备选方法也失败: {e2}")

print("\n完成测试")