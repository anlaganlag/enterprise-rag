"""
深入调试OpenAI初始化问题
"""
import os
import sys
import traceback
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

print("=== OpenAI 初始化调试 ===\n")

# 1. 检查库版本
print("1. 检查库版本:")
try:
    import openai
    print(f"   openai版本: {openai.__version__}")
except:
    print("   openai未安装")

try:
    import langchain
    print(f"   langchain版本: {langchain.__version__}")
except:
    print("   langchain未安装")

try:
    import langchain_openai
    print(f"   langchain_openai版本: {langchain_openai.__version__}")
except:
    print("   langchain_openai未安装")

# 2. 测试原生OpenAI客户端
print("\n2. 测试原生OpenAI客户端:")
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("   ✓ 原生OpenAI客户端初始化成功")
except Exception as e:
    print(f"   ✗ 错误: {e}")

# 3. 测试不同的初始化方式
print("\n3. 测试LangChain OpenAIEmbeddings:")

# 方式A: 默认初始化
print("\n   方式A - 默认初始化:")
try:
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or "test-key"
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()
    print("   ✓ 默认初始化成功")
except Exception as e:
    print(f"   ✗ 错误: {e}")
    traceback.print_exc()

# 方式B: 只传model
print("\n   方式B - 只传model参数:")
try:
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("   ✓ 只传model成功")
except Exception as e:
    print(f"   ✗ 错误: {e}")

# 方式C: 传递api_key
print("\n   方式C - 传递api_key参数:")
try:
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    print("   ✓ 传递api_key成功")
except Exception as e:
    print(f"   ✗ 错误: {e}")

# 4. 检查httpx和代理相关
print("\n4. 检查httpx（OpenAI的HTTP客户端）:")
try:
    import httpx
    print(f"   httpx版本: {httpx.__version__}")
    # 检查httpx是否支持proxies参数
    try:
        test_client = httpx.Client(proxies={})
        print("   httpx支持proxies参数")
        test_client.close()
    except:
        print("   httpx不支持proxies参数")
except ImportError:
    print("   httpx未安装")

print("\n" + "="*50)