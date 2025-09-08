"""
测试.env文件加载
"""
import os
from pathlib import Path

print("=== .env文件加载测试 ===\n")

# 显示当前工作目录
print(f"当前目录: {os.getcwd()}")
print(f".env文件路径: {Path('.env').absolute()}")
print(f".env文件存在: {Path('.env').exists()}")

# 方法1：手动加载.env
print("\n方法1：使用python-dotenv加载")
from dotenv import load_dotenv, dotenv_values

# 显式加载.env文件
load_result = load_dotenv(override=True, verbose=True)
print(f"load_dotenv结果: {load_result}")

# 检查环境变量
api_key = os.getenv("OPENAI_API_KEY")
print(f"OPENAI_API_KEY加载后: {api_key[:10] + '...' if api_key else '未加载'}")

# 方法2：直接读取.env文件内容
print("\n方法2：直接读取.env文件")
env_values = dotenv_values(".env")
print(f"读取到的配置项: {list(env_values.keys())}")
if "OPENAI_API_KEY" in env_values:
    key_value = env_values["OPENAI_API_KEY"]
    print(f"OPENAI_API_KEY值: {key_value[:10]}...{key_value[-5:]}")
    
    # 手动设置环境变量
    os.environ["OPENAI_API_KEY"] = key_value
    print(f"手动设置后: {os.getenv('OPENAI_API_KEY')[:10]}...")

# 方法3：检查文件编码
print("\n方法3：检查文件内容")
with open(".env", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        if "OPENAI_API_KEY" in line:
            print(f"找到OPENAI_API_KEY行: {line.strip()[:30]}...")
            # 检查是否有特殊字符
            if line.startswith("\ufeff"):
                print("⚠️ 检测到BOM标记")
            if "\r" in line:
                print("⚠️ 检测到Windows换行符")

print("\n" + "="*50)