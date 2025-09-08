"""
检查httpx和OpenAI的代理问题
"""
import os
import sys

print("=== 检查代理设置 ===\n")

# 1. 检查所有环境变量中的代理
print("1. 环境变量中的代理设置:")
for key, value in os.environ.items():
    if 'proxy' in key.lower() or 'PROXY' in key:
        print(f"   {key} = {value}")

# 2. 检查httpx默认配置
print("\n2. 检查httpx配置:")
try:
    import httpx
    print(f"   httpx版本: {httpx.__version__}")
    
    # 尝试创建不带代理的客户端
    try:
        client = httpx.Client()
        print("   ✓ httpx Client创建成功（无代理）")
        client.close()
    except Exception as e:
        print(f"   ✗ httpx Client创建失败: {e}")
        
except ImportError:
    print("   httpx未安装")

# 3. 找到问题根源
print("\n3. 查找问题根源:")
print("   检查是否有全局httpx配置...")

# 检查是否有httpx配置文件
import pathlib
possible_configs = [
    pathlib.Path.home() / ".httpx.toml",
    pathlib.Path.home() / ".config" / "httpx" / "config.toml",
    pathlib.Path.cwd() / "httpx.toml"
]

for config_path in possible_configs:
    if config_path.exists():
        print(f"   找到配置文件: {config_path}")

# 4. 检查OpenAI的问题
print("\n4. OpenAI客户端测试:")
try:
    # 尝试直接导入
    import openai
    print(f"   openai版本: {openai.__version__}")
    
    # 检查OpenAI是否在某处设置了代理
    if hasattr(openai, '_default_proxies'):
        print(f"   OpenAI默认代理: {openai._default_proxies}")
    
    # 尝试monkey patch修复
    print("\n5. 尝试修复:")
    
    # 方法1: 修改httpx.Client的初始化
    original_httpx_client = httpx.Client.__init__
    
    def patched_httpx_client_init(self, **kwargs):
        # 移除proxies参数
        kwargs.pop('proxies', None)
        original_httpx_client(self, **kwargs)
    
    httpx.Client.__init__ = patched_httpx_client_init
    print("   ✓ 已修补httpx.Client")
    
    # 现在测试OpenAI
    from openai import OpenAI
    try:
        client = OpenAI(api_key="test-key")
        print("   ✓ OpenAI客户端创建成功（使用修补）")
    except Exception as e:
        print(f"   ✗ 仍然失败: {e}")
        
except Exception as e:
    print(f"   错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)