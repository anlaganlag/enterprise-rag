"""
检查环境变量和代理设置
"""
import os
import sys

def check_environment():
    print("=== 环境检查 ===\n")
    
    # 检查Python版本
    print(f"Python版本: {sys.version}")
    
    # 检查OpenAI API密钥
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"✓ OPENAI_API_KEY已设置 (长度: {len(api_key)})")
        # 检查密钥格式
        if api_key.startswith("sk-"):
            print("  密钥格式正确")
        else:
            print("  ⚠️ 密钥格式可能不正确（应以'sk-'开头）")
    else:
        print("✗ OPENAI_API_KEY未设置")
    
    # 检查代理设置（这可能是问题所在）
    print("\n代理设置:")
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY']
    proxy_found = False
    for var in proxy_vars:
        value = os.environ.get(var)
        if value:
            print(f"  {var} = {value}")
            proxy_found = True
    
    if not proxy_found:
        print("  未检测到代理设置")
    else:
        print("\n⚠️ 检测到代理设置，这可能导致OpenAI客户端问题")
        print("建议临时禁用代理:")
        print("  set HTTP_PROXY=")
        print("  set HTTPS_PROXY=")
    
    # 检查.env文件
    print("\n.env文件:")
    if os.path.exists(".env"):
        print("✓ .env文件存在")
        # 读取并显示非敏感信息
        from dotenv import dotenv_values
        config = dotenv_values(".env")
        print(f"  配置项数量: {len(config)}")
        if "OPENAI_API_KEY" in config:
            print(f"  OPENAI_API_KEY长度: {len(config['OPENAI_API_KEY'])}")
    else:
        print("✗ .env文件不存在")
    
    # 测试OpenAI连接
    print("\n测试OpenAI连接:")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        # 简单测试
        models = client.models.list()
        print("✓ 成功连接到OpenAI API")
    except Exception as e:
        print(f"✗ 连接失败: {e}")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    check_environment()