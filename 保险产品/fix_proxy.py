"""
修复代理问题的脚本
"""
import os
import sys

def fix_proxy_and_run():
    """清理代理设置并运行主程序"""
    
    print("清理可能干扰的环境变量...")
    
    # 清理所有代理相关的环境变量
    proxy_vars = [
        'HTTP_PROXY', 'HTTPS_PROXY', 
        'http_proxy', 'https_proxy', 
        'ALL_PROXY', 'all_proxy',
        'NO_PROXY', 'no_proxy'
    ]
    
    for var in proxy_vars:
        if var in os.environ:
            print(f"删除环境变量: {var}")
            del os.environ[var]
    
    # 确保.env文件被正确加载
    from dotenv import load_dotenv
    load_dotenv(override=True)
    
    # 验证API密钥
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("错误：OPENAI_API_KEY未设置")
        print("请确保.env文件包含有效的API密钥")
        return False
    
    print(f"API密钥已加载（长度：{len(api_key)}）")
    
    # 导入并运行主程序
    try:
        from main import build_vector_store, run_qa_system
        
        print("\n开始构建向量存储...")
        if build_vector_store():
            print("\n开始运行问答系统...")
            run_qa_system()
            return True
        else:
            print("构建向量存储失败")
            return False
            
    except Exception as e:
        print(f"运行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== 保险产品RAG系统 - 代理修复版 ===\n")
    
    if fix_proxy_and_run():
        print("\n✓ 执行成功！")
    else:
        print("\n✗ 执行失败，请检查错误信息")