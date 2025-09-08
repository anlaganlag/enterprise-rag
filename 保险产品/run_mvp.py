"""
MVP运行脚本 - 完全绕过proxies问题
"""
import os
import sys
from pathlib import Path

# 确保环境变量设置
from dotenv import load_dotenv
load_dotenv()

# 替换vector_store模块
sys.path.insert(0, str(Path(__file__).parent))
import vector_store_bypass as vector_store
sys.modules['vector_store'] = vector_store

# 现在导入main
from main_fixed import build_vector_store, run_qa_system

def main():
    print("=== 保险产品RAG系统 MVP ===\n")
    
    # 验证API密钥
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("错误：未找到OPENAI_API_KEY")
        print("请检查.env文件")
        return
    
    print(f"API密钥已加载（长度：{len(api_key)}）")
    
    try:
        print("\n开始构建向量存储...")
        if build_vector_store():
            print("\n开始运行问答系统...")
            run_qa_system()
            print("\n✓ 执行成功！")
        else:
            print("\n✗ 构建失败")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()