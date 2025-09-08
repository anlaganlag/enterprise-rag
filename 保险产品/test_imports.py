"""
测试所有必要的导入是否正常
"""

def test_imports():
    print("测试导入...")
    
    try:
        # PDF处理
        import PyPDF2
        print("✓ PyPDF2")
        
        import pdfplumber
        print("✓ pdfplumber")
        
        # LangChain
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        print("✓ langchain_openai")
        
        from langchain_community.vectorstores import FAISS
        print("✓ langchain_community.vectorstores")
        
        from langchain_core.documents import Document
        print("✓ langchain_core.documents")
        
        from langchain.chains import RetrievalQA
        print("✓ langchain.chains")
        
        # 其他
        import openai
        print("✓ openai")
        
        import faiss
        print("✓ faiss")
        
        import tiktoken
        print("✓ tiktoken")
        
        print("\n所有导入成功！可以运行主程序了。")
        return True
        
    except ImportError as e:
        print(f"\n✗ 导入失败: {e}")
        return False

if __name__ == "__main__":
    test_imports()