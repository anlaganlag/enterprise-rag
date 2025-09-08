"""
向量存储模块 - 管理文档向量化和检索
"""
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from tqdm import tqdm

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """向量存储管理器"""
    
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        """初始化向量存储管理器"""
        self.embedding_model = embedding_model
        # 修复：显式设置OpenAI API密钥，避免proxies参数问题
        import os
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.vector_store = None
        
    def create_documents(self, indexed_documents: List[Dict[str, Any]]) -> List[Document]:
        """将索引文档转换为LangChain Document对象"""
        documents = []
        
        for doc in indexed_documents:
            document = Document(
                page_content=doc["content"],
                metadata=doc.get("metadata", {})
            )
            documents.append(document)
            
        logger.info(f"创建了 {len(documents)} 个Document对象")
        return documents
    
    def build_vector_store(self, documents: List[Document]) -> FAISS:
        """构建FAISS向量存储"""
        logger.info(f"开始构建向量存储，共 {len(documents)} 个文档")
        
        # 批量处理以避免API限制
        batch_size = 100
        all_vectors = []
        
        for i in tqdm(range(0, len(documents), batch_size), desc="向量化文档"):
            batch = documents[i:i + batch_size]
            
            if i == 0:
                # 创建初始向量存储
                self.vector_store = FAISS.from_documents(batch, self.embeddings)
            else:
                # 添加到现有向量存储
                texts = [doc.page_content for doc in batch]
                metadatas = [doc.metadata for doc in batch]
                self.vector_store.add_texts(texts, metadatas)
                
        logger.info("向量存储构建完成")
        return self.vector_store
    
    def save_vector_store(self, save_path: Path):
        """保存向量存储到本地"""
        if self.vector_store is None:
            raise ValueError("向量存储尚未构建")
            
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # FAISS保存
        self.vector_store.save_local(str(save_path))
        logger.info(f"向量存储已保存到: {save_path}")
    
    def load_vector_store(self, load_path: Path) -> FAISS:
        """从本地加载向量存储"""
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"向量存储文件不存在: {load_path}")
            
        self.vector_store = FAISS.load_local(
            str(load_path), 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info(f"向量存储已从 {load_path} 加载")
        return self.vector_store
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """执行相似度搜索"""
        if self.vector_store is None:
            raise ValueError("向量存储尚未构建或加载")
            
        results = self.vector_store.similarity_search(query, k=k)
        logger.info(f"查询: '{query[:50]}...' 返回 {len(results)} 个结果")
        return results
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """执行带分数的相似度搜索"""
        if self.vector_store is None:
            raise ValueError("向量存储尚未构建或加载")
            
        results = self.vector_store.similarity_search_with_score(query, k=k)
        logger.info(f"查询: '{query[:50]}...' 返回 {len(results)} 个结果")
        return results
    
    def get_retriever(self, search_kwargs: Optional[Dict] = None):
        """获取检索器对象"""
        if self.vector_store is None:
            raise ValueError("向量存储尚未构建或加载")
            
        if search_kwargs is None:
            search_kwargs = {"k": 5}
            
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
    
    def add_documents(self, new_documents: List[Document]):
        """向现有向量存储添加新文档"""
        if self.vector_store is None:
            raise ValueError("向量存储尚未构建")
            
        texts = [doc.page_content for doc in new_documents]
        metadatas = [doc.metadata for doc in new_documents]
        
        self.vector_store.add_texts(texts, metadatas)
        logger.info(f"添加了 {len(new_documents)} 个新文档到向量存储")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取向量存储统计信息"""
        if self.vector_store is None:
            return {"status": "未初始化"}
            
        # 获取文档数量
        # 注意: FAISS没有直接的文档计数方法，这里使用近似方法
        stats = {
            "status": "已初始化",
            "embedding_model": self.embedding_model,
            "index_size": self.vector_store.index.ntotal if hasattr(self.vector_store, 'index') else "未知"
        }
        
        return stats


if __name__ == "__main__":
    # 测试代码
    from config import VECTOR_STORE_DIR, EMBEDDING_MODEL
    from pdf_processor import PDFProcessor
    from config import PDF_FILES, CHUNK_SIZE, CHUNK_OVERLAP
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 处理PDF获取文档
    processor = PDFProcessor()
    pdf_paths = list(PDF_FILES.values())
    documents = processor.process_multiple_pdfs(pdf_paths)
    indexed_docs = processor.prepare_documents_for_indexing(
        documents,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )
    
    # 初始化向量存储管理器
    vector_manager = VectorStoreManager(embedding_model=EMBEDDING_MODEL)
    
    # 创建Document对象
    langchain_docs = vector_manager.create_documents(indexed_docs)
    
    # 构建向量存储
    vector_store = vector_manager.build_vector_store(langchain_docs)
    
    # 保存向量存储
    vector_manager.save_vector_store(VECTOR_STORE_DIR / "insurance_vectors")
    
    # 测试检索
    test_query = "What is the minimum premium?"
    results = vector_manager.similarity_search(test_query, k=3)
    
    print(f"\n查询: {test_query}")
    print(f"找到 {len(results)} 个相关文档:")
    for i, doc in enumerate(results, 1):
        print(f"\n结果 {i}:")
        print(f"内容: {doc.page_content[:200]}...")
        print(f"来源: {doc.metadata.get('source', '未知')}")