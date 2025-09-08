"""
向量存储模块 - 绕过proxies问题的版本
"""
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from langchain_core.documents import Document
from tqdm import tqdm
import os

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """向量存储管理器 - 修复版"""
    
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        """初始化向量存储管理器"""
        self.embedding_model = embedding_model
        self.vector_store = None
        
        # 延迟初始化embeddings，避免初始化时的问题
        self._embeddings = None
        
    @property
    def embeddings(self):
        """延迟加载embeddings"""
        if self._embeddings is None:
            # 确保环境变量设置
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                from dotenv import load_dotenv
                load_dotenv()
                api_key = os.getenv("OPENAI_API_KEY")
            
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found")
            
            # 设置环境变量（某些版本的langchain需要这个）
            os.environ["OPENAI_API_KEY"] = api_key
            
            # 尝试不同的初始化方式
            try:
                # 方式1：最简单的初始化
                from langchain_openai import OpenAIEmbeddings
                self._embeddings = OpenAIEmbeddings()
                logger.info("使用默认初始化成功")
            except Exception as e1:
                try:
                    # 方式2：只传model
                    from langchain_openai import OpenAIEmbeddings
                    self._embeddings = OpenAIEmbeddings(model=self.embedding_model)
                    logger.info("使用model参数初始化成功")
                except Exception as e2:
                    # 方式3：直接使用OpenAI客户端
                    logger.warning(f"LangChain初始化失败，使用备用方案: {e2}")
                    self._embeddings = self._create_fallback_embeddings()
        
        return self._embeddings
    
    def _create_fallback_embeddings(self):
        """创建备用的embeddings对象"""
        from openai import OpenAI
        
        class FallbackEmbeddings:
            def __init__(self, model="text-embedding-3-small"):
                self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                self.model = model
            
            def embed_documents(self, texts):
                """嵌入多个文档"""
                embeddings = []
                for text in texts:
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=text
                    )
                    embeddings.append(response.data[0].embedding)
                return embeddings
            
            def embed_query(self, text):
                """嵌入单个查询"""
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text
                )
                return response.data[0].embedding
        
        return FallbackEmbeddings(self.embedding_model)
        
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
    
    def build_vector_store(self, documents: List[Document]):
        """构建FAISS向量存储"""
        logger.info(f"开始构建向量存储，共 {len(documents)} 个文档")
        
        try:
            from langchain_community.vectorstores import FAISS
            
            # 批量处理以避免API限制
            batch_size = 100
            
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
                    
        except Exception as e:
            logger.error(f"FAISS构建失败: {e}")
            # 使用备用方案
            self._build_simple_vector_store(documents)
            
        logger.info("向量存储构建完成")
        return self.vector_store
    
    def _build_simple_vector_store(self, documents):
        """简单的向量存储备用方案"""
        logger.info("使用简单向量存储方案")
        
        # 存储文档和向量
        self.documents = documents
        self.document_embeddings = []
        
        # 批量获取embeddings
        batch_size = 50
        for i in tqdm(range(0, len(documents), batch_size)):
            batch = documents[i:i+batch_size]
            texts = [doc.page_content for doc in batch]
            embeddings = self.embeddings.embed_documents(texts)
            self.document_embeddings.extend(embeddings)
        
        # 创建一个简单的检索器
        class SimpleVectorStore:
            def __init__(self, docs, embeddings):
                self.documents = docs
                self.embeddings = np.array(embeddings)
            
            def similarity_search(self, query, k=5):
                # 获取查询向量
                query_embedding = self.embeddings.embed_query(query)
                query_vec = np.array(query_embedding)
                
                # 计算余弦相似度
                similarities = np.dot(self.embeddings, query_vec)
                top_k_indices = np.argsort(similarities)[-k:][::-1]
                
                return [self.documents[i] for i in top_k_indices]
            
            def as_retriever(self, search_kwargs=None):
                return self
            
            def get_relevant_documents(self, query):
                k = search_kwargs.get("k", 5) if search_kwargs else 5
                return self.similarity_search(query, k)
        
        self.vector_store = SimpleVectorStore(documents, self.document_embeddings)
    
    def save_vector_store(self, save_path: Path):
        """保存向量存储到本地"""
        if self.vector_store is None:
            raise ValueError("向量存储尚未构建")
            
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # 尝试FAISS保存
            self.vector_store.save_local(str(save_path))
        except:
            # 备用：pickle保存
            with open(save_path.with_suffix('.pkl'), 'wb') as f:
                pickle.dump({
                    'documents': self.documents if hasattr(self, 'documents') else [],
                    'embeddings': self.document_embeddings if hasattr(self, 'document_embeddings') else []
                }, f)
        
        logger.info(f"向量存储已保存到: {save_path}")
    
    def load_vector_store(self, load_path: Path):
        """从本地加载向量存储"""
        load_path = Path(load_path)
        
        if not load_path.exists() and not load_path.with_suffix('.pkl').exists():
            raise FileNotFoundError(f"向量存储文件不存在: {load_path}")
        
        try:
            from langchain_community.vectorstores import FAISS
            self.vector_store = FAISS.load_local(
                str(load_path), 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        except:
            # 备用：pickle加载
            with open(load_path.with_suffix('.pkl'), 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.document_embeddings = data['embeddings']
                self._build_simple_vector_store(self.documents)
        
        logger.info(f"向量存储已从 {load_path} 加载")
        return self.vector_store
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """执行相似度搜索"""
        if self.vector_store is None:
            raise ValueError("向量存储尚未构建或加载")
            
        results = self.vector_store.similarity_search(query, k=k)
        logger.info(f"查询: '{query[:50]}...' 返回 {len(results)} 个结果")
        return results
    
    def get_retriever(self, search_kwargs: Optional[Dict] = None):
        """获取检索器对象"""
        if self.vector_store is None:
            raise ValueError("向量存储尚未构建或加载")
            
        if search_kwargs is None:
            search_kwargs = {"k": 5}
            
        if hasattr(self.vector_store, 'as_retriever'):
            return self.vector_store.as_retriever(search_kwargs=search_kwargs)
        else:
            # 简单检索器
            return self.vector_store