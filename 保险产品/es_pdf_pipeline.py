"""
ElasticSearch PDF实时处理管道
实现PDF上传、解析、分块、向量化、索引的完整流程
"""
import os
import hashlib
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import pdfplumber
import PyPDF2
from elasticsearch import AsyncElasticsearch, Elasticsearch
from elasticsearch.helpers import async_bulk, bulk
import openai
from dotenv import load_dotenv
import numpy as np
import json
import re

load_dotenv()


class PDFProcessor:
    """PDF文档处理器"""
    
    def __init__(self, chunk_size=500, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """从PDF提取文本和元数据"""
        doc_info = {
            "file_path": pdf_path,
            "file_name": Path(pdf_path).name,
            "pages": [],
            "tables": [],
            "total_pages": 0,
            "extracted_at": datetime.now().isoformat()
        }
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                doc_info["total_pages"] = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages):
                    # 提取文本
                    text = page.extract_text() or ""
                    
                    # 提取表格
                    tables = page.extract_tables()
                    
                    page_data = {
                        "page_num": page_num + 1,
                        "text": text,
                        "tables": []
                    }
                    
                    # 处理表格
                    for table in tables:
                        if table and len(table) > 1:
                            table_text = self._table_to_text(table)
                            page_data["tables"].append({
                                "raw": table,
                                "text": table_text
                            })
                            doc_info["tables"].append({
                                "page": page_num + 1,
                                "content": table_text
                            })
                    
                    doc_info["pages"].append(page_data)
                    
        except Exception as e:
            print(f"PDF提取错误: {e}")
            # 备用方法
            try:
                with open(pdf_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    doc_info["total_pages"] = len(reader.pages)
                    
                    for page_num, page in enumerate(reader.pages):
                        text = page.extract_text()
                        doc_info["pages"].append({
                            "page_num": page_num + 1,
                            "text": text,
                            "tables": []
                        })
            except Exception as e2:
                print(f"备用提取也失败: {e2}")
                
        return doc_info
        
    def _table_to_text(self, table: List[List]) -> str:
        """将表格转换为文本"""
        lines = []
        for row in table:
            if row and any(cell for cell in row):
                row_text = " | ".join(str(cell or "").strip() for cell in row)
                if row_text.strip():
                    lines.append(row_text)
        return "\n".join(lines)
        
    def create_chunks(self, doc_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """智能文档分块"""
        chunks = []
        doc_id = hashlib.md5(doc_info["file_name"].encode()).hexdigest()
        
        for page_data in doc_info["pages"]:
            page_num = page_data["page_num"]
            text = page_data["text"]
            
            # 1. 段落分块
            paragraphs = self._split_into_paragraphs(text)
            for i, para in enumerate(paragraphs):
                if len(para.strip()) > 20:  # 过滤太短的段落
                    chunks.append({
                        "doc_id": doc_id,
                        "doc_name": doc_info["file_name"],
                        "chunk_id": f"{doc_id}_p{page_num}_para{i}",
                        "chunk_type": "paragraph",
                        "content": para,
                        "page_num": page_num,
                        "metadata": {
                            "position": i,
                            "length": len(para)
                        }
                    })
            
            # 2. 表格分块
            for j, table_data in enumerate(page_data["tables"]):
                chunks.append({
                    "doc_id": doc_id,
                    "doc_name": doc_info["file_name"],
                    "chunk_id": f"{doc_id}_p{page_num}_table{j}",
                    "chunk_type": "table",
                    "content": table_data["text"],
                    "page_num": page_num,
                    "metadata": {
                        "table_index": j,
                        "rows": len(table_data.get("raw", []))
                    }
                })
        
        # 3. 滑动窗口分块（用于长文本）
        full_text = "\n".join(p["text"] for p in doc_info["pages"])
        window_chunks = self._sliding_window_chunks(full_text, doc_id, doc_info["file_name"])
        chunks.extend(window_chunks)
        
        return chunks
        
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """按段落分割文本"""
        # 使用多种分割符
        paragraphs = re.split(r'\n\n+|\r\n\r\n+', text)
        
        # 进一步处理
        result = []
        for para in paragraphs:
            # 处理编号列表
            if re.match(r'^\d+\.|\w\)', para):
                result.append(para)
            # 合并太短的段落
            elif len(para) < 100 and result:
                result[-1] += "\n" + para
            else:
                result.append(para)
                
        return result
        
    def _sliding_window_chunks(self, text: str, doc_id: str, doc_name: str) -> List[Dict]:
        """滑动窗口分块"""
        chunks = []
        sentences = re.split(r'[。！？.!?]\s*', text)
        
        current_chunk = []
        current_size = 0
        
        for sent in sentences:
            current_chunk.append(sent)
            current_size += len(sent)
            
            if current_size >= self.chunk_size:
                chunk_text = "。".join(current_chunk) if any("。" in s for s in current_chunk) else ". ".join(current_chunk)
                
                chunks.append({
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "chunk_id": f"{doc_id}_window_{len(chunks)}",
                    "chunk_type": "window",
                    "content": chunk_text,
                    "page_num": 0,  # 跨页
                    "metadata": {
                        "window_size": self.chunk_size,
                        "overlap": self.overlap
                    }
                })
                
                # 保留重叠部分
                overlap_count = max(1, len(current_chunk) * self.overlap // self.chunk_size)
                current_chunk = current_chunk[-overlap_count:]
                current_size = sum(len(s) for s in current_chunk)
                
        # 处理剩余内容
        if current_chunk:
            chunk_text = "。".join(current_chunk) if any("。" in s for s in current_chunk) else ". ".join(current_chunk)
            chunks.append({
                "doc_id": doc_id,
                "doc_name": doc_name,
                "chunk_id": f"{doc_id}_window_{len(chunks)}",
                "chunk_type": "window",
                "content": chunk_text,
                "page_num": 0,
                "metadata": {
                    "window_size": len(chunk_text),
                    "is_last": True
                }
            })
            
        return chunks


class VectorEncoder:
    """向量编码器"""
    
    def __init__(self, model="text-embedding-ada-002"):
        self.model = model
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
    async def encode_batch(self, texts: List[str], batch_size=20) -> List[List[float]]:
        """批量编码文本为向量"""
        vectors = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            try:
                response = await asyncio.to_thread(
                    openai.Embedding.create,
                    input=batch,
                    model=self.model
                )
                
                for item in response['data']:
                    vectors.append(item['embedding'])
                    
            except Exception as e:
                print(f"向量编码错误: {e}")
                # 返回零向量作为后备
                vectors.extend([[0.0] * 1536 for _ in batch])
                
        return vectors
        
    def encode_sync(self, text: str) -> List[float]:
        """同步编码单个文本"""
        try:
            response = openai.Embedding.create(
                input=[text],
                model=self.model
            )
            return response['data'][0]['embedding']
        except Exception as e:
            print(f"向量编码错误: {e}")
            return [0.0] * 1536


class ESIndexer:
    """ElasticSearch索引器"""
    
    def __init__(self, es_host="http://localhost:9200"):
        self.es = Elasticsearch([es_host])
        self.async_es = AsyncElasticsearch([es_host])
        self.doc_index = "insurance_documents"
        self.field_index = "insurance_fields"
        
    async def index_chunks(self, chunks: List[Dict], vectors: List[List[float]]) -> bool:
        """索引文档块到ES"""
        if len(chunks) != len(vectors):
            print(f"块数量({len(chunks)})与向量数量({len(vectors)})不匹配")
            return False
            
        actions = []
        for chunk, vector in zip(chunks, vectors):
            action = {
                "_index": self.doc_index,
                "_id": chunk["chunk_id"],
                "_source": {
                    **chunk,
                    "content_vector": vector,
                    "upload_time": datetime.now().isoformat()
                }
            }
            actions.append(action)
            
        try:
            success, failed = await async_bulk(
                self.async_es,
                actions,
                raise_on_error=False
            )
            
            print(f"索引成功: {success} 个文档")
            if failed:
                print(f"索引失败: {len(failed)} 个文档")
                for item in failed[:5]:
                    print(f"  失败原因: {item}")
                    
            return success > 0
            
        except Exception as e:
            print(f"批量索引错误: {e}")
            return False
            
    def index_extracted_fields(self, doc_id: str, fields: Dict[str, Any]) -> bool:
        """索引提取的字段"""
        try:
            self.es.index(
                index=self.field_index,
                id=doc_id,
                body={
                    "doc_id": doc_id,
                    **fields,
                    "extraction_time": datetime.now().isoformat()
                },
                refresh=True
            )
            return True
        except Exception as e:
            print(f"字段索引错误: {e}")
            return False
            
    def search(self, query: str, size=10) -> Dict:
        """混合搜索"""
        # 获取查询向量
        encoder = VectorEncoder()
        query_vector = encoder.encode_sync(query)
        
        # 构建混合查询
        body = {
            "size": size,
            "query": {
                "bool": {
                    "should": [
                        # BM25搜索
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["content^2", "content.english"],
                                "type": "best_fields",
                                "boost": 1.0
                            }
                        },
                        # 向量搜索
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'content_vector') + 1.0",
                                    "params": {
                                        "query_vector": query_vector
                                    }
                                },
                                "boost": 0.5
                            }
                        }
                    ]
                }
            },
            "highlight": {
                "fields": {
                    "content": {
                        "fragment_size": 150,
                        "number_of_fragments": 3,
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"]
                    }
                }
            },
            "_source": ["doc_name", "page_num", "content", "chunk_type"],
            "track_scores": True
        }
        
        return self.es.search(index=self.doc_index, body=body)


class PDFPipeline:
    """PDF处理管道主类"""
    
    def __init__(self):
        self.processor = PDFProcessor()
        self.encoder = VectorEncoder()
        self.indexer = ESIndexer()
        
    async def process_pdf(self, pdf_path: str, extract_fields: bool = True) -> Dict[str, Any]:
        """处理单个PDF的完整流程"""
        print(f"\n处理PDF: {Path(pdf_path).name}")
        
        # 1. 提取文本
        print("  1. 提取文本...")
        doc_info = self.processor.extract_text_from_pdf(pdf_path)
        print(f"     - 页数: {doc_info['total_pages']}")
        print(f"     - 表格数: {len(doc_info['tables'])}")
        
        # 2. 创建分块
        print("  2. 创建分块...")
        chunks = self.processor.create_chunks(doc_info)
        print(f"     - 分块数: {len(chunks)}")
        
        # 3. 向量编码
        print("  3. 向量编码...")
        texts = [chunk["content"] for chunk in chunks]
        vectors = await self.encoder.encode_batch(texts)
        print(f"     - 向量数: {len(vectors)}")
        
        # 4. 索引到ES
        print("  4. 索引到ES...")
        success = await self.indexer.index_chunks(chunks, vectors)
        
        # 5. 提取结构化字段（可选）
        if extract_fields and success:
            print("  5. 提取结构化字段...")
            fields = self._extract_structured_fields(doc_info)
            doc_id = hashlib.md5(doc_info["file_name"].encode()).hexdigest()
            self.indexer.index_extracted_fields(doc_id, fields)
            
        return {
            "success": success,
            "doc_id": hashlib.md5(doc_info["file_name"].encode()).hexdigest(),
            "chunks": len(chunks),
            "pages": doc_info["total_pages"],
            "tables": len(doc_info["tables"])
        }
        
    def _extract_structured_fields(self, doc_info: Dict) -> Dict[str, Any]:
        """从文档提取结构化字段"""
        fields = {}
        full_text = "\n".join(p["text"] for p in doc_info["pages"])
        
        # 简单的模式匹配提取
        patterns = {
            "insurer_entity_name": r"(?:Insurer|保險公司|Company)[:\s]*([\w\s]+)",
            "product_name": r"(?:Product|產品|計劃)[:\s]*([\w\s]+)",
            "minimum_premium": r"(?:Minimum|最低).*?(?:Premium|保費).*?([0-9,]+)",
            "policy_currency": r"(?:Currency|貨幣)[:\s]*(USD|HKD|EUR|CNY)"
        }
        
        for field, pattern in patterns.items():
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                fields[field] = match.group(1).strip()
                
        return fields


async def main():
    """测试管道"""
    pipeline = PDFPipeline()
    
    # 测试PDF文件
    pdf_files = [
        r"D:\桌面\RAG\保险产品\RoyalFortune_Product Brochure_EN.pdf",
        r"D:\桌面\RAG\保险产品\AIA FlexiAchieverSavingsPlan_tc-活享儲蓄計劃.pdf"
    ]
    
    for pdf_file in pdf_files:
        if Path(pdf_file).exists():
            result = await pipeline.process_pdf(pdf_file)
            print(f"\n处理结果: {result}")
            
    # 测试搜索
    print("\n测试搜索功能:")
    test_queries = ["minimum premium", "保費", "Sun Life"]
    
    for query in test_queries:
        results = pipeline.indexer.search(query, size=3)
        print(f"\n搜索 '{query}': {results['hits']['total']['value']} 个结果")
        for hit in results['hits']['hits'][:2]:
            print(f"  - 分数: {hit['_score']:.2f}")
            print(f"    内容: {hit['_source']['content'][:100]}...")


if __name__ == "__main__":
    asyncio.run(main())