"""
简化版API服务 - 用于快速演示
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from elasticsearch import Elasticsearch
import hashlib
import uvicorn

# 创建FastAPI应用
app = FastAPI(title="Insurance RAG API")

# CORS配置 - 允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 连接ES
es = Elasticsearch(['http://localhost:9200'])

# 请求模型
class SearchRequest(BaseModel):
    query: str
    size: int = 10
    doc_filter: Optional[str] = None

class QARequest(BaseModel):
    question: str
    include_sources: bool = True

# API端点
@app.get("/")
async def root():
    return {"message": "Insurance RAG API", "version": "1.0.0"}

@app.get("/api/health")
async def health_check():
    """健康检查"""
    try:
        es_health = es.cluster.health()
        return {
            "status": "healthy",
            "elasticsearch": es_health['status'],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    """获取统计信息"""
    try:
        # 检查索引是否存在
        if not es.indices.exists(index="insurance_documents"):
            # 创建索引
            es.indices.create(
                index="insurance_documents",
                body={
                    "mappings": {
                        "properties": {
                            "doc_name": {"type": "keyword"},
                            "content": {"type": "text"},
                            "page_num": {"type": "integer"},
                            "upload_time": {"type": "date"}
                        }
                    }
                }
            )
        
        doc_count = es.count(index="insurance_documents")
        
        return {
            "total_documents": doc_count.get('count', 0),
            "total_chunks": doc_count.get('count', 0),
            "index_size": "0 MB",
            "last_updated": datetime.now().isoformat()
        }
    except:
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "index_size": "0 MB",
            "last_updated": datetime.now().isoformat()
        }

@app.get("/api/documents")
async def list_documents():
    """列出文档"""
    try:
        if not es.indices.exists(index="insurance_documents"):
            return {"documents": [], "total": 0}
        
        result = es.search(
            index="insurance_documents",
            body={"size": 100, "query": {"match_all": {}}}
        )
        
        docs = []
        doc_names = set()
        for hit in result['hits']['hits']:
            doc_name = hit['_source'].get('doc_name', 'Unknown')
            if doc_name not in doc_names:
                doc_names.add(doc_name)
                docs.append({
                    "doc_name": doc_name,
                    "doc_id": hit['_id'],
                    "chunk_count": 1
                })
        
        return {"documents": docs, "total": len(docs)}
    except:
        return {"documents": [], "total": 0}

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """上传PDF"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    
    # 简单处理：创建一个示例文档
    doc_id = hashlib.md5(file.filename.encode()).hexdigest()
    
    # 确保索引存在
    if not es.indices.exists(index="insurance_documents"):
        es.indices.create(
            index="insurance_documents",
            body={
                "mappings": {
                    "properties": {
                        "doc_id": {"type": "keyword"},
                        "doc_name": {"type": "keyword"},
                        "content": {"type": "text"},
                        "page_num": {"type": "integer"},
                        "upload_time": {"type": "date"}
                    }
                }
            }
        )
    
    # 插入示例文档
    es.index(
        index="insurance_documents",
        body={
            "doc_id": doc_id,
            "doc_name": file.filename,
            "content": f"This is a sample content for {file.filename}",
            "page_num": 1,
            "upload_time": datetime.now().isoformat()
        },
        refresh=True
    )
    
    return {
        "message": "Upload successful",
        "doc_id": doc_id,
        "filename": file.filename,
        "status": "completed"
    }

@app.post("/api/search")
async def search(request: SearchRequest):
    """搜索文档"""
    try:
        if not es.indices.exists(index="insurance_documents"):
            return []
        
        result = es.search(
            index="insurance_documents",
            body={
                "query": {"match": {"content": request.query}},
                "size": request.size,
                "highlight": {
                    "fields": {"content": {}},
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"]
                }
            }
        )
        
        results = []
        for hit in result['hits']['hits']:
            source = hit['_source']
            highlight = None
            if 'highlight' in hit:
                highlight = hit['highlight']['content'][0]
            
            results.append({
                "doc_name": source.get('doc_name', 'Unknown'),
                "page_num": source.get('page_num', 1),
                "content": source.get('content', ''),
                "score": hit['_score'],
                "highlight": highlight
            })
        
        return results
    except:
        return []

@app.post("/api/qa")
async def qa(request: QARequest):
    """问答接口"""
    # 简化版：返回模拟答案
    return {
        "answer": f"Based on the documents, the answer to '{request.question}' is: This is a demo answer. The system would search relevant documents and generate a proper answer using AI.",
        "sources": [
            {
                "doc_name": "RoyalFortune.pdf",
                "page_num": 1,
                "content": "Sample source content",
                "score": 0.95
            }
        ],
        "confidence": 0.85,
        "processing_time": 1.5
    }

if __name__ == "__main__":
    import sys
    
    # 检查端口参数
    port = 8000
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv):
            if arg == "--port" and i + 1 < len(sys.argv):
                port = int(sys.argv[i + 1])
    
    print("="*60)
    print("简化版 Insurance RAG API")
    print("="*60)
    print(f"API文档: http://localhost:{port}/docs")
    print("="*60)
    
    uvicorn.run(app, host="0.0.0.0", port=port)