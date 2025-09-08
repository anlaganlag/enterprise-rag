"""
演示API - 不使用向量编码，纯BM25搜索
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from elasticsearch import Elasticsearch
import hashlib
import uvicorn
import pdfplumber
from pathlib import Path
import tempfile
import os

# 创建FastAPI应用
app = FastAPI(title="Insurance RAG Demo API")

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 连接ES
es = Elasticsearch(['http://localhost:9200'])

# 确保索引存在
if not es.indices.exists(index="demo_documents"):
    es.indices.create(
        index="demo_documents",
        body={
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "doc_name": {"type": "keyword"},
                    "content": {"type": "text", "analyzer": "standard"},
                    "page_num": {"type": "integer"},
                    "upload_time": {"type": "date"}
                }
            }
        }
    )

# 请求模型
class SearchRequest(BaseModel):
    query: str
    size: int = 10

class QARequest(BaseModel):
    question: str

@app.get("/api/health")
async def health():
    return {"status": "healthy", "elasticsearch": es.cluster.health()['status']}

@app.get("/api/stats")
async def stats():
    doc_count = es.count(index="demo_documents")
    return {
        "total_documents": doc_count.get('count', 0),
        "total_chunks": doc_count.get('count', 0),
        "index_size": "0 MB",
        "last_updated": datetime.now().isoformat()
    }

@app.get("/api/documents")
async def list_docs():
    result = es.search(
        index="demo_documents",
        body={
            "size": 0,
            "aggs": {
                "docs": {
                    "terms": {"field": "doc_name.keyword", "size": 100}
                }
            }
        }
    )
    
    docs = []
    for bucket in result['aggregations']['docs']['buckets']:
        docs.append({
            "doc_name": bucket['key'],
            "doc_id": hashlib.md5(bucket['key'].encode()).hexdigest(),
            "chunk_count": bucket['doc_count']
        })
    
    return {"documents": docs, "total": len(docs)}

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """处理PDF上传 - 简化版"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files allowed")
    
    # 保存临时文件
    temp_path = Path(tempfile.gettempdir()) / file.filename
    content = await file.read()
    temp_path.write_bytes(content)
    
    try:
        # 提取文本
        chunks = []
        with pdfplumber.open(temp_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    chunks.append({
                        "doc_id": hashlib.md5(file.filename.encode()).hexdigest(),
                        "doc_name": file.filename,
                        "content": text,
                        "page_num": i + 1,
                        "upload_time": datetime.now().isoformat()
                    })
        
        # 索引到ES
        for chunk in chunks:
            es.index(index="demo_documents", body=chunk, refresh=True)
        
        return {
            "message": "Upload successful",
            "doc_id": chunks[0]["doc_id"] if chunks else "",
            "filename": file.filename,
            "chunks": len(chunks),
            "status": "completed"
        }
    finally:
        temp_path.unlink()

@app.post("/api/search")
async def search(request: SearchRequest):
    """纯BM25搜索"""
    result = es.search(
        index="demo_documents",
        body={
            "query": {
                "multi_match": {
                    "query": request.query,
                    "fields": ["content"],
                    "type": "best_fields"
                }
            },
            "size": request.size,
            "highlight": {
                "fields": {
                    "content": {
                        "fragment_size": 150,
                        "number_of_fragments": 3,
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"]
                    }
                }
            }
        }
    )
    
    results = []
    for hit in result['hits']['hits']:
        source = hit['_source']
        highlight = ""
        if 'highlight' in hit:
            highlight = " ... ".join(hit['highlight']['content'])
        
        results.append({
            "doc_name": source['doc_name'],
            "page_num": source['page_num'],
            "content": source['content'][:300],
            "score": hit['_score'],
            "highlight": highlight
        })
    
    return results

@app.post("/api/qa")
async def qa(request: QARequest):
    """简单问答"""
    # 先搜索
    search_result = await search(SearchRequest(query=request.question, size=3))
    
    if not search_result:
        answer = "No relevant information found."
    else:
        # 简单拼接答案
        context = "\n".join([r["content"][:200] for r in search_result[:3]])
        answer = f"Based on the documents:\n{context[:500]}..."
    
    return {
        "answer": answer,
        "sources": search_result[:3],
        "confidence": 0.75,
        "processing_time": 0.5
    }

if __name__ == "__main__":
    print("="*60)
    print("演示API（无向量版）")
    print("="*60)
    print("访问: http://localhost:8000/docs")
    print("="*60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)