"""
ElasticSearch RAG API服务
提供RESTful API接口用于PDF上传、搜索、问答
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
import hashlib
import os
import tempfile
from pathlib import Path
import shutil

from es_pdf_pipeline import PDFPipeline, ESIndexer
from elasticsearch import Elasticsearch
import openai
from dotenv import load_dotenv

load_dotenv()

# 创建FastAPI应用
app = FastAPI(
    title="Insurance RAG API",
    description="企业级保险文档RAG系统API",
    version="1.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
pipeline = PDFPipeline()
indexer = ESIndexer()
es = Elasticsearch(['http://localhost:9200'])
openai.api_key = os.getenv("OPENAI_API_KEY")

# 请求/响应模型
class SearchRequest(BaseModel):
    query: str
    size: int = 10
    doc_filter: Optional[str] = None
    
class QARequest(BaseModel):
    question: str
    doc_filter: Optional[str] = None
    include_sources: bool = True
    
class SearchResult(BaseModel):
    doc_name: str
    page_num: int
    content: str
    score: float
    highlight: Optional[str] = None
    
class QAResponse(BaseModel):
    answer: str
    sources: List[SearchResult]
    confidence: float
    processing_time: float

class IndexStats(BaseModel):
    total_documents: int
    total_chunks: int
    index_size: str
    last_updated: str

# API端点

# 添加静态文件路由
@app.get("/web_demo.html")
async def serve_demo():
    """提供Web演示页面"""
    return FileResponse("web_demo.html")

@app.get("/")
async def root():
    """API根路径"""
    return {
        "message": "Insurance RAG API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/api/upload",
            "search": "/api/search",
            "qa": "/api/qa",
            "stats": "/api/stats",
            "health": "/api/health"
        }
    }

@app.get("/api/health")
async def health_check():
    """健康检查"""
    try:
        # 检查ES连接
        es_health = es.cluster.health()
        
        # 检查索引
        doc_index_exists = es.indices.exists(index="insurance_documents")
        field_index_exists = es.indices.exists(index="insurance_fields")
        
        return {
            "status": "healthy",
            "elasticsearch": {
                "status": es_health['status'],
                "nodes": es_health['number_of_nodes']
            },
            "indices": {
                "documents": doc_index_exists,
                "fields": field_index_exists
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")

@app.post("/api/upload")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """上传PDF文档"""
    # 验证文件类型
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # 保存临时文件
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / file.filename
    
    try:
        # 保存上传的文件
        with open(temp_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        # 计算文档ID
        doc_id = hashlib.md5(file.filename.encode()).hexdigest()
        
        # 异步处理PDF
        background_tasks.add_task(process_pdf_background, str(temp_path), temp_dir)
        
        return {
            "message": "PDF upload initiated",
            "doc_id": doc_id,
            "filename": file.filename,
            "size": len(content),
            "status": "processing"
        }
        
    except Exception as e:
        # 清理临时文件
        if temp_path.exists():
            os.remove(temp_path)
        os.rmdir(temp_dir)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

async def process_pdf_background(pdf_path: str, temp_dir: str):
    """后台处理PDF"""
    try:
        # 处理PDF
        result = await pipeline.process_pdf(pdf_path)
        print(f"PDF处理完成: {result}")
    finally:
        # 清理临时文件
        if Path(pdf_path).exists():
            os.remove(pdf_path)
        if Path(temp_dir).exists():
            os.rmdir(temp_dir)

@app.post("/api/search", response_model=List[SearchResult])
async def search_documents(request: SearchRequest):
    """搜索文档"""
    try:
        # 构建查询
        query_body = {
            "size": request.size,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": request.query,
                                "fields": ["content^2", "content.english"],
                                "type": "best_fields"
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
            "_source": ["doc_name", "page_num", "content", "chunk_type"]
        }
        
        # 添加文档过滤
        if request.doc_filter:
            query_body["query"]["bool"]["filter"] = [
                {"term": {"doc_name.keyword": request.doc_filter}}
            ]
        
        # 执行搜索
        response = es.search(index="insurance_documents", body=query_body)
        
        # 格式化结果
        results = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            
            # 获取高亮内容
            highlight = None
            if 'highlight' in hit and 'content' in hit['highlight']:
                highlight = " ... ".join(hit['highlight']['content'])
            
            results.append(SearchResult(
                doc_name=source['doc_name'],
                page_num=source.get('page_num', 0),
                content=source['content'][:500],  # 限制长度
                score=hit['_score'],
                highlight=highlight
            ))
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/api/qa", response_model=QAResponse)
async def question_answering(request: QARequest):
    """问答接口"""
    start_time = datetime.now()
    
    try:
        # 1. 搜索相关文档
        search_results = await search_documents(
            SearchRequest(query=request.question, size=5, doc_filter=request.doc_filter)
        )
        
        if not search_results:
            return QAResponse(
                answer="No relevant information found in the documents.",
                sources=[],
                confidence=0.0,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
        
        # 2. 构建上下文
        context = "\n\n".join([
            f"[Source: {r.doc_name}, Page {r.page_num}]\n{r.content}"
            for r in search_results[:3]
        ])
        
        # 3. 调用GPT生成答案
        prompt = f"""Based on the following insurance document excerpts, answer the question.
        
Context:
{context}

Question: {request.question}

Instructions:
1. Answer based ONLY on the provided context
2. Be specific and cite the source when possible
3. If the answer is not in the context, say so
4. For insurance terms, include relevant details

Answer:"""
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an insurance expert assistant. Answer questions based on the provided document context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )
        
        answer = response['choices'][0]['message']['content']
        
        # 4. 计算置信度（基于搜索分数）
        avg_score = sum(r.score for r in search_results[:3]) / min(3, len(search_results))
        confidence = min(avg_score / 10, 1.0)  # 归一化到0-1
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return QAResponse(
            answer=answer,
            sources=search_results[:3] if request.include_sources else [],
            confidence=confidence,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QA failed: {str(e)}")

@app.get("/api/stats", response_model=IndexStats)
async def get_statistics():
    """获取索引统计信息"""
    try:
        # 获取文档索引统计
        doc_stats = es.indices.stats(index="insurance_documents")
        doc_count = es.count(index="insurance_documents")
        
        # 获取唯一文档数
        unique_docs = es.search(
            index="insurance_documents",
            body={
                "size": 0,
                "aggs": {
                    "unique_docs": {
                        "cardinality": {
                            "field": "doc_name.keyword"
                        }
                    }
                }
            }
        )
        
        # 格式化大小
        size_bytes = doc_stats['indices']['insurance_documents']['total']['store']['size_in_bytes']
        size_mb = size_bytes / (1024 * 1024)
        
        return IndexStats(
            total_documents=unique_docs['aggregations']['unique_docs']['value'],
            total_chunks=doc_count['count'],
            index_size=f"{size_mb:.2f} MB",
            last_updated=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")

@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """删除文档"""
    try:
        # 删除所有相关块
        es.delete_by_query(
            index="insurance_documents",
            body={
                "query": {
                    "term": {"doc_id": doc_id}
                }
            }
        )
        
        # 删除字段索引
        es.delete(index="insurance_fields", id=doc_id, ignore=404)
        
        return {"message": f"Document {doc_id} deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

@app.get("/api/documents")
async def list_documents():
    """列出所有文档"""
    try:
        response = es.search(
            index="insurance_documents",
            body={
                "size": 0,
                "aggs": {
                    "documents": {
                        "terms": {
                            "field": "doc_name.keyword",
                            "size": 100
                        },
                        "aggs": {
                            "doc_info": {
                                "top_hits": {
                                    "size": 1,
                                    "_source": ["doc_id", "upload_time"]
                                }
                            }
                        }
                    }
                }
            }
        )
        
        documents = []
        for bucket in response['aggregations']['documents']['buckets']:
            doc_info = bucket['doc_info']['hits']['hits'][0]['_source']
            documents.append({
                "doc_name": bucket['key'],
                "doc_id": doc_info.get('doc_id', 'unknown'),
                "chunk_count": bucket['doc_count'],
                "upload_time": doc_info.get('upload_time', 'unknown')
            })
        
        return {"documents": documents, "total": len(documents)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"List documents failed: {str(e)}")

# 启动服务器
if __name__ == "__main__":
    import uvicorn
    
    print("="*60)
    print("Insurance RAG API Server")
    print("="*60)
    print("API文档: http://localhost:8000/docs")
    print("="*60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)