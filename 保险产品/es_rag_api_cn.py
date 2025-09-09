"""
中文优化版RAG API服务
集成中文查询优化器和改进的Prompt
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
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
from chinese_query_optimizer import ChineseQueryOptimizer

load_dotenv()

# 创建FastAPI应用
app = FastAPI(
    title="Insurance RAG API (中文优化版)",
    description="支持中文的保险文档RAG系统API",
    version="2.0.0"
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
chinese_optimizer = ChineseQueryOptimizer()

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
    language: str  # 添加语言标识

# API端点

@app.get("/web_demo.html")
async def serve_demo():
    """提供Web演示页面"""
    return FileResponse("web_demo.html")

@app.get("/")
async def root():
    """API根路径"""
    return {
        "message": "Insurance RAG API (中文优化版)",
        "version": "2.0.0",
        "features": [
            "中文查询优化",
            "术语自动翻译",
            "同义词扩展",
            "中文友好回答"
        ],
        "endpoints": {
            "upload": "/api/upload",
            "search": "/api/search",
            "search_cn": "/api/search_cn",
            "qa": "/api/qa",
            "qa_cn": "/api/qa_cn",
            "stats": "/api/stats",
            "health": "/api/health"
        }
    }

@app.get("/api/health")
async def health_check():
    """健康检查"""
    try:
        es_health = es.cluster.health()
        doc_index_exists = es.indices.exists(index="insurance_documents")
        
        return {
            "status": "healthy",
            "elasticsearch": {
                "status": es_health['status'],
                "nodes": es_health['number_of_nodes']
            },
            "indices": {
                "documents": doc_index_exists
            },
            "chinese_optimizer": "enabled",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")

@app.post("/api/search_cn", response_model=List[SearchResult])
async def search_documents_chinese(request: SearchRequest):
    """中文优化搜索"""
    try:
        # 1. 优化查询
        expanded_queries, query_info = chinese_optimizer.optimize_for_chinese(request.query)
        
        # 2. 构建多查询
        should_clauses = []
        for expanded_query in expanded_queries[:3]:  # 使用前3个扩展查询
            should_clauses.append({
                "multi_match": {
                    "query": expanded_query,
                    "fields": ["content^2", "content.english"],
                    "type": "best_fields"
                }
            })
        
        query_body = {
            "size": request.size,
            "query": {
                "bool": {
                    "should": should_clauses,
                    "minimum_should_match": 1
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
            
            highlight = None
            if 'highlight' in hit and 'content' in hit['highlight']:
                highlight = " ... ".join(hit['highlight']['content'])
            
            results.append(SearchResult(
                doc_name=source['doc_name'],
                page_num=source.get('page_num', 0),
                content=source['content'][:500],
                score=hit['_score'],
                highlight=highlight
            ))
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/api/qa_cn", response_model=QAResponse)
async def question_answering_chinese(request: QARequest):
    """中文优化问答"""
    start_time = datetime.now()
    
    try:
        # 1. 检测语言
        language = chinese_optimizer.detect_language(request.question)
        
        # 2. 优化搜索
        search_results = await search_documents_chinese(
            SearchRequest(query=request.question, size=5, doc_filter=request.doc_filter)
        )
        
        if not search_results:
            answer = "抱歉，未找到相关信息。" if language == "chinese" else "No relevant information found."
            return QAResponse(
                answer=answer,
                sources=[],
                confidence=0.0,
                processing_time=(datetime.now() - start_time).total_seconds(),
                language=language
            )
        
        # 3. 构建上下文
        context = "\n\n".join([
            f"[来源: {r.doc_name}, 第{r.page_num}页]\n{r.content}"
            for r in search_results[:3]
        ])
        
        # 4. 选择合适的Prompt
        if language == "chinese":
            system_prompt = """你是一位专业的保险顾问，请用中文回答客户的问题。
要求：
1. 使用通俗易懂的语言，避免过多专业术语
2. 如果涉及金额，使用中文习惯表达（如：12.5万美元）
3. 保持专业但友好的语气
4. 如果文档是英文的，请翻译关键信息
5. 只根据提供的文档内容回答，不要编造信息"""
            
            user_prompt = f"""根据以下保险文档内容，回答问题。

文档内容：
{context}

问题：{request.question}

请用中文回答："""
        else:
            system_prompt = "You are a professional insurance consultant. Answer questions based on the provided document context."
            user_prompt = f"""Based on the following insurance document excerpts, answer the question.

Context:
{context}

Question: {request.question}

Answer:"""
        
        # 5. 调用GPT
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )
        
        answer = response['choices'][0]['message']['content']
        
        # 6. 计算置信度
        avg_score = sum(r.score for r in search_results[:3]) / min(3, len(search_results))
        confidence = min(avg_score / 10, 1.0)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return QAResponse(
            answer=answer,
            sources=search_results[:3] if request.include_sources else [],
            confidence=confidence,
            processing_time=processing_time,
            language=language
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QA failed: {str(e)}")

# 保留原有的英文接口
@app.post("/api/search", response_model=List[SearchResult])
async def search_documents(request: SearchRequest):
    """标准搜索（向后兼容）"""
    # 检测语言，如果是中文则自动使用中文优化
    if chinese_optimizer.detect_language(request.query) == "chinese":
        return await search_documents_chinese(request)
    
    # 原有的英文搜索逻辑
    try:
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
        
        if request.doc_filter:
            query_body["query"]["bool"]["filter"] = [
                {"term": {"doc_name.keyword": request.doc_filter}}
            ]
        
        response = es.search(index="insurance_documents", body=query_body)
        
        results = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            
            highlight = None
            if 'highlight' in hit and 'content' in hit['highlight']:
                highlight = " ... ".join(hit['highlight']['content'])
            
            results.append(SearchResult(
                doc_name=source['doc_name'],
                page_num=source.get('page_num', 0),
                content=source['content'][:500],
                score=hit['_score'],
                highlight=highlight
            ))
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/api/qa", response_model=QAResponse)
async def question_answering(request: QARequest):
    """标准问答（向后兼容）"""
    # 检测语言，如果是中文则自动使用中文优化
    if chinese_optimizer.detect_language(request.question) == "chinese":
        return await question_answering_chinese(request)
    
    # 原有的英文问答逻辑
    start_time = datetime.now()
    
    try:
        search_results = await search_documents(
            SearchRequest(query=request.question, size=5, doc_filter=request.doc_filter)
        )
        
        if not search_results:
            return QAResponse(
                answer="No relevant information found in the documents.",
                sources=[],
                confidence=0.0,
                processing_time=(datetime.now() - start_time).total_seconds(),
                language="english"
            )
        
        context = "\n\n".join([
            f"[Source: {r.doc_name}, Page {r.page_num}]\n{r.content}"
            for r in search_results[:3]
        ])
        
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
        
        avg_score = sum(r.score for r in search_results[:3]) / min(3, len(search_results))
        confidence = min(avg_score / 10, 1.0)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return QAResponse(
            answer=answer,
            sources=search_results[:3] if request.include_sources else [],
            confidence=confidence,
            processing_time=processing_time,
            language="english"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QA failed: {str(e)}")

# 其他接口保持不变...
@app.post("/api/upload")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """上传PDF文档"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / file.filename
    
    try:
        with open(temp_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        doc_id = hashlib.md5(file.filename.encode()).hexdigest()
        background_tasks.add_task(process_pdf_background, str(temp_path), temp_dir)
        
        return {
            "message": "PDF upload initiated",
            "doc_id": doc_id,
            "filename": file.filename,
            "size": len(content),
            "status": "processing"
        }
        
    except Exception as e:
        if temp_path.exists():
            os.remove(temp_path)
        os.rmdir(temp_dir)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

async def process_pdf_background(pdf_path: str, temp_dir: str):
    """后台处理PDF"""
    try:
        result = await pipeline.process_pdf(pdf_path)
        print(f"PDF处理完成: {result}")
    finally:
        if Path(pdf_path).exists():
            os.remove(pdf_path)
        if Path(temp_dir).exists():
            os.rmdir(temp_dir)

@app.get("/api/stats")
async def get_statistics():
    """获取索引统计信息"""
    try:
        doc_stats = es.indices.stats(index="insurance_documents")
        doc_count = es.count(index="insurance_documents")
        
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
        
        size_bytes = doc_stats['indices']['insurance_documents']['total']['store']['size_in_bytes']
        size_mb = size_bytes / (1024 * 1024)
        
        return {
            "total_documents": unique_docs['aggregations']['unique_docs']['value'],
            "total_chunks": doc_count['count'],
            "index_size": f"{size_mb:.2f} MB",
            "chinese_optimizer": "enabled",
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")

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
    print("Insurance RAG API Server (中文优化版)")
    print("="*60)
    print("新增功能：")
    print("  - 中文查询优化")
    print("  - 术语自动翻译")
    print("  - 同义词扩展")
    print("  - 中文友好回答")
    print("="*60)
    print("API文档: http://localhost:8001/docs")
    print("="*60)
    
    uvicorn.run(app, host="0.0.0.0", port=8001)