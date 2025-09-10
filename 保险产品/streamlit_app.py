"""
保险产品RAG系统 - 自定义导航版本
解决标签页跳转问题
"""
import streamlit as st
import os
from typing import List, Dict
import json
from datetime import datetime
import pandas as pd
import time
import logging
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,  # 改为INFO级别，大幅提升性能
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('rag_debug.log', encoding='utf-8')
    ]
)

# 特别禁用pdfminer的调试日志（这是性能杀手！）
logging.getLogger('pdfminer').setLevel(logging.WARNING)
logging.getLogger('pdfplumber').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# 核心库
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import PyPDF2
import pdfplumber
# 导入改进的PDF处理器和诊断工具
from pdf_processor_improved import process_pdf_with_timeout, check_dependencies
from pdf_processor_fast import process_pdf_fast, process_pdf_parallel  # 导入快速处理器
from pdf_diagnostic import diagnose_pdf, display_pdf_diagnosis, should_process_pdf, get_processing_timeout

# 页面配置
st.set_page_config(
    page_title="保险产品智能问答系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .nav-button {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.2rem;
        background: #f0f2f6;
        border-radius: 8px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .nav-button:hover {
        background: #667eea;
        color: white;
    }
    
    .nav-button.active {
        background: #667eea;
        color: white;
        font-weight: bold;
    }
    
    .question-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    
    .confidence-high {
        background: #28a745;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
    }
    
    .confidence-medium {
        background: #ffc107;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
    }
    
    .confidence-low {
        background: #dc3545;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
    }
    
    .answer-card {
        background: #f0f9ff;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .category-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# 初始化session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "智能问答"
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'selected_question' not in st.session_state:
    st.session_state.selected_question = ""
if 'auto_submit' not in st.session_state:
    st.session_state.auto_submit = False
if 'last_answer' not in st.session_state:
    st.session_state.last_answer = None

# 问题分类结构
QUESTION_CATEGORIES = {
    "📦 产品信息": {
        "questions": [
            ("保险公司名称", "What is the insurer entity name?", True),
            ("产品名称", "What is the product name?", True),
            ("产品资产配置", "What is the product asset mix?", True),
            ("产品类型", "What is the product type?", True),
            ("发行地区", "What is the issuing jurisdiction?", True),
        ],
        "completion": "9/9"
    },
    "📄 计划详情": {
        "questions": [
            ("最低保费", "What is the minimum premium?", True),
            ("投保年龄", "What is the issue age range?", True),
            ("保单货币", "What are the policy currencies?", True),
            ("缴费期限", "What are the premium terms?", True),
            ("身故赔付特点", "What are the death settlement features?", True),
        ],
        "completion": "9/11"
    },
    "💰 分红型终身寿险": {
        "questions": [
            ("首日保证现金价值", "What is the Day 1 GCV?", True),
            ("退保价值组成", "What are the surrender value components?", True),
            ("身故赔付组成", "What are the death benefit components?", True),
        ],
        "completion": "3/3"
    }
}

# 热门快速问题
HOT_QUESTIONS = [
    ("最低保费", "RoyalFortune的最低保费是多少？"),
    ("投保年龄", "投保年龄范围是什么？"),
    ("保证现金价值", "保证现金价值是多少？"),
    ("身故赔付", "身故赔付有哪些特点？"),
]

def initialize_openai():
    """初始化OpenAI客户端"""
    api_key = None
    
    # 从.env文件读取
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
    except:
        pass
    
    if api_key:
        openai.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key
    
    return api_key

@st.cache_data
def process_pdf(uploaded_file) -> List[Dict]:
    """处理PDF文件"""
    chunks = []
    try:
        logger.info(f"开始处理PDF文件: {uploaded_file.name}")
        
        with pdfplumber.open(uploaded_file) as pdf:
            text = ""
            empty_pages = 0
            total_pages = len(pdf.pages)
            logger.info(f"PDF总页数: {total_pages}")
            
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text() or ""
                text_length = len(page_text.strip())
                logger.debug(f"第 {page_num} 页提取文本长度: {text_length}")
                
                # 统计空页面
                if not page_text.strip():
                    empty_pages += 1
                    logger.warning(f"第 {page_num} 页未提取到文本内容")
                    st.warning(f"⚠️ 第 {page_num} 页未提取到文本内容")
                else:
                    # 显示前100个字符作为预览
                    preview = page_text[:100].replace('\n', ' ')
                    logger.info(f"第 {page_num} 页文本预览: {preview}...")
                
                text += f"\n[Page {page_num}]\n{page_text}\n"
                
                # 提取表格
                tables = page.extract_tables()
                if tables:
                    logger.info(f"第 {page_num} 页发现 {len(tables)} 个表格")
                for table_idx, table in enumerate(tables):
                    if table:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        logger.debug(f"表格 {table_idx+1} 大小: {df.shape}")
                        text += f"\n[Table on Page {page_num}]\n{df.to_string()}\n"
            
            # 显示提取统计
            logger.info(f"文档提取完成 - 总文本长度: {len(text)}, 空页面: {empty_pages}/{total_pages}")
            
            if empty_pages > 0:
                st.warning(f"📊 文档提取统计: 总页数 {total_pages}, 空页面 {empty_pages}, 有效页面 {total_pages - empty_pages}")
            
            # 检查是否提取到有效内容
            if not text.strip():
                logger.error("未能从PDF中提取到任何文本内容")
                st.error("❌ 未能从PDF中提取到任何文本内容，请检查PDF是否为扫描版")
                return []
        
        # 使用更小的chunk_size和更大的overlap以保持上下文完整性
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # 增加到1000
            chunk_overlap=300,  # 增加到300
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
        
        chunks = text_splitter.split_text(text)
        logger.info(f"文本分块完成: 生成 {len(chunks)} 个chunks")
        
        # 过滤掉过短的chunks
        valid_chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]
        logger.info(f"有效chunks: {len(valid_chunks)}/{len(chunks)}")
        
        # 记录前3个chunks作为示例
        for i, chunk in enumerate(valid_chunks[:3]):
            logger.debug(f"Chunk {i} 预览: {chunk[:100]}...")
        
        st.info(f"📄 文本分块: 原始 {len(chunks)} 块, 有效 {len(valid_chunks)} 块")
        
        return [{"content": chunk, "metadata": {"source": uploaded_file.name, "chunk_index": i}} 
                for i, chunk in enumerate(valid_chunks)]
    
    except Exception as e:
        logger.exception(f"PDF处理错误: {str(e)}")
        st.error(f"PDF处理错误: {str(e)}")
        st.error("💡 提示: 如果是扫描版PDF，请尝试使用OCR工具先转换为文本")
        return []

def create_vector_store(documents: List[Dict]):
    """创建向量存储"""
    try:
        logger.info(f"开始创建向量存储，文档数: {len(documents)}")
        
        embeddings = OpenAIEmbeddings()
        texts = [doc["content"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        
        logger.info("正在生成embeddings...")
        vector_store = FAISS.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas
        )
        
        logger.info("向量存储创建成功")
        return vector_store
    except Exception as e:
        logger.exception(f"创建向量存储失败: {str(e)}")
        st.error(f"创建向量存储失败: {str(e)}")
        return None

def create_advanced_prompt(question: str, context: str, question_type: str = "general") -> str:
    """创建高级多阶段提示词"""
    
    # 问题类型判断
    if any(keyword in question.lower() for keyword in ["minimum", "最低", "least", "min"]):
        question_type = "specific_value"
    elif any(keyword in question.lower() for keyword in ["how", "what", "explain", "什么", "如何", "解释"]):
        question_type = "explanation"
    elif any(keyword in question.lower() for keyword in ["compare", "difference", "vs", "比较", "区别"]):
        question_type = "comparison"
    
    # 基础系统角色
    base_system = """You are a professional insurance product analyst with expertise in policy details, financial products, and regulatory compliance. You have deep knowledge of insurance terminology in both Chinese and English."""
    
    # 针对不同问题类型的专业提示
    type_prompts = {
        "specific_value": """
TASK: Extract and provide specific numerical or factual information.
APPROACH: 
1. Scan the context for exact values, amounts, percentages, or specific terms
2. Quote the exact information found
3. Provide the source context where you found this information
4. If multiple values exist, list them clearly
5. If no exact match is found, state clearly what information is missing
""",
        "explanation": """
TASK: Provide comprehensive explanations with clear reasoning.
APPROACH:
1. Break down complex concepts into understandable components
2. Explain the significance or implications of the information
3. Use analogies or examples if helpful
4. Structure your response logically (definition, features, benefits, etc.)
5. Cite specific evidence from the provided context
""",
        "comparison": """
TASK: Analyze and compare different options or features.
APPROACH:
1. Identify all relevant comparison points
2. Create a structured comparison highlighting differences and similarities
3. Explain the implications of each difference
4. Provide recommendations if appropriate based on the context
5. Support all comparisons with specific evidence from the context
""",
        "general": """
TASK: Provide accurate, comprehensive information based on the context.
APPROACH:
1. Directly address the specific question asked
2. Provide supporting details and context
3. Explain any technical terms used
4. Structure information clearly and logically
5. Cite specific sources from the provided context
"""
    }
    
    # 质量控制指令
    quality_control = """
QUALITY REQUIREMENTS:
- ACCURACY: Only use information explicitly stated in the provided context
- PRECISION: Quote exact figures, percentages, and terms when available
- COMPLETENESS: Address all parts of the question
- CLARITY: Use clear, professional language appropriate for insurance clients
- HONESTY: If information is not available in the context, state this clearly
- CITATIONS: Reference specific parts of the context that support your answer

LANGUAGE RULES:
- Respond in the same language as the question when possible
- For mixed-language contexts, prioritize clarity over strict language matching
- Use standard insurance terminology
- Maintain professional tone throughout
"""
    
    # 输出格式指导
    format_guide = """
OUTPUT FORMAT:
1. Direct Answer: Start with a clear, direct response to the question
2. Supporting Details: Provide relevant background information and context
3. Specific Evidence: Quote or reference specific parts of the source material
4. Limitations: Note any limitations or missing information
5. Source Reference: Indicate which parts of the context were most relevant

If you cannot answer the question based on the provided context, clearly state:
"Based on the provided context, I cannot find sufficient information to answer [specific part of question]. The available information covers [what is available]."
"""
    
    # 组装最终提示
    final_prompt = f"""{base_system}

{type_prompts[question_type]}

{quality_control}

{format_guide}

CONTEXT:
{context}

USER QUESTION: {question}

Please provide a comprehensive, accurate response following the guidelines above:"""
    
    return final_prompt

def expand_query(question: str) -> List[str]:
    """查询扩展和重写"""
    try:
        # 基本查询扩展
        expanded_queries = [question]
        
        # 关键词同义词映射
        synonyms_map = {
            "minimum premium": ["minimum premium", "min premium", "lowest premium", "最低保费", "最少保费"],
            "最低保费": ["minimum premium", "min premium", "lowest premium", "最低保费", "最少保费"],
            "age range": ["age range", "issue age", "eligible age", "投保年龄", "年龄范围"],
            "投保年龄": ["age range", "issue age", "eligible age", "投保年龄", "年龄范围"],
            "death benefit": ["death benefit", "death settlement", "mortality benefit", "身故赔付", "死亡保险金"],
            "身故赔付": ["death benefit", "death settlement", "mortality benefit", "身故赔付", "死亡保险金"],
            "cash value": ["cash value", "surrender value", "guaranteed cash value", "现金价值", "退保价值"],
            "现金价值": ["cash value", "surrender value", "guaranteed cash value", "现金价值", "退保价值"],
        }
        
        # 按关键词扩展
        for keyword, synonyms in synonyms_map.items():
            if keyword.lower() in question.lower():
                for synonym in synonyms:
                    if synonym not in expanded_queries:
                        expanded_queries.append(question.replace(keyword, synonym))
        
        # 添加不同表达形式
        if "what is" in question.lower():
            expanded_queries.append(question.replace("what is", "tell me about"))
            expanded_queries.append(question.replace("what is", "explain"))
        
        logger.info(f"查询扩展: {len(expanded_queries)} 个变体")
        return expanded_queries[:5]  # 限制数量
    except Exception as e:
        logger.error(f"查询扩展失败: {e}")
        return [question]

def extract_keywords(text: str) -> List[str]:
    """提取关键词"""
    import re
    
    # 保险相关关键词
    insurance_terms = [
        "premium", "保费", "policy", "保单", "coverage", "保障", 
        "benefit", "利益", "claim", "理赔", "deductible", "免赔额",
        "cash value", "现金价值", "death benefit", "身故赔付",
        "surrender", "退保", "dividend", "分红", "guaranteed", "保证"
    ]
    
    # 提取文本中的关键词
    words = re.findall(r'\b\w+\b', text.lower())
    keywords = []
    
    for word in words:
        if len(word) > 3 and (word in insurance_terms or any(term in word for term in insurance_terms)):
            keywords.append(word)
    
    # 添加数字相关的关键词
    numbers = re.findall(r'\d+', text)
    keywords.extend(numbers)
    
    return list(set(keywords))[:5]  # 最多返回5个关键词

def adaptive_retrieval_strategy(vector_store, question: str, history: List = None) -> Dict:
    """自适应检索策略"""
    try:
        # 分析问题类型
        question_type = analyze_question_type(question)
        
        # 根据历史表现调整参数
        retrieval_params = get_adaptive_params(question_type, history)
        
        # 执行检索
        docs = advanced_retrieval_strategies(vector_store, question, k=retrieval_params['k'])
        
        # 性能监控
        performance_metrics = monitor_retrieval_performance(docs, question, question_type)
        
        return {
            "documents": docs,
            "question_type": question_type,
            "retrieval_params": retrieval_params,
            "performance_metrics": performance_metrics
        }
        
    except Exception as e:
        logger.error(f"自适应检索失败: {e}")
        return {
            "documents": vector_store.similarity_search(question, k=20),
            "question_type": "general",
            "retrieval_params": {"k": 20},
            "performance_metrics": {}
        }

def analyze_question_type(question: str) -> str:
    """分析问题类型"""
    question_lower = question.lower()
    
    # 具体数值查询
    if any(word in question_lower for word in ["minimum", "maximum", "最低", "最高", "多少", "how much", "what is the"]):
        return "specific_value"
    
    # 比较类查询
    if any(word in question_lower for word in ["compare", "difference", "vs", "versus", "比较", "区别", "不同"]):
        return "comparison"
    
    # 解释类查询
    if any(word in question_lower for word in ["explain", "what", "why", "how", "解释", "什么", "为什么", "如何"]):
        return "explanation"
    
    # 列表类查询
    if any(word in question_lower for word in ["list", "types", "kinds", "列出", "种类", "类型"]):
        return "listing"
    
    return "general"

def get_adaptive_params(question_type: str, history: List = None) -> Dict:
    """根据问题类型和历史表现获取自适应参数"""
    # 默认参数
    default_params = {
        "specific_value": {"k": 25, "mmr_lambda": 0.5, "keyword_weight": 0.3},
        "comparison": {"k": 35, "mmr_lambda": 0.8, "keyword_weight": 0.2},
        "explanation": {"k": 30, "mmr_lambda": 0.7, "keyword_weight": 0.2},
        "listing": {"k": 40, "mmr_lambda": 0.9, "keyword_weight": 0.1},
        "general": {"k": 25, "mmr_lambda": 0.7, "keyword_weight": 0.25}
    }
    
    params = default_params.get(question_type, default_params["general"])
    
    # 根据历史表现调整
    if history:
        recent_performance = [qa.get("confidence", 0.5) for qa in history[-5:] if qa.get("question_type") == question_type]
        if recent_performance:
            avg_confidence = sum(recent_performance) / len(recent_performance)
            
            # 如果表现不佳，增加检索数量
            if avg_confidence < 0.6:
                params["k"] = min(50, int(params["k"] * 1.3))
                params["keyword_weight"] *= 1.2
            # 如果表现很好，可以稍微减少检索数量以提高速度
            elif avg_confidence > 0.8:
                params["k"] = max(15, int(params["k"] * 0.9))
    
    return params

def monitor_retrieval_performance(docs: List, question: str, question_type: str) -> Dict:
    """监控检索性能"""
    try:
        metrics = {
            "total_documents": len(docs),
            "question_type": question_type,
            "avg_content_length": sum(len(doc.page_content) for doc in docs) / len(docs) if docs else 0,
            "strategies_distribution": {},
            "keyword_coverage": 0.0,
            "diversity_score": 0.0
        }
        
        # 统计策略分布
        strategies = [doc.metadata.get('strategy', 'unknown') for doc in docs]
        for strategy in set(strategies):
            metrics["strategies_distribution"][strategy] = strategies.count(strategy)
        
        # 计算关键词覆盖率
        question_keywords = set(extract_keywords(question))
        if question_keywords:
            doc_contents = " ".join([doc.page_content for doc in docs])
            covered_keywords = sum(1 for kw in question_keywords if kw.lower() in doc_contents.lower())
            metrics["keyword_coverage"] = covered_keywords / len(question_keywords)
        
        # 计算多样性分数（基于文档长度差异）
        if len(docs) > 1:
            content_lengths = [len(doc.page_content) for doc in docs]
            avg_length = sum(content_lengths) / len(content_lengths)
            variance = sum((length - avg_length) ** 2 for length in content_lengths) / len(content_lengths)
            metrics["diversity_score"] = min(1.0, variance / 10000)  # 归一化
        
        return metrics
        
    except Exception as e:
        logger.error(f"性能监控失败: {e}")
        return {"error": str(e)}

def advanced_retrieval_strategies(vector_store, question: str, k: int = 30) -> List:
    """高级多策略检索融合"""
    try:
        all_docs = []
        expanded_queries = expand_query(question)
        
        # 策略1: 基础相似度搜索
        logger.info("执行基础相似度搜索...")
        for query in expanded_queries:
            docs = vector_store.similarity_search(query, k=k//len(expanded_queries))
            for doc in docs:
                if not hasattr(doc, 'metadata'):
                    doc.metadata = {}
                doc.metadata['strategy'] = 'similarity'
                doc.metadata['query_used'] = query
                all_docs.append(doc)
        
        # 策略2: 最大边际相关性搜索
        logger.info("执行MMR搜索...")
        for query in expanded_queries:
            try:
                mmr_docs = vector_store.max_marginal_relevance_search(
                    query, 
                    k=k//len(expanded_queries), 
                    fetch_k=k*2,
                    lambda_mult=0.7  # 平衡相关性和多样性
                )
                for doc in mmr_docs:
                    if not hasattr(doc, 'metadata'):
                        doc.metadata = {}
                    doc.metadata['strategy'] = 'mmr'
                    doc.metadata['query_used'] = query
                    all_docs.append(doc)
            except Exception as mmr_error:
                logger.warning(f"MMR搜索失败: {mmr_error}")
        
        # 策略3: 带分数的相似度搜索
        logger.info("执行带分数的相似度搜索...")
        try:
            scored_docs = vector_store.similarity_search_with_score(question, k=k//2)
            for doc, score in scored_docs:
                if not hasattr(doc, 'metadata'):
                    doc.metadata = {}
                doc.metadata['strategy'] = 'scored_similarity'
                doc.metadata['similarity_score'] = 1 - score  # 转换为相似度
                doc.metadata['query_used'] = question
                all_docs.append(doc)
        except Exception as score_error:
            logger.warning(f"带分数搜索失败: {score_error}")
        
        # 策略4: 关键词增强搜索
        logger.info("执行关键词增强搜索...")
        keywords = extract_keywords(question)
        for keyword in keywords:
            try:
                keyword_docs = vector_store.similarity_search(keyword, k=max(1, k//8))
                for doc in keyword_docs:
                    if not hasattr(doc, 'metadata'):
                        doc.metadata = {}
                    doc.metadata['strategy'] = 'keyword'
                    doc.metadata['keyword_used'] = keyword
                    all_docs.append(doc)
            except Exception as kw_error:
                logger.warning(f"关键词搜索失败 ({keyword}): {kw_error}")
        
        # 去重处理
        unique_docs = deduplicate_docs(all_docs)
        
        logger.info(f"多策略检索: 原始{len(all_docs)} -> 去重后{len(unique_docs)}")
        return unique_docs[:k]
        
    except Exception as e:
        logger.error(f"高级检索失败: {e}")
        return vector_store.similarity_search(question, k=k)

def deduplicate_docs(docs: List) -> List:
    """去重文档"""
    try:
        seen_hashes = set()
        unique_docs = []
        
        for doc in docs:
            # 创建内容哈希
            content_hash = hash(doc.page_content[:200])  # 使用前200字符
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_docs.append(doc)
        
        # 按策略权重排序
        strategy_weights = {
            'similarity': 0.4,
            'mmr': 0.3,
            'scored_similarity': 0.2,
            'keyword': 0.1
        }
        
        def get_doc_score(doc):
            strategy = doc.metadata.get('strategy', 'similarity')
            base_score = strategy_weights.get(strategy, 0.25)
            
            # 如果有相似度分数，使用它
            if 'similarity_score' in doc.metadata:
                return doc.metadata['similarity_score'] * base_score
            
            # 内容长度加分
            content_bonus = min(0.1, len(doc.page_content) / 5000)
            return base_score + content_bonus
        
        # 排序
        unique_docs.sort(key=get_doc_score, reverse=True)
        
        return unique_docs
        
    except Exception as e:
        logger.error(f"去重失败: {e}")
        return docs

def answer_question_with_confidence(question: str, vector_store) -> Dict:
    """回答问题并返回置信度"""
    try:
        logger.info(f"开始回答问题: {question}")
        
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("未设置OpenAI API密钥")
            return {
                "question": question,
                "answer": "请先设置OpenAI API密钥",
                "sources": [],
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat()
            }
        
        # 增强的系统提示，支持中英文混合
        system_prompt = """You are an insurance product expert assistant. 
        Please answer questions based on the provided context. 
        If the context is in Chinese, you can answer in Chinese.
        If you cannot find relevant information, please say so clearly.
        Always cite the specific information from the context."""
        
        logger.info("初始化LLM...")
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=800  # 增加token限制
        )
        
        logger.info("开始自适应检索策略...")
        start_time = time.time()
        
        # 使用自适应检索策略
        retrieval_result = adaptive_retrieval_strategy(vector_store, question, st.session_state.qa_history)
        retrieved_docs = retrieval_result["documents"]
        question_type = retrieval_result["question_type"]
        performance_metrics = retrieval_result["performance_metrics"]
        
        # 构建上下文
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # 使用高级提示词工程，根据问题类型调整
        prompt = create_advanced_prompt(question, context, question_type)
        
        # 生成答案
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        response_time = time.time() - start_time
        
        logger.info(f"答案生成完成，耗时: {response_time:.2f}秒")
        logger.info(f"答案内容: {answer[:200]}...")
        
        source_documents = retrieved_docs
        logger.info(f"检索到 {len(source_documents)} 个相关文档")
        
        # 更精细的置信度计算
        confidence = 0.3  # 基础置信度
        if source_documents:
            # 根据文档数量调整置信度
            doc_count = len(source_documents)
            if doc_count >= 8:
                confidence = 0.95
            elif doc_count >= 5:
                confidence = 0.85
            elif doc_count >= 3:
                confidence = 0.75
            elif doc_count >= 1:
                confidence = 0.6
        
        # 处理所有源文档（最多15个）
        sources = []
        for i, doc in enumerate(source_documents[:15], 1):
            # 增加显示的文本长度到500字符
            content = doc.page_content
            # 清理文本，去除多余的换行和空格
            content = ' '.join(content.split())
            
            logger.debug(f"源文档 {i}: {content[:100]}...")
            
            sources.append({
                "doc_name": doc.metadata.get("source", "Unknown"),
                "content": content[:500] + ("..." if len(content) > 500 else ""),
                "full_content": content,  # 保存完整内容
                "score": 0.95 - (i * 0.05),  # 根据排序给出递减的相关度分数
                "index": i
            })
        
        logger.info(f"置信度: {confidence:.2f}, 源文档数: {len(sources)}")
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "response_time": response_time,
            "timestamp": datetime.now().isoformat(),
            "search_strategy": "adaptive_retrieval",
            "question_type": question_type,
            "expanded_queries": expand_query(question),
            "retrieval_stats": {
                "total_docs_retrieved": len(source_documents),
                "strategies_used": ["similarity", "mmr", "scored_similarity", "keyword"],
                "keywords_extracted": extract_keywords(question)
            },
            "performance_metrics": performance_metrics
        }
    except Exception as e:
        logger.exception(f"回答问题时出错: {str(e)}")
        return {
            "question": question,
            "answer": f"错误: {str(e)}",
            "sources": [],
            "confidence": 0.0,
            "response_time": 0,
            "timestamp": datetime.now().isoformat()
        }

def display_confidence_badge(confidence: float):
    """显示置信度标签"""
    percentage = confidence * 100
    if confidence >= 0.7:
        badge_class = "confidence-high"
        emoji = "✅"
    elif confidence >= 0.3:
        badge_class = "confidence-medium"
        emoji = "⚠️"
    else:
        badge_class = "confidence-low"
        emoji = "❌"
    
    st.markdown(
        f'<span class="{badge_class}">{emoji} 置信度: {percentage:.1f}%</span>',
        unsafe_allow_html=True
    )

# 主界面
st.title("🤖 保险产品智能问答系统")
st.markdown("### 基于RAG技术的企业级PoC演示 | 支持中英文智能问答")

# 侧边栏
with st.sidebar:
    st.header("⚙️ 系统配置")
    
    # 显示日志查看器
    with st.expander("📋 查看调试日志", expanded=False):
        if st.button("🔄 刷新日志"):
            try:
                with open('rag_debug.log', 'r', encoding='utf-8') as f:
                    logs = f.readlines()
                    # 显示最后50行
                    recent_logs = logs[-50:] if len(logs) > 50 else logs
                    st.text_area(
                        "最近日志",
                        value=''.join(recent_logs),
                        height=300,
                        disabled=True
                    )
            except FileNotFoundError:
                st.info("日志文件尚未创建")
            except Exception as e:
                st.error(f"读取日志失败: {e}")
        
        if st.button("🗑️ 清空日志"):
            try:
                open('rag_debug.log', 'w').close()
                st.success("日志已清空")
            except Exception as e:
                st.error(f"清空日志失败: {e}")
    
    st.divider()
    
    # API密钥管理
    api_key = initialize_openai()
    
    # 创建一个可折叠的API配置区域
    with st.expander("🔑 API密钥配置", expanded=not bool(api_key)):
        if api_key:
            # 显示当前API密钥状态
            st.success("✅ API密钥已配置")
            # 显示部分隐藏的密钥
            masked_key = api_key[:10] + "..." + api_key[-4:] if len(api_key) > 14 else "***"
            st.info(f"当前密钥: {masked_key}")
            
            # 提供重新配置选项
            if st.button("🔄 重新配置API密钥", use_container_width=True):
                # 清除现有密钥
                if 'api_key' in st.session_state:
                    del st.session_state.api_key
                os.environ.pop("OPENAI_API_KEY", None)
                st.rerun()
        
        # API密钥输入框
        new_api_key = st.text_input(
            "输入新的OpenAI API Key",
            type="password",
            help="格式: sk-...",
            placeholder="sk-proj-..."
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 保存密钥", use_container_width=True, type="primary"):
                if new_api_key and new_api_key.startswith("sk-"):
                    os.environ["OPENAI_API_KEY"] = new_api_key
                    st.session_state.api_key = new_api_key
                    st.success("✅ API密钥已更新")
                    time.sleep(1)
                    st.rerun()
                elif new_api_key:
                    st.error("❌ 密钥格式不正确，应以'sk-'开头")
        
        with col2:
            # 从.env文件重新加载
            if st.button("📂 从.env加载", use_container_width=True):
                try:
                    from dotenv import load_dotenv
                    load_dotenv(override=True)
                    env_key = os.getenv("OPENAI_API_KEY")
                    if env_key:
                        os.environ["OPENAI_API_KEY"] = env_key
                        st.session_state.api_key = env_key
                        st.success("✅ 已从.env文件加载密钥")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ .env文件中未找到OPENAI_API_KEY")
                except Exception as e:
                    st.error(f"❌ 加载失败: {str(e)}")
        
        # 测试API连接
        if api_key or new_api_key:
            if st.button("🧪 测试API连接", use_container_width=True):
                test_key = new_api_key if new_api_key else api_key
                with st.spinner("正在测试连接..."):
                    try:
                        # 测试API连接
                        from openai import OpenAI
                        client = OpenAI(api_key=test_key)
                        # 发送一个简单的测试请求
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": "Hi"}],
                            max_tokens=5
                        )
                        st.success("✅ API连接成功！")
                        st.info(f"模型响应: {response.choices[0].message.content}")
                    except Exception as e:
                        st.error(f"❌ API连接失败: {str(e)}")
                        if "api_key" in str(e).lower():
                            st.warning("💡 提示: 请检查API密钥是否正确")
                        elif "rate" in str(e).lower():
                            st.warning("💡 提示: API调用频率限制，请稍后再试")
                        else:
                            st.warning("💡 提示: 请检查网络连接或API密钥权限")
    
    st.divider()
    
    # 文件上传
    st.header("📄 文档管理")
    uploaded_files = st.file_uploader(
        "选择PDF文件",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.info(f"已选择 {len(uploaded_files)} 个文件")
        
        # PDF诊断功能
        if st.checkbox("📋 启用PDF诊断 (推荐)", value=True):
            st.markdown("##### 📊 文件诊断结果")
            
            all_diagnoses = []
            for file in uploaded_files:
                with st.expander(f"📄 {file.name}"):
                    try:
                        diagnosis = diagnose_pdf(file)
                        display_pdf_diagnosis(diagnosis)
                        all_diagnoses.append(diagnosis)
                    except Exception as e:
                        st.error(f"诊断失败: {str(e)}")
                        all_diagnoses.append(None)
            
            # 整体建议
            processable_files = sum(1 for d in all_diagnoses if d and should_process_pdf(d))
            if processable_files < len(uploaded_files):
                st.warning(f"⚠️ {len(uploaded_files) - processable_files} 个文件可能无法正常处理")
            
            # 显示处理信息
            if all_diagnoses:
                max_timeout = max([get_processing_timeout(d) for d in all_diagnoses if d], default=300)
                if max_timeout > 300:
                    st.warning(f"⚠️ 检测到大文件，建议的处理时间为 {max_timeout//60} 分钟，但系统将使用5分钟超时")
                else:
                    st.info("💡 文件大小适中，5分钟处理时间应该足够")
        
        # 选择处理模式
        col1, col2 = st.columns(2)
        with col1:
            processing_mode = st.selectbox(
                "选择处理模式",
                ["快速模式 (推荐)", "标准模式", "并行模式"],
                help="快速模式：最快但不提取表格\n标准模式：完整功能但较慢\n并行模式：多线程处理"
            )
        with col2:
            st.info(f"💡 当前模式: {processing_mode}")
        
        # 检查PDF处理依赖
        deps_ok = check_dependencies()
        if not deps_ok:
            st.warning("⚠️ 缺少PDF处理库，请运行: pip install pdfplumber PyPDF2")
        
        if st.button("🔄 处理文档", use_container_width=True, type="primary", disabled=not deps_ok):
            if not api_key:
                st.error("❌ 请先设置OpenAI API密钥")
            elif not deps_ok:
                st.error("❌ 缺少必要的PDF处理库，请检查依赖安装")
            else:
                logger.info(f"开始处理 {len(uploaded_files)} 个PDF文档")
                
                # 设置默认超时时间为10分钟
                timeout_setting = 600  # 10分钟
                
                all_documents = []
                failed_files = []
                
                for i, file in enumerate(uploaded_files):
                    st.info(f"📄 正在处理文件 {i+1}/{len(uploaded_files)}: {file.name}")
                    
                    try:
                        # 根据模式选择处理器
                        if "快速" in processing_mode:
                            logger.info(f"使用快速模式处理 {file.name}")
                            docs = process_pdf_fast(file)
                        elif "并行" in processing_mode:
                            logger.info(f"使用并行模式处理 {file.name}")
                            docs = process_pdf_parallel(file)
                        else:
                            logger.info(f"使用标准模式处理 {file.name}")
                            docs = process_pdf_with_timeout(file, timeout_seconds=timeout_setting)
                        
                        if docs:
                            all_documents.extend(docs)
                            st.success(f"✅ {file.name} 处理成功: {len(docs)} 个文本块")
                        else:
                            failed_files.append(file.name)
                            st.error(f"❌ {file.name} 处理失败")
                            
                    except Exception as e:
                        failed_files.append(file.name)
                        logger.error(f"处理文件 {file.name} 时出错: {e}")
                        st.error(f"❌ {file.name} 处理出错: {str(e)}")
                
                # 处理结果汇总
                if all_documents:
                    st.session_state.documents = all_documents
                    st.session_state.vector_store = create_vector_store(all_documents)
                    
                    logger.info(f"文档处理完成，共 {len(all_documents)} 个文本块")
                    
                    # 显示处理统计
                    success_count = len(uploaded_files) - len(failed_files)
                    st.success(f"🎉 处理完成!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("成功处理", f"{success_count}/{len(uploaded_files)}")
                    with col2:
                        st.metric("文档数", len(set([d["metadata"]["source"] for d in all_documents])))
                    with col3:
                        st.metric("文本块数", len(all_documents))
                    
                    if failed_files:
                        st.warning(f"⚠️ 以下文件处理失败: {', '.join(failed_files)}")
                    
                    st.balloons()
                else:
                    logger.error("没有成功处理任何文档")
                    st.error("❌ 所有文档处理都失败了")
                    
                    # 提供故障排除建议
                    with st.expander("🛠️ 故障排除建议"):
                        st.markdown("""
                        **可能的原因和解决方案:**
                        
                        1. **文件损坏或加密**: 尝试用其他PDF阅读器打开文件
                        2. **文件过大**: 尝试拆分大文件或增加超时时间
                        3. **复杂PDF格式**: 某些PDF可能包含复杂的图形或表格
                        4. **扫描版PDF**: 系统无法处理图片版PDF，需要OCR工具
                        5. **网络问题**: 检查网络连接是否稳定
                        
                        **建议操作:**
                        - 尝试处理单个文件
                        - 增加处理超时时间
                        - 检查PDF文件是否可以正常打开
                        - 尝试将PDF转换为文本格式
                        """)

# 主要内容区域
if st.session_state.vector_store:
    # 显示系统状态
    status_cols = st.columns(5)
    with status_cols[0]:
        st.metric("文档数", len(set([d["metadata"]["source"] for d in st.session_state.documents])))
    with status_cols[1]:
        st.metric("文本块", len(st.session_state.documents))
    with status_cols[2]:
        st.metric("历史问答", len(st.session_state.qa_history))
    with status_cols[3]:
        avg_confidence = sum([qa["confidence"] for qa in st.session_state.qa_history[-10:]]) / min(10, len(st.session_state.qa_history)) if st.session_state.qa_history else 0
        st.metric("平均置信度", f"{avg_confidence:.1%}")
    with status_cols[4]:
        # 显示最常见的问题类型
        if st.session_state.qa_history:
            question_types = [qa.get("question_type", "general") for qa in st.session_state.qa_history[-10:]]
            most_common_type = max(set(question_types), key=question_types.count) if question_types else "general"
            st.metric("主要问题类型", most_common_type.replace("_", " ").title())
        else:
            st.metric("系统状态", "✅ 就绪")
    
    # 自定义导航栏
    st.markdown("---")
    nav_cols = st.columns(4)
    
    pages = ["💬 智能问答", "📋 问题分类", "📊 批量问答", "📈 性能监控"]
    
    for i, page in enumerate(pages):
        with nav_cols[i]:
            if st.button(page, key=f"nav_{i}", use_container_width=True,
                        type="primary" if st.session_state.current_page == page.split()[1] else "secondary"):
                st.session_state.current_page = page.split()[1]
                st.rerun()
    
    st.markdown("---")
    
    # 根据当前页面显示内容
    if st.session_state.current_page == "智能问答":
        # 如果有待处理的问题
        if st.session_state.auto_submit and st.session_state.selected_question:
            st.info(f"💡 正在回答问题：{st.session_state.selected_question}")
            
            with st.spinner("🤔 正在分析..."):
                result = answer_question_with_confidence(
                    st.session_state.selected_question, 
                    st.session_state.vector_store
                )
                st.session_state.qa_history.insert(0, result)
                st.session_state.last_answer = result
                st.session_state.auto_submit = False
            
            # 显示答案
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("### 📝 AI回答")
            with col2:
                display_confidence_badge(result["confidence"])
            
            st.markdown(f"""
            <div class="answer-card">
                {result["answer"]}
            </div>
            """, unsafe_allow_html=True)
            
            # 显示所有信息来源（最多10个）
            if result["sources"]:
                st.markdown("### 📚 信息来源")
                st.info(f"找到 {len(result['sources'])} 个相关文档片段")
                
                for source in result["sources"]:
                    score = source.get('score', 0)
                    if score >= 0.8:
                        score_color = "🟢"
                    elif score >= 0.6:
                        score_color = "🟡"
                    else:
                        score_color = "🔴"
                    
                    with st.expander(
                        f"{score_color} 来源 {source.get('index', '')} | {source['doc_name']} | 相关度: {score:.2%} | 块: {source.get('chunk_index', 'N/A')}",
                        expanded=(source.get('index', 0) <= 3)
                    ):
                        st.markdown("**文档内容片段：**")
                        st.text_area(
                            "",
                            value=source.get('full_content', source['content']),
                            height=150,
                            disabled=True,
                            key=f"auto_source_{source.get('index', '')}_{id(source)}"
                        )
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption(f"📄 文档: {source['doc_name']}")
                        with col2:
                            st.caption(f"📊 相关度得分: {score:.2%}")
            
            st.divider()
        
        # 热门问题
        st.markdown("#### 🔥 热门问题")
        cols = st.columns(len(HOT_QUESTIONS))
        for i, (label, question) in enumerate(HOT_QUESTIONS):
            with cols[i]:
                if st.button(label, key=f"hot_{i}", use_container_width=True):
                    st.session_state.selected_question = question
                    st.session_state.auto_submit = True
                    st.rerun()
        
        st.divider()
        
        # 显示最近的回答（如果有）
        if st.session_state.last_answer and not st.session_state.auto_submit:
            st.markdown("### 📌 最近回答")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"问题: {st.session_state.last_answer['question']}")
            with col2:
                display_confidence_badge(st.session_state.last_answer["confidence"])
            
            st.markdown(f"""
            <div class="answer-card">
                {st.session_state.last_answer["answer"]}
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()
        
        # 问题输入
        col1, col2 = st.columns([4, 1])
        with col1:
            question = st.text_input(
                "输入您的问题",
                value="" if st.session_state.auto_submit else st.session_state.selected_question,
                placeholder="例如: What is the minimum premium?",
                key="question_input"
            )
        with col2:
            submit_button = st.button("🔍 提问", type="primary", use_container_width=True)
        
        # 处理提问
        if submit_button:
            if question and question.strip():
                st.info(f"📝 正在处理问题: {question}")
                
                try:
                    with st.spinner("🤔 正在思考中..."):
                        result = answer_question_with_confidence(question, st.session_state.vector_store)
                        st.session_state.qa_history.insert(0, result)
                        st.session_state.last_answer = result
                    
                    # 显示答案
                    st.success("✅ 回答成功")
                    st.markdown("---")
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown("### 📝 AI回答")
                    with col2:
                        display_confidence_badge(result["confidence"])
                    
                    st.markdown(f"""
                    <div class="answer-card">
                        {result["answer"]}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 显示响应时间
                    if "response_time" in result:
                        st.caption(f"⏱️ 响应时间: {result['response_time']:.2f}秒")
                    
                    # 显示所有信息来源（最多10个）
                    if result["sources"]:
                        st.markdown("### 📚 信息来源")
                        st.info(f"找到 {len(result['sources'])} 个相关文档片段")
                        
                        # 显示每个来源
                        for source in result["sources"]:
                            # 使用不同的颜色标记相关度
                            score = source.get('score', 0)
                            if score >= 0.8:
                                score_color = "🟢"  # 高相关度
                            elif score >= 0.6:
                                score_color = "🟡"  # 中相关度
                            else:
                                score_color = "🔴"  # 低相关度
                            
                            # 创建可展开的区域显示每个来源
                            with st.expander(
                                f"{score_color} 来源 {source.get('index', '')} | {source['doc_name']} | 相关度: {score:.2%} | 块: {source.get('chunk_index', 'N/A')}",
                                expanded=(source.get('index', 0) <= 3)  # 默认展开前3个
                            ):
                                # 显示内容
                                st.markdown("**文档内容片段：**")
                                st.text_area(
                                    "",
                                    value=source.get('full_content', source['content']),
                                    height=150,
                                    disabled=True,
                                    key=f"source_content_{source.get('index', '')}_{id(source)}"
                                )
                                
                                # 显示元信息
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.caption(f"📄 文档: {source['doc_name']}")
                                with col2:
                                    st.caption(f"📊 相关度得分: {score:.2%}")
                
                except Exception as e:
                    st.error(f"❌ 处理问题时出错: {str(e)}")
                    st.exception(e)
            else:
                st.warning("⚠️ 请输入问题")
    
    elif st.session_state.current_page == "问题分类":
        st.markdown("### 📋 可回答问题列表")
        st.success("💡 点击问题将自动跳转到智能问答页面并获取答案")
        
        for category_name, category_data in QUESTION_CATEGORIES.items():
            st.markdown(f"""
            <div class="category-header">
                {category_name} {category_data['completion']}
            </div>
            """, unsafe_allow_html=True)
            
            for cn_name, en_question, is_answerable in category_data["questions"]:
                col1, col2 = st.columns([4, 1])
                with col1:
                    if is_answerable:
                        if st.button(f"✅ {cn_name}", key=f"cat_{en_question}", 
                                   use_container_width=True):
                            # 设置问题并切换页面
                            st.session_state.selected_question = en_question
                            st.session_state.auto_submit = True
                            st.session_state.current_page = "智能问答"
                            st.rerun()
                    else:
                        st.button(f"❌ {cn_name}", key=f"cat_{en_question}", 
                                use_container_width=True, disabled=True)
                with col2:
                    if is_answerable:
                        st.success("可回答")
                    else:
                        st.error("不可用")
    
    elif st.session_state.current_page == "批量问答":
        st.markdown("### 🚀 批量回答预设问题")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("开始批量问答", type="primary", use_container_width=True):
                all_questions = []
                for category_data in QUESTION_CATEGORIES.values():
                    for cn_name, en_question, is_answerable in category_data["questions"]:
                        if is_answerable:
                            all_questions.append((cn_name, en_question))
                
                progress = st.progress(0)
                batch_results = []
                
                for i, (cn_name, question) in enumerate(all_questions):
                    with st.spinner(f"处理: {cn_name}"):
                        result = answer_question_with_confidence(question, st.session_state.vector_store)
                        st.session_state.qa_history.insert(0, result)
                        batch_results.append((cn_name, result))
                        progress.progress((i + 1) / len(all_questions))
                
                # 显示批量结果统计
                st.success(f"✅ 完成 {len(all_questions)} 个问题")
                
                # 召回率分析
                high_conf = sum(1 for _, r in batch_results if r["confidence"] >= 0.7)
                med_conf = sum(1 for _, r in batch_results if 0.4 <= r["confidence"] < 0.7)
                low_conf = sum(1 for _, r in batch_results if r["confidence"] < 0.4)
                
                st.markdown("#### 📊 召回质量分析")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("高置信度 (≥70%)", high_conf, f"{high_conf/len(all_questions)*100:.1f}%")
                with col2:
                    st.metric("中置信度 (40-70%)", med_conf, f"{med_conf/len(all_questions)*100:.1f}%")
                with col3:
                    st.metric("低置信度 (<40%)", low_conf, f"{low_conf/len(all_questions)*100:.1f}%")
        
        with col2:
            if st.button("📈 召回率测试", type="secondary", use_container_width=True):
                st.markdown("#### 🔍 召回率测试结果")
                
                # 测试查询集合
                test_queries = [
                    ("最低保费测试", ["minimum premium", "最低保费", "min premium"]),
                    ("投保年龄测试", ["age range", "投保年龄", "issue age"]),
                    ("现金价值测试", ["cash value", "现金价值", "surrender value"]),
                ]
                
                for test_name, queries in test_queries:
                    st.markdown(f"**{test_name}**")
                    
                    all_docs = []
                    for query in queries:
                        docs = st.session_state.vector_store.similarity_search(query, k=5)
                        all_docs.extend(docs)
                    
                    # 去重
                    unique_docs = []
                    seen = set()
                    for doc in all_docs:
                        doc_id = hash(doc.page_content[:50])
                        if doc_id not in seen:
                            seen.add(doc_id)
                            unique_docs.append(doc)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"查询变体: {len(queries)} 个")
                    with col2:
                        st.success(f"检索文档: {len(unique_docs)} 个")
                    
                    # 显示部分结果
                    for i, doc in enumerate(unique_docs[:3]):
                        with st.expander(f"文档片段 {i+1}"):
                            st.text(doc.page_content[:300] + "...")
                    
                    st.divider()
    
    elif st.session_state.current_page == "性能监控":
        st.markdown("### 📈 系统性能监控")
        
        if st.session_state.qa_history:
            # 整体性能统计
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_queries = len(st.session_state.qa_history)
                st.metric("总查询数", total_queries)
            
            with col2:
                avg_response_time = sum([qa.get("response_time", 0) for qa in st.session_state.qa_history]) / total_queries
                st.metric("平均响应时间", f"{avg_response_time:.2f}秒")
            
            with col3:
                high_confidence_queries = sum(1 for qa in st.session_state.qa_history if qa.get("confidence", 0) >= 0.7)
                success_rate = high_confidence_queries / total_queries
                st.metric("高置信度率", f"{success_rate:.1%}")
            
            with col4:
                question_types = [qa.get("question_type", "general") for qa in st.session_state.qa_history]
                unique_types = len(set(question_types))
                st.metric("问题类型数", unique_types)
            
            st.divider()
            
            # 问题类型分析
            st.markdown("#### 📊 问题类型分析")
            col1, col2 = st.columns(2)
            
            with col1:
                # 问题类型分布
                from collections import Counter
                type_counts = Counter(question_types)
                
                st.markdown("**问题类型分布：**")
                for qtype, count in type_counts.most_common():
                    percentage = count / total_queries * 100
                    st.text(f"• {qtype.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
            
            with col2:
                # 各类型平均置信度
                st.markdown("**各类型平均置信度：**")
                type_confidence = {}
                for qtype in set(question_types):
                    confidences = [qa["confidence"] for qa in st.session_state.qa_history if qa.get("question_type") == qtype]
                    if confidences:
                        avg_conf = sum(confidences) / len(confidences)
                        type_confidence[qtype] = avg_conf
                        st.text(f"• {qtype.replace('_', ' ').title()}: {avg_conf:.1%}")
            
            st.divider()
            
            # 检索策略效果分析
            st.markdown("#### 🔍 检索策略效果")
            
            retrieval_stats = []
            for qa in st.session_state.qa_history:
                if "performance_metrics" in qa and qa["performance_metrics"]:
                    metrics = qa["performance_metrics"]
                    retrieval_stats.append({
                        "question_type": qa.get("question_type", "general"),
                        "confidence": qa.get("confidence", 0),
                        "keyword_coverage": metrics.get("keyword_coverage", 0),
                        "diversity_score": metrics.get("diversity_score", 0),
                        "total_documents": metrics.get("total_documents", 0)
                    })
            
            if retrieval_stats:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_coverage = sum([s["keyword_coverage"] for s in retrieval_stats]) / len(retrieval_stats)
                    st.metric("平均关键词覆盖率", f"{avg_coverage:.1%}")
                
                with col2:
                    avg_diversity = sum([s["diversity_score"] for s in retrieval_stats]) / len(retrieval_stats)
                    st.metric("平均内容多样性", f"{avg_diversity:.2f}")
                
                with col3:
                    avg_docs = sum([s["total_documents"] for s in retrieval_stats]) / len(retrieval_stats)
                    st.metric("平均检索文档数", f"{avg_docs:.1f}")
            
            st.divider()
            
            # 最近查询详情
            st.markdown("#### 📋 最近查询详情")
            
            for i, qa in enumerate(st.session_state.qa_history[:5], 1):
                with st.expander(f"查询 {i}: {qa['question'][:50]}..."):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.text(f"问题类型: {qa.get('question_type', 'general')}")
                        st.text(f"置信度: {qa.get('confidence', 0):.1%}")
                        st.text(f"响应时间: {qa.get('response_time', 0):.2f}秒")
                    
                    with col2:
                        if "performance_metrics" in qa:
                            metrics = qa["performance_metrics"]
                            st.text(f"关键词覆盖: {metrics.get('keyword_coverage', 0):.1%}")
                            st.text(f"检索文档数: {metrics.get('total_documents', 0)}")
                            if "strategies_distribution" in metrics:
                                strategies = ", ".join([f"{k}:{v}" for k, v in metrics["strategies_distribution"].items()])
                                st.text(f"策略分布: {strategies}")
        else:
            st.info("暂无性能数据")
    
    elif st.session_state.current_page == "历史记录":
        st.markdown("### 📜 问答历史")
        
        if st.session_state.qa_history:
            for item in st.session_state.qa_history[:10]:
                with st.expander(f"❓ {item['question'][:50]}..."):
                    st.write(f"**答案:** {item['answer']}")
                    if 'confidence' in item:
                        display_confidence_badge(item['confidence'])
                    if 'question_type' in item:
                        st.caption(f"问题类型: {item['question_type']} | 时间: {item['timestamp']}")
                    else:
                        st.caption(f"时间: {item['timestamp']}")
        else:
            st.info("暂无历史记录")

else:
    st.info("👆 请先在侧边栏上传PDF文档开始使用")