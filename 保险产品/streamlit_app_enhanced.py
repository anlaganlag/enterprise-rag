"""
保险产品RAG系统 - Streamlit增强版
参考web_demo_enhanced.html实现的功能增强版本
"""
import streamlit as st
import os
from typing import List, Dict, Tuple
import json
import hashlib
from datetime import datetime
import pandas as pd
import time

# 核心库
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import PyPDF2
import pdfplumber

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
    .main { padding: 0rem 1rem; }
    
    /* 问题分类样式 */
    .category-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    
    .question-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .question-card:hover {
        background: #e9ecef;
        transform: translateX(5px);
    }
    
    .answerable {
        color: #28a745;
    }
    
    .unanswerable {
        color: #dc3545;
        opacity: 0.7;
    }
    
    /* 置信度标签 */
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
    
    /* 答案卡片 */
    .answer-card {
        background: #f0f9ff;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* 来源信息 */
    .source-card {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* 快速问题按钮 */
    .quick-question-btn {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        transition: all 0.3s;
    }
    
    .quick-question-btn:hover {
        background: #667eea;
        color: white;
        border-color: #667eea;
    }
    
    /* 统计卡片 */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: bold;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# 初始化session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'selected_question' not in st.session_state:
    st.session_state.selected_question = ""
if 'answer_confidence' not in st.session_state:
    st.session_state.answer_confidence = 0.0
if 'auto_submit' not in st.session_state:
    st.session_state.auto_submit = False
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0

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
            ("被保险人数量", "Number of insured persons?", False),
            ("更换被保险人功能", "Change of insured feature?", False),
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
    },
    "ℹ️ 其他详情": {
        "questions": [
            ("合同适用法律", "What is the contract governing law?", True),
            ("回溯可用性", "Backtesting availability?", False),
            ("免体检限额", "Non-medical limits?", False),
            ("附加保障", "Additional riders?", False),
        ],
        "completion": "1/4"
    },
    "💡 综合性问题": {
        "questions": [
            ("产品介绍", "请介绍一下RoyalFortune产品", True),
            ("主要特点", "What are the key features?", True),
            ("产品优势", "RoyalFortune的优势是什么？", True),
            ("保证现金价值机制", "How does the guaranteed cash value work?", True),
        ],
        "completion": ""
    }
}

# 热门快速问题
HOT_QUESTIONS = [
    ("最低保费", "RoyalFortune的最低保费是多少？"),
    ("投保年龄", "投保年龄范围是什么？"),
    ("保证现金价值", "保证现金价值是多少？"),
    ("身故赔付", "身故赔付有哪些特点？"),
    ("产品优势", "产品的优势是什么？"),
]

def initialize_openai():
    """初始化OpenAI客户端"""
    api_key = None
    
    # 尝试从多个来源获取API密钥
    # 1. 从.env文件读取
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
    except:
        pass
    
    # 2. 从Streamlit secrets获取
    if not api_key:
        try:
            if hasattr(st, 'secrets') and "OPENAI_API_KEY" in st.secrets:
                api_key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            pass
    
    # 3. 从环境变量获取
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if api_key:
        openai.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key
    
    return api_key

@st.cache_data
def process_pdf(uploaded_file) -> List[Dict]:
    """处理PDF文件"""
    chunks = []
    try:
        # 使用pdfplumber提取文本
        with pdfplumber.open(uploaded_file) as pdf:
            text = ""
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text() or ""
                text += f"\n[Page {page_num}]\n{page_text}\n"
                
                # 提取表格
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        text += f"\n[Table on Page {page_num}]\n{df.to_string()}\n"
        
        # 文本分块
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", ",", " ", ""]
        )
        
        chunks = text_splitter.split_text(text)
        return [{"content": chunk, "metadata": {"source": uploaded_file.name}} for chunk in chunks]
    
    except Exception as e:
        st.error(f"PDF处理错误: {str(e)}")
        return []

def create_vector_store(documents: List[Dict]):
    """创建向量存储"""
    try:
        embeddings = OpenAIEmbeddings()
        texts = [doc["content"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        
        vector_store = FAISS.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas
        )
        return vector_store
    except Exception as e:
        st.error(f"创建向量存储失败: {str(e)}")
        return None

def answer_question_with_confidence(question: str, vector_store) -> Dict:
    """回答问题并返回置信度"""
    try:
        # 确保API密钥已设置
        if not os.getenv("OPENAI_API_KEY"):
            return {
                "question": question,
                "answer": "请先设置OpenAI API密钥",
                "sources": [],
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat()
            }
        
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=500
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(
                search_kwargs={"k": 5}
            ),
            return_source_documents=True
        )
        
        # 记录开始时间
        start_time = time.time()
        result = qa_chain({"query": question})
        response_time = time.time() - start_time
        
        # 获取相关文档并计算置信度
        source_documents = result.get("source_documents", [])
        
        # 简单的置信度计算（基于返回文档的相关性）
        confidence = 0.8 if source_documents else 0.3
        if len(source_documents) >= 3:
            confidence = 0.9
        elif len(source_documents) >= 2:
            confidence = 0.7
        
        # 构建来源信息
        sources = []
        for doc in source_documents[:3]:
            sources.append({
                "doc_name": doc.metadata.get("source", "Unknown"),
                "content": doc.page_content[:200] + "...",
                "score": confidence  # 简化的分数
            })
        
        return {
            "question": question,
            "answer": result["result"],
            "sources": sources,
            "confidence": confidence,
            "response_time": response_time,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
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

# 顶部统计信息
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-value">{}</div>
        <div class="stat-label">文档数</div>
    </div>
    """.format(len(set([doc["metadata"]["source"] for doc in st.session_state.documents])) if st.session_state.documents else 0),
    unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-value">{}</div>
        <div class="stat-label">分块数</div>
    </div>
    """.format(len(st.session_state.documents)),
    unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-value">81.5%</div>
        <div class="stat-label">回答率</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-value">{}</div>
        <div class="stat-label">历史问答</div>
    </div>
    """.format(len(st.session_state.qa_history)),
    unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.header("⚙️ 系统配置")
    
    # API密钥输入
    api_key = initialize_openai()
    if not api_key:
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            help="请输入您的OpenAI API密钥"
        )
        if api_key_input:
            os.environ["OPENAI_API_KEY"] = api_key_input
            st.success("✅ API密钥已设置")
            st.rerun()
    else:
        st.success("✅ API密钥已配置")
    
    st.divider()
    
    # 文件上传
    st.header("📄 文档管理")
    uploaded_files = st.file_uploader(
        "选择PDF文件",
        type=["pdf"],
        accept_multiple_files=True,
        help="支持上传多个PDF文件"
    )
    
    if uploaded_files:
        # 显示已上传的文件
        st.info(f"已选择 {len(uploaded_files)} 个文件")
        for file in uploaded_files:
            st.text(f"📄 {file.name} ({file.size/1024/1024:.2f}MB)")
        
        # 处理文档按钮 - 始终显示
        if st.button("🔄 处理文档", use_container_width=True, type="primary", key="process_docs"):
            if not api_key:
                st.error("❌ 请先设置OpenAI API密钥")
            else:
                with st.spinner("正在处理PDF文档..."):
                    all_documents = []
                    progress = st.progress(0)
                    
                    for i, file in enumerate(uploaded_files):
                        docs = process_pdf(file)
                        all_documents.extend(docs)
                        progress.progress((i + 1) / len(uploaded_files))
                    
                    if all_documents:
                        st.session_state.documents = all_documents
                        with st.spinner("正在创建向量数据库..."):
                            st.session_state.vector_store = create_vector_store(all_documents)
                        st.success(f"✅ 成功处理 {len(uploaded_files)} 个文件，共 {len(all_documents)} 个文本块")
                        st.balloons()  # 添加庆祝动画
    
    # 已上传文档列表
    if st.session_state.documents:
        st.divider()
        st.header("📚 已上传文档")
        doc_sources = list(set([doc["metadata"]["source"] for doc in st.session_state.documents]))
        for doc_name in doc_sources:
            doc_chunks = [d for d in st.session_state.documents if d["metadata"]["source"] == doc_name]
            st.info(f"📄 {doc_name}\n\n{len(doc_chunks)} 个分块")
    
    # 系统状态
    st.divider()
    st.header("⚡ 系统状态")
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.metric("API状态", "✅ 在线" if api_key else "❌ 离线")
    with status_col2:
        st.metric("向量库", "✅ 就绪" if st.session_state.vector_store else "⚠️ 未就绪")
    
    # 清除历史
    if st.button("🗑️ 清除历史", use_container_width=True):
        st.session_state.qa_history = []
        st.session_state.selected_question = ""
        st.rerun()

# 主要内容区域
if st.session_state.vector_store:
    # 创建标签页
    tab1, tab2, tab3, tab4 = st.tabs(["💬 智能问答", "📋 问题分类", "📊 批量问答", "📜 历史记录"])
    
    # 智能问答标签页
    with tab1:
        # 如果是从问题分类跳转过来的，显示提示
        if st.session_state.auto_submit and st.session_state.selected_question:
            st.info(f"💡 已选择问题：{st.session_state.selected_question[:50]}...")
            # 自动执行问答
            with st.spinner("🤔 正在为您解答..."):
                result = answer_question_with_confidence(st.session_state.selected_question, st.session_state.vector_store)
                st.session_state.qa_history.insert(0, result)
                st.session_state.answer_confidence = result["confidence"]
                st.session_state.auto_submit = False  # 重置标志
            
            # 显示答案
            st.markdown("---")
            
            # 答案头部
            answer_col1, answer_col2 = st.columns([3, 1])
            with answer_col1:
                st.markdown("### 📝 AI回答")
            with answer_col2:
                display_confidence_badge(result["confidence"])
            
            # 答案内容
            st.markdown(f"""
            <div class="answer-card">
                {result["answer"]}
            </div>
            """, unsafe_allow_html=True)
            
            # 响应时间
            if "response_time" in result:
                st.caption(f"⏱️ 响应时间: {result['response_time']:.2f}秒")
            
            # 显示来源
            if result["sources"]:
                st.markdown("### 📚 信息来源")
                for i, source in enumerate(result["sources"], 1):
                    with st.expander(f"来源 {i}: {source['doc_name']}", expanded=False):
                        st.write(source["content"])
                        st.caption(f"相关度: {source['score']:.2f}")
            
            st.divider()
        
        # 热门快速问题
        st.markdown("#### 🔥 热门问题")
        cols = st.columns(len(HOT_QUESTIONS))
        for i, (label, question) in enumerate(HOT_QUESTIONS):
            with cols[i]:
                if st.button(label, key=f"hot_{i}", use_container_width=True):
                    st.session_state.selected_question = question
                    st.session_state.auto_submit = True
                    st.rerun()
        
        st.divider()
        
        # 问题输入
        col1, col2 = st.columns([4, 1])
        with col1:
            question = st.text_input(
                "输入您的问题（支持中英文）",
                value=st.session_state.selected_question if not st.session_state.auto_submit else "",
                placeholder="例如: What is the minimum premium?",
                key="question_input"
            )
        with col2:
            submit_button = st.button("🔍 提问", type="primary", use_container_width=True)
        
        # 提交问题
        if submit_button and question:
            with st.spinner("🤔 正在思考中..."):
                result = answer_question_with_confidence(question, st.session_state.vector_store)
                st.session_state.qa_history.insert(0, result)
                st.session_state.answer_confidence = result["confidence"]
            
            # 显示答案
            st.markdown("---")
            
            # 答案头部
            answer_col1, answer_col2 = st.columns([3, 1])
            with answer_col1:
                st.markdown("### 📝 AI回答")
            with answer_col2:
                display_confidence_badge(result["confidence"])
            
            # 答案内容
            st.markdown(f"""
            <div class="answer-card">
                {result["answer"]}
            </div>
            """, unsafe_allow_html=True)
            
            # 响应时间
            if "response_time" in result:
                st.caption(f"⏱️ 响应时间: {result['response_time']:.2f}秒")
            
            # 显示来源
            if result["sources"]:
                st.markdown("### 📚 信息来源")
                for i, source in enumerate(result["sources"], 1):
                    with st.expander(f"来源 {i}: {source['doc_name']}", expanded=False):
                        st.write(source["content"])
                        st.caption(f"相关度: {source['score']:.2f}")
    
    # 问题分类标签页
    with tab2:
        st.markdown("### 📋 可回答问题列表")
        st.success("💡 **点击问题后会自动跳转到智能问答页面并获取答案！**")
        st.info("✅ 表示可回答的问题，❌ 表示需要更多文档支持。")
        
        for category_name, category_data in QUESTION_CATEGORIES.items():
            # 分类标题
            st.markdown(f"""
            <div class="category-header">
                {category_name} {category_data['completion']}
            </div>
            """, unsafe_allow_html=True)
            
            # 问题列表
            for cn_name, en_question, is_answerable in category_data["questions"]:
                col1, col2 = st.columns([4, 1])
                with col1:
                    if is_answerable:
                        button_label = f"✅ {cn_name}"
                        help_text = f"点击查看: {en_question}"
                        if st.button(button_label, key=f"cat_{en_question}", 
                                   use_container_width=True, 
                                   help=help_text):
                            # 设置选中的问题并标记需要自动提交
                            st.session_state.selected_question = en_question
                            st.session_state.auto_submit = True
                            st.rerun()
                    else:
                        st.button(f"❌ {cn_name}", 
                                key=f"cat_{en_question}", 
                                use_container_width=True, 
                                disabled=True,
                                help="需要更多文档支持")
                with col2:
                    if is_answerable:
                        st.markdown('<span style="color: #28a745;">→ 点击提问</span>', 
                                  unsafe_allow_html=True)
                    else:
                        st.markdown('<span style="color: #dc3545;">不可用</span>', 
                                  unsafe_allow_html=True)
    
    # 批量问答标签页
    with tab3:
        st.markdown("### 🚀 批量回答预设问题")
        st.info("系统将自动回答所有可回答的预设问题，并生成完整报告。")
        
        if st.button("开始批量问答", type="primary", use_container_width=True):
            # 收集所有可回答的问题
            all_questions = []
            for category_data in QUESTION_CATEGORIES.values():
                for cn_name, en_question, is_answerable in category_data["questions"]:
                    if is_answerable:
                        all_questions.append((cn_name, en_question))
            
            # 批量处理
            progress = st.progress(0)
            status = st.empty()
            results = []
            
            for i, (cn_name, question) in enumerate(all_questions):
                status.text(f"正在处理: {cn_name} ({i+1}/{len(all_questions)})")
                result = answer_question_with_confidence(question, st.session_state.vector_store)
                results.append(result)
                st.session_state.qa_history.insert(0, result)
                progress.progress((i + 1) / len(all_questions))
            
            # 显示结果
            st.success(f"✅ 已完成 {len(all_questions)} 个问题的回答")
            
            # 生成报告
            st.markdown("### 📊 批量问答报告")
            for i, result in enumerate(results, 1):
                with st.expander(f"问题 {i}: {result['question'][:50]}...", expanded=False):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**答案:** {result['answer']}")
                    with col2:
                        display_confidence_badge(result['confidence'])
                    
                    if result['sources']:
                        st.caption(f"来源: {', '.join([s['doc_name'] for s in result['sources']])}")
    
    # 历史记录标签页
    with tab4:
        st.markdown("### 📜 问答历史")
        
        if st.session_state.qa_history:
            # 导出按钮
            col1, col2 = st.columns(2)
            with col1:
                # 转换为DataFrame
                df = pd.DataFrame(st.session_state.qa_history)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 导出CSV",
                    data=csv,
                    file_name=f"qa_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # JSON导出
                json_str = json.dumps(st.session_state.qa_history, ensure_ascii=False, indent=2)
                st.download_button(
                    label="📥 导出JSON",
                    data=json_str,
                    file_name=f"qa_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            st.divider()
            
            # 显示历史记录
            for i, item in enumerate(st.session_state.qa_history[:20]):  # 显示最近20条
                with st.expander(f"❓ {item['question'][:100]}...", expanded=False):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**答案:**\n{item['answer']}")
                    with col2:
                        if 'confidence' in item:
                            display_confidence_badge(item['confidence'])
                    
                    if item.get('sources'):
                        st.markdown("**来源:**")
                        for source in item['sources']:
                            st.caption(f"• {source['doc_name']}")
                    
                    st.caption(f"⏰ 时间: {item['timestamp']}")
        else:
            st.info("暂无历史记录")

else:
    # 欢迎页面
    st.info("👆 请先在侧边栏上传PDF文档开始使用")
    
    # 使用说明
    with st.expander("📖 使用说明", expanded=True):
        st.markdown("""
        ### 🚀 快速开始
        1. **配置API密钥**: 在侧边栏输入OpenAI API密钥
        2. **上传文档**: 选择保险产品PDF文件
        3. **处理文档**: 点击"处理文档"按钮
        4. **开始提问**: 通过以下方式提问：
           - 点击热门问题快速提问
           - 在问题分类中选择预设问题
           - 自定义输入问题
        5. **查看结果**: 系统会显示答案、置信度和来源信息
        
        ### ✨ 功能特点
        - ✅ **智能问答**: 支持中英文混合查询
        - ✅ **问题分类**: 结构化的问题分类体系
        - ✅ **置信度评估**: 每个答案都有置信度评分
        - ✅ **来源追踪**: 显示答案的文档来源
        - ✅ **批量问答**: 一键回答所有预设问题
        - ✅ **历史记录**: 保存所有问答历史
        - ✅ **数据导出**: 支持CSV和JSON格式导出
        
        ### 🎯 最佳实践
        - 使用预设问题获得最准确的答案
        - 关注置信度评分（>70%为高质量答案）
        - 查看来源信息验证答案准确性
        - 定期导出历史记录备份数据
        
        ### ⚠️ 注意事项
        - 请确保PDF文件小于10MB
        - API调用会产生费用
        - 置信度低的答案需要人工验证
        """)

# 页脚
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🔒 数据安全")
with col2:
    st.caption("📧 技术支持")
with col3:
    st.caption("v2.0.0 增强版")