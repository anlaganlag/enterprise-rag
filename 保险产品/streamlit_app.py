"""
保险产品RAG系统 - Streamlit免费部署版本
优化了内存使用，适配Streamlit Cloud限制
"""
import streamlit as st
import os
from typing import List, Dict
import json
import hashlib
from datetime import datetime
import pandas as pd

# 核心库
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import pypdf2
import pdfplumber

# 页面配置
st.set_page_config(
    page_title="保险产品智能问答系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 样式定制
st.markdown("""
<style>
    .main { padding: 0rem 1rem; }
    .stAlert { padding: 1rem; margin: 1rem 0; }
    h1 { color: #1e3a8a; }
    .question-card {
        background: #f8fafc;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #3b82f6;
    }
    .answer-card {
        background: #f0f9ff;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
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

# 预定义问题列表
PRESET_QUESTIONS = [
    "What is the minimum premium?",
    "What is the policy currency?",
    "What is the minimum issue age?",
    "What is the maximum issue age?",
    "What is the benefit term?",
    "What are the premium payment terms available?",
    "What premium payment modes are available?",
    "What is the minimum modal premium?",
    "What is the minimum prepayment amount?",
    "What is the prepayment interest rate?",
    "What are the non-guaranteed benefits or returns?",
    "What are the guaranteed benefits or returns?",
    "What is the surrender value?",
    "What are the fees and charges?",
    "What is the free-look period?",
    "What is the premium holiday option?",
    "What are the exclusions?",
    "What are the riders available?",
    "What is the automatic premium loan?",
    "What is the grace period for premium payment?",
    "What are the paid-up options?",
    "What is the policy loan feature?",
    "What is the maximum policy loan amount?",
    "What is the policy loan interest rate?",
    "What are the partial withdrawal options?",
    "What is the minimum partial withdrawal amount?",
    "What is the maximum partial withdrawal amount?",
    "What are the settlement options?",
    "What are the tax benefits?",
    "What is the bonus structure?",
    "What are the death benefit options?",
    "What is the maturity benefit?",
    "What is the accidental death benefit?",
    "What are the total death benefit components?"
]

@st.cache_resource
def initialize_openai():
    """初始化OpenAI客户端"""
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    openai.api_key = api_key
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

def answer_question(question: str, vector_store) -> Dict:
    """回答问题"""
    try:
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
        
        result = qa_chain({"query": question})
        
        return {
            "question": question,
            "answer": result["result"],
            "sources": [doc.metadata.get("source", "Unknown") for doc in result.get("source_documents", [])][:3],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "question": question,
            "answer": f"错误: {str(e)}",
            "sources": [],
            "timestamp": datetime.now().isoformat()
        }

# 主界面
st.title("🏥 保险产品智能问答系统")
st.markdown("### 基于RAG技术的保险文档智能分析")

# 侧边栏
with st.sidebar:
    st.header("⚙️ 配置")
    
    # API密钥输入
    if not st.secrets.get("OPENAI_API_KEY"):
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            help="请输入您的OpenAI API密钥"
        )
        if api_key_input:
            os.environ["OPENAI_API_KEY"] = api_key_input
            st.success("API密钥已设置")
    
    st.divider()
    
    # 文件上传
    st.header("📄 文档上传")
    uploaded_files = st.file_uploader(
        "选择PDF文件",
        type=["pdf"],
        accept_multiple_files=True,
        help="支持上传多个PDF文件"
    )
    
    if uploaded_files:
        if st.button("🔄 处理文档", use_container_width=True):
            with st.spinner("正在处理PDF文档..."):
                all_documents = []
                progress = st.progress(0)
                
                for i, file in enumerate(uploaded_files):
                    docs = process_pdf(file)
                    all_documents.extend(docs)
                    progress.progress((i + 1) / len(uploaded_files))
                
                if all_documents:
                    st.session_state.documents = all_documents
                    st.session_state.vector_store = create_vector_store(all_documents)
                    st.success(f"✅ 成功处理 {len(uploaded_files)} 个文件，共 {len(all_documents)} 个文本块")
    
    # 显示状态
    st.divider()
    st.header("📊 系统状态")
    if st.session_state.vector_store:
        st.success("✅ 向量库已就绪")
        st.info(f"文档数: {len(st.session_state.documents)}")
    else:
        st.warning("⚠️ 请先上传并处理文档")
    
    # 清除历史
    if st.button("🗑️ 清除历史", use_container_width=True):
        st.session_state.qa_history = []
        st.rerun()

# 主要内容区域
if st.session_state.vector_store:
    # 创建两列布局
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 问答界面")
        
        # 问题输入方式选择
        input_mode = st.radio(
            "选择输入方式",
            ["自定义问题", "预设问题", "批量问答"],
            horizontal=True
        )
        
        if input_mode == "自定义问题":
            question = st.text_input(
                "输入您的问题",
                placeholder="例如: What is the minimum premium?"
            )
            
            if st.button("🔍 获取答案", use_container_width=True):
                if question:
                    with st.spinner("正在分析..."):
                        result = answer_question(question, st.session_state.vector_store)
                        st.session_state.qa_history.insert(0, result)
                        st.rerun()
        
        elif input_mode == "预设问题":
            selected_question = st.selectbox(
                "选择一个预设问题",
                PRESET_QUESTIONS
            )
            
            if st.button("🔍 获取答案", use_container_width=True):
                with st.spinner("正在分析..."):
                    result = answer_question(selected_question, st.session_state.vector_store)
                    st.session_state.qa_history.insert(0, result)
                    st.rerun()
        
        else:  # 批量问答
            if st.button("🚀 回答所有预设问题", use_container_width=True):
                progress = st.progress(0)
                status = st.empty()
                
                for i, question in enumerate(PRESET_QUESTIONS):
                    status.text(f"正在处理: {question[:50]}...")
                    result = answer_question(question, st.session_state.vector_store)
                    st.session_state.qa_history.insert(0, result)
                    progress.progress((i + 1) / len(PRESET_QUESTIONS))
                
                st.success(f"✅ 已完成 {len(PRESET_QUESTIONS)} 个问题")
                st.rerun()
        
        # 显示历史记录
        st.divider()
        st.header("📝 问答历史")
        
        for item in st.session_state.qa_history[:10]:  # 显示最近10条
            with st.expander(f"❓ {item['question'][:100]}...", expanded=False):
                st.markdown(f"**答案:**")
                st.info(item['answer'])
                if item['sources']:
                    st.markdown(f"**来源:** {', '.join(item['sources'])}")
                st.caption(f"时间: {item['timestamp']}")
    
    with col2:
        st.header("📈 统计信息")
        
        # 统计卡片
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("总问答数", len(st.session_state.qa_history))
        with metric_col2:
            st.metric("文档块数", len(st.session_state.documents))
        
        # 导出功能
        st.divider()
        st.header("💾 导出结果")
        
        if st.session_state.qa_history:
            # 转换为DataFrame
            df = pd.DataFrame(st.session_state.qa_history)
            
            # CSV下载
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 下载CSV",
                data=csv,
                file_name=f"qa_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # JSON下载
            json_str = json.dumps(st.session_state.qa_history, ensure_ascii=False, indent=2)
            st.download_button(
                label="📥 下载JSON",
                data=json_str,
                file_name=f"qa_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

else:
    # 欢迎页面
    st.info("👆 请先在侧边栏上传PDF文档开始使用")
    
    # 使用说明
    with st.expander("📖 使用说明", expanded=True):
        st.markdown("""
        ### 快速开始
        1. **配置API密钥**: 在侧边栏输入OpenAI API密钥
        2. **上传文档**: 选择保险产品PDF文件
        3. **处理文档**: 点击"处理文档"按钮
        4. **提问**: 输入问题或选择预设问题
        5. **导出结果**: 下载CSV或JSON格式的结果
        
        ### 功能特点
        - ✅ 支持中英文PDF文档
        - ✅ 自动提取表格数据
        - ✅ 34个预设保险问题
        - ✅ 批量问答功能
        - ✅ 结果导出功能
        
        ### 注意事项
        - 请确保PDF文件小于10MB
        - API调用会产生费用
        - 建议使用预设问题以获得最佳效果
        """)

# 页脚
st.divider()
st.caption("🔒 数据安全 | 📧 技术支持 | v1.0.0")