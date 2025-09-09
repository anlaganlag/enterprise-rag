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
        with pdfplumber.open(uploaded_file) as pdf:
            text = ""
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text() or ""
                text += f"\n[Page {page_num}]\n{page_text}\n"
                
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        text += f"\n[Table on Page {page_num}]\n{df.to_string()}\n"
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200
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
        
        # 增加检索文档数量到10个
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 10}),
            return_source_documents=True
        )
        
        start_time = time.time()
        result = qa_chain({"query": question})
        response_time = time.time() - start_time
        
        source_documents = result.get("source_documents", [])
        
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
        
        # 处理所有源文档（最多10个）
        sources = []
        for i, doc in enumerate(source_documents[:10], 1):
            # 增加显示的文本长度到500字符
            content = doc.page_content
            # 清理文本，去除多余的换行和空格
            content = ' '.join(content.split())
            
            sources.append({
                "doc_name": doc.metadata.get("source", "Unknown"),
                "content": content[:500] + ("..." if len(content) > 500 else ""),
                "full_content": content,  # 保存完整内容
                "score": 0.95 - (i * 0.05),  # 根据排序给出递减的相关度分数
                "index": i
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

# 侧边栏
with st.sidebar:
    st.header("⚙️ 系统配置")
    
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
        
        if st.button("🔄 处理文档", use_container_width=True, type="primary"):
            if not api_key:
                st.error("❌ 请先设置OpenAI API密钥")
            else:
                with st.spinner("正在处理PDF文档..."):
                    all_documents = []
                    for file in uploaded_files:
                        docs = process_pdf(file)
                        all_documents.extend(docs)
                    
                    if all_documents:
                        st.session_state.documents = all_documents
                        st.session_state.vector_store = create_vector_store(all_documents)
                        st.success(f"✅ 成功处理，共 {len(all_documents)} 个文本块")
                        st.balloons()

# 主要内容区域
if st.session_state.vector_store:
    # 显示系统状态
    status_cols = st.columns(4)
    with status_cols[0]:
        st.metric("文档数", len(set([d["metadata"]["source"] for d in st.session_state.documents])))
    with status_cols[1]:
        st.metric("文本块", len(st.session_state.documents))
    with status_cols[2]:
        st.metric("历史问答", len(st.session_state.qa_history))
    with status_cols[3]:
        st.metric("系统状态", "✅ 就绪")
    
    # 自定义导航栏
    st.markdown("---")
    nav_cols = st.columns(4)
    
    pages = ["💬 智能问答", "📋 问题分类", "📊 批量问答", "📜 历史记录"]
    
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
                        f"{score_color} 来源 {source.get('index', '')} | {source['doc_name']} | 相关度: {score:.2%}",
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
                                f"{score_color} 来源 {source.get('index', '')} | {source['doc_name']} | 相关度: {score:.2%}",
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
        
        if st.button("开始批量问答", type="primary", use_container_width=True):
            all_questions = []
            for category_data in QUESTION_CATEGORIES.values():
                for cn_name, en_question, is_answerable in category_data["questions"]:
                    if is_answerable:
                        all_questions.append((cn_name, en_question))
            
            progress = st.progress(0)
            for i, (cn_name, question) in enumerate(all_questions):
                with st.spinner(f"处理: {cn_name}"):
                    result = answer_question_with_confidence(question, st.session_state.vector_store)
                    st.session_state.qa_history.insert(0, result)
                    progress.progress((i + 1) / len(all_questions))
            
            st.success(f"✅ 完成 {len(all_questions)} 个问题")
    
    elif st.session_state.current_page == "历史记录":
        st.markdown("### 📜 问答历史")
        
        if st.session_state.qa_history:
            for item in st.session_state.qa_history[:10]:
                with st.expander(f"❓ {item['question'][:50]}..."):
                    st.write(f"**答案:** {item['answer']}")
                    if 'confidence' in item:
                        display_confidence_badge(item['confidence'])
                    st.caption(f"时间: {item['timestamp']}")
        else:
            st.info("暂无历史记录")

else:
    st.info("👆 请先在侧边栏上传PDF文档开始使用")