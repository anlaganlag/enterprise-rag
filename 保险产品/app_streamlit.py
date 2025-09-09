"""
保险产品RAG系统 - Streamlit版本
"""
import streamlit as st
import os
from typing import List, Dict
import json
from datetime import datetime
import pandas as pd
import PyPDF2
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# 页面配置
st.set_page_config(
    page_title="保险产品RAG系统",
    page_icon="🏥",
    layout="wide"
)

# 初始化session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'documents' not in st.session_state:
    st.session_state.documents = []

# 预设问题
PRESET_QUESTIONS = [
    "What is the minimum premium?",
    "What is the policy currency?",
    "What is the minimum issue age?",
    "What is the maximum issue age?",
    "What is the benefit term?",
    "What are the premium payment terms available?",
    "What premium payment modes are available?",
    "What is the minimum modal premium?",
    "What is the surrender value?",
    "What are the fees and charges?",
    "What is the free-look period?",
    "What are the exclusions?",
    "What are the riders available?",
    "What is the grace period for premium payment?",
    "What are the death benefit options?",
]

def process_pdf(pdf_file, api_key: str) -> str:
    """处理上传的PDF文件"""
    if not api_key:
        return "❌ 请先输入OpenAI API密钥"
    
    try:
        # 设置API密钥
        os.environ["OPENAI_API_KEY"] = api_key
        
        # 提取PDF文本
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
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
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(text)
        
        # 创建文档
        st.session_state.documents = [
            {"content": chunk, "metadata": {"source": pdf_file.name}} 
            for chunk in chunks
        ]
        
        # 创建向量存储
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        texts = [doc["content"] for doc in st.session_state.documents]
        metadatas = [doc["metadata"] for doc in st.session_state.documents]
        
        st.session_state.vector_store = FAISS.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas
        )
        
        return f"✅ 成功处理PDF: {pdf_file.name}\n共创建 {len(chunks)} 个文本块"
    
    except Exception as e:
        return f"❌ 处理PDF时出错: {str(e)}"

def answer_question(question: str, api_key: str) -> Dict:
    """回答单个问题"""
    if not api_key:
        return {"error": "请先输入OpenAI API密钥"}
    
    if st.session_state.vector_store is None:
        return {"error": "请先上传并处理PDF文档"}
    
    if not question:
        return {"error": "请输入问题"}
    
    try:
        os.environ["OPENAI_API_KEY"] = api_key
        
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=500
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True
        )
        
        result = qa_chain({"query": question})
        answer = result["result"]
        sources = [doc.metadata.get("source", "Unknown") for doc in result.get("source_documents", [])][:3]
        
        # 保存到历史
        st.session_state.qa_history.append({
            "question": question,
            "answer": answer,
            "sources": sources,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "answer": answer,
            "sources": sources
        }
    
    except Exception as e:
        return {"error": f"❌ 错误: {str(e)}"}

# Streamlit UI
st.title("🏥 保险产品智能问答系统")
st.markdown("### 基于RAG技术的保险文档智能分析")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 配置")
    
    # API密钥输入
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="您的API密钥仅在会话中使用，不会被存储"
    )
    
    # 文档上传
    st.header("📄 文档上传")
    pdf_file = st.file_uploader(
        "上传PDF文件",
        type=["pdf"],
        help="选择要分析的保险产品PDF文档"
    )
    
    if pdf_file is not None:
        if st.button("🔄 处理文档", type="primary"):
            with st.spinner("处理中..."):
                status = process_pdf(pdf_file, api_key)
                if "✅" in status:
                    st.success(status)
                else:
                    st.error(status)
    
    # 快速操作
    st.header("📊 快速操作")
    
    if st.button("💾 导出历史记录"):
        if st.session_state.qa_history:
            df = pd.DataFrame(st.session_state.qa_history)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="下载CSV",
                data=csv,
                file_name=f"qa_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            json_str = json.dumps(st.session_state.qa_history, ensure_ascii=False, indent=2)
            st.download_button(
                label="下载JSON",
                data=json_str,
                file_name=f"qa_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        else:
            st.info("暂无历史记录可导出")

# 主界面
tabs = st.tabs(["💬 问答", "📝 预设问题", "🚀 批量问答", "📜 历史记录"])

# 自定义问答标签
with tabs[0]:
    st.header("自定义问题")
    
    question = st.text_area(
        "输入您的问题",
        placeholder="例如: What is the minimum premium?",
        height=100
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔍 获取答案", type="primary"):
            if question:
                with st.spinner("思考中..."):
                    result = answer_question(question, api_key)
                    
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success("✅ 回答成功")
                        st.markdown("### 答案")
                        st.write(result["answer"])
                        
                        if result["sources"]:
                            st.markdown("### 📚 来源")
                            st.write(", ".join(set(result["sources"])))

# 预设问题标签
with tabs[1]:
    st.header("预设问题")
    
    selected_question = st.selectbox(
        "选择一个预设问题",
        PRESET_QUESTIONS
    )
    
    if st.button("🔍 回答预设问题", type="primary"):
        with st.spinner("思考中..."):
            result = answer_question(selected_question, api_key)
            
            if "error" in result:
                st.error(result["error"])
            else:
                st.success("✅ 回答成功")
                st.markdown(f"**问题:** {selected_question}")
                st.markdown("### 答案")
                st.write(result["answer"])
                
                if result["sources"]:
                    st.markdown("### 📚 来源")
                    st.write(", ".join(set(result["sources"])))

# 批量问答标签
with tabs[2]:
    st.header("批量回答预设问题")
    
    if st.button("🚀 开始批量问答", type="primary"):
        if not api_key:
            st.error("请先输入OpenAI API密钥")
        elif st.session_state.vector_store is None:
            st.error("请先上传并处理PDF文档")
        else:
            progress_bar = st.progress(0)
            results_container = st.container()
            
            for i, preset_q in enumerate(PRESET_QUESTIONS):
                progress_bar.progress((i + 1) / len(PRESET_QUESTIONS))
                
                result = answer_question(preset_q, api_key)
                
                with results_container:
                    st.markdown(f"---")
                    st.markdown(f"**Q{i+1}: {preset_q}**")
                    
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.write(result["answer"])
                        if result["sources"]:
                            st.caption(f"来源: {', '.join(set(result['sources']))}")
            
            st.success("✅ 批量问答完成！")

# 历史记录标签
with tabs[3]:
    st.header("问答历史")
    
    if st.session_state.qa_history:
        for i, item in enumerate(reversed(st.session_state.qa_history)):
            with st.expander(f"问题 {len(st.session_state.qa_history) - i}: {item['question'][:50]}..."):
                st.markdown(f"**时间:** {item['timestamp']}")
                st.markdown(f"**问题:** {item['question']}")
                st.markdown(f"**答案:** {item['answer']}")
                if item['sources']:
                    st.markdown(f"**来源:** {', '.join(set(item['sources']))}")
    else:
        st.info("暂无历史记录")

# 页脚
st.markdown("---")
st.markdown("""
### 📖 使用说明
1. **输入API密钥**: 在侧边栏输入您的OpenAI API密钥
2. **上传文档**: 选择保险产品PDF文件
3. **处理文档**: 点击"处理文档"按钮
4. **提问**: 输入问题或选择预设问题
5. **导出结果**: 下载CSV或JSON格式的结果

### 🌟 特点
- ✅ 支持中英文PDF文档
- ✅ 自动提取表格数据
- ✅ 批量问答功能
- ✅ 结果导出功能
""")