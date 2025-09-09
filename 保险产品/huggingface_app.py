"""
保险产品RAG系统 - HuggingFace Spaces免费部署版本
使用Gradio界面，完全免费托管
"""
import gradio as gr
import os
from typing import List, Dict, Tuple
import json
from datetime import datetime
import pandas as pd
import pypdf2
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# 全局变量
vector_store = None
qa_history = []
documents = []

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

def process_pdf(pdf_file) -> str:
    """处理上传的PDF文件"""
    global documents, vector_store
    
    if pdf_file is None:
        return "请先上传PDF文件"
    
    try:
        # 提取PDF文本
        text = ""
        with pdfplumber.open(pdf_file.name) as pdf:
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
        documents = [{"content": chunk, "metadata": {"source": os.path.basename(pdf_file.name)}} 
                    for chunk in chunks]
        
        # 创建向量存储
        embeddings = OpenAIEmbeddings()
        texts = [doc["content"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        
        vector_store = FAISS.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas
        )
        
        return f"✅ 成功处理PDF: {os.path.basename(pdf_file.name)}\n共创建 {len(chunks)} 个文本块"
    
    except Exception as e:
        return f"❌ 处理PDF时出错: {str(e)}"

def answer_question(question: str, api_key: str) -> Tuple[str, str]:
    """回答单个问题"""
    global vector_store, qa_history
    
    if not api_key:
        return "请先输入OpenAI API密钥", ""
    
    if vector_store is None:
        return "请先上传并处理PDF文档", ""
    
    if not question:
        return "请输入问题", ""
    
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
            retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True
        )
        
        result = qa_chain({"query": question})
        answer = result["result"]
        sources = [doc.metadata.get("source", "Unknown") for doc in result.get("source_documents", [])][:3]
        
        # 保存到历史
        qa_history.append({
            "question": question,
            "answer": answer,
            "sources": sources,
            "timestamp": datetime.now().isoformat()
        })
        
        sources_text = f"\n\n📚 来源: {', '.join(set(sources))}" if sources else ""
        
        return answer, sources_text
    
    except Exception as e:
        return f"❌ 错误: {str(e)}", ""

def batch_answer(api_key: str, progress=gr.Progress()) -> str:
    """批量回答预设问题"""
    global vector_store
    
    if not api_key:
        return "请先输入OpenAI API密钥"
    
    if vector_store is None:
        return "请先上传并处理PDF文档"
    
    results = []
    for i, question in enumerate(progress.tqdm(PRESET_QUESTIONS, desc="处理问题")):
        answer, sources = answer_question(question, api_key)
        results.append(f"**Q{i+1}: {question}**\n{answer}\n")
    
    return "\n---\n".join(results)

def export_results() -> Tuple[str, str]:
    """导出问答结果"""
    global qa_history
    
    if not qa_history:
        return None, None
    
    # 生成CSV
    df = pd.DataFrame(qa_history)
    csv_path = f"qa_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_path, index=False)
    
    # 生成JSON
    json_path = f"qa_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(qa_history, f, ensure_ascii=False, indent=2)
    
    return csv_path, json_path

def create_interface():
    """创建Gradio界面"""
    with gr.Blocks(title="保险产品RAG系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🏥 保险产品智能问答系统
        ### 基于RAG技术的保险文档智能分析 - 免费托管版本
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ 配置")
                api_key = gr.Textbox(
                    label="OpenAI API Key",
                    type="password",
                    placeholder="sk-...",
                    info="您的API密钥仅在会话中使用，不会被存储"
                )
                
                gr.Markdown("### 📄 文档上传")
                pdf_file = gr.File(
                    label="上传PDF文件",
                    file_types=[".pdf"],
                    type="filepath"
                )
                
                process_btn = gr.Button("🔄 处理文档", variant="primary")
                process_status = gr.Textbox(label="处理状态", interactive=False)
                
                gr.Markdown("### 📊 快速操作")
                batch_btn = gr.Button("🚀 批量回答预设问题", variant="secondary")
                export_btn = gr.Button("💾 导出结果", variant="secondary")
                
            with gr.Column(scale=2):
                gr.Markdown("### 💬 问答界面")
                
                with gr.Tab("自定义问题"):
                    custom_question = gr.Textbox(
                        label="输入您的问题",
                        placeholder="例如: What is the minimum premium?",
                        lines=2
                    )
                    custom_submit = gr.Button("🔍 获取答案")
                    
                with gr.Tab("预设问题"):
                    preset_question = gr.Dropdown(
                        label="选择预设问题",
                        choices=PRESET_QUESTIONS,
                        value=PRESET_QUESTIONS[0]
                    )
                    preset_submit = gr.Button("🔍 获取答案")
                
                answer_output = gr.Textbox(
                    label="答案",
                    lines=6,
                    interactive=False
                )
                
                sources_output = gr.Textbox(
                    label="来源",
                    lines=2,
                    interactive=False
                )
                
                with gr.Tab("批量结果"):
                    batch_output = gr.Markdown()
                
                with gr.Tab("导出"):
                    csv_file = gr.File(label="CSV文件")
                    json_file = gr.File(label="JSON文件")
        
        # 事件绑定
        process_btn.click(
            fn=process_pdf,
            inputs=[pdf_file],
            outputs=[process_status]
        )
        
        custom_submit.click(
            fn=answer_question,
            inputs=[custom_question, api_key],
            outputs=[answer_output, sources_output]
        )
        
        preset_submit.click(
            fn=answer_question,
            inputs=[preset_question, api_key],
            outputs=[answer_output, sources_output]
        )
        
        batch_btn.click(
            fn=batch_answer,
            inputs=[api_key],
            outputs=[batch_output]
        )
        
        export_btn.click(
            fn=export_results,
            inputs=[],
            outputs=[csv_file, json_file]
        )
        
        # 示例
        gr.Examples(
            examples=[
                ["What is the minimum premium?"],
                ["What are the death benefit options?"],
                ["What is the surrender value?"],
                ["What are the fees and charges?"],
            ],
            inputs=[custom_question]
        )
        
        gr.Markdown("""
        ---
        ### 📖 使用说明
        1. 输入OpenAI API密钥
        2. 上传保险产品PDF文件
        3. 点击"处理文档"按钮
        4. 输入问题或选择预设问题
        5. 获取AI生成的答案
        
        ### 🌟 特点
        - 完全免费托管在HuggingFace Spaces
        - 支持中英文PDF文档
        - 自动提取表格数据
        - 批量问答功能
        - 结果导出功能
        """)
    
    return demo

# 启动应用
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )