"""
改进的PDF处理模块
解决PDF处理超时和卡死问题
"""
import streamlit as st
import pandas as pd
import logging
from typing import List, Dict
import time
import threading
import queue
import os

logger = logging.getLogger(__name__)

def process_pdf_with_timeout(uploaded_file, timeout_seconds=600) -> List[Dict]:
    """
    带超时机制的PDF处理函数
    默认超时时间10分钟
    """
    logger.info(f"开始处理PDF文件 (超时限制: {timeout_seconds//60}分钟): {uploaded_file.name}")
    
    # 使用队列在线程间传递结果
    result_queue = queue.Queue()
    
    def process_worker():
        """工作线程函数"""
        try:
            logger.info("PDF处理线程开始")
            result = process_pdf_safe(uploaded_file)
            
            if result and len(result) > 0:
                logger.info(f"PDF处理成功，返回 {len(result)} 个文本块")
                result_queue.put(("success", result))
            else:
                logger.warning("PDF处理返回空结果")
                result_queue.put(("error", "PDF处理未返回有效内容"))
                
        except Exception as e:
            logger.exception(f"PDF处理异常: {e}")
            result_queue.put(("error", str(e)))
        finally:
            logger.info("PDF处理线程结束")
    
    # 启动处理线程
    worker_thread = threading.Thread(target=process_worker)
    worker_thread.daemon = True
    worker_thread.start()
    
    # 等待结果或超时
    start_time = time.time()
    progress_placeholder = st.empty()
    
    while True:
        elapsed = time.time() - start_time
        
        # 检查是否超时
        if elapsed >= timeout_seconds:
            logger.error(f"PDF处理超时 (已用时: {elapsed:.1f}秒)")
            progress_placeholder.empty()
            st.error(f"⚠️ PDF处理超时 (>{timeout_seconds//60}分钟)，可能是文件过大或格式复杂")
            return []
        
        # 尝试获取结果
        try:
            status, result = result_queue.get(timeout=0.5)
            progress_placeholder.empty()
            
            if status == "success":
                logger.info(f"PDF处理成功完成 (用时: {elapsed:.1f}秒)")
                st.success(f"✅ PDF处理成功 (用时: {int(elapsed)}秒)")
                return result
            else:
                logger.error(f"PDF处理失败: {result}")
                st.error(f"PDF处理失败: {result}")
                return []
                
        except queue.Empty:
            # 更新进度显示
            elapsed_min = int(elapsed // 60)
            elapsed_sec = int(elapsed % 60)
            timeout_min = int(timeout_seconds // 60)
            
            if elapsed_min > 0:
                progress_text = f"正在处理PDF... ({elapsed_min}分{elapsed_sec}秒/{timeout_min}分钟)"
            else:
                progress_text = f"正在处理PDF... ({elapsed_sec}秒/{timeout_min}分钟)"
            
            progress_placeholder.info(progress_text)
            
            # 检查线程是否还活着
            if not worker_thread.is_alive():
                logger.warning("处理线程已结束但未返回结果")
                # 再尝试获取一次结果
                try:
                    status, result = result_queue.get_nowait()
                    progress_placeholder.empty()
                    if status == "success":
                        logger.info(f"PDF处理成功完成 (用时: {elapsed:.1f}秒)")
                        st.success(f"✅ PDF处理成功 (用时: {int(elapsed)}秒)")
                        return result
                except queue.Empty:
                    logger.error("处理线程结束但没有结果")
                    progress_placeholder.empty()
                    st.error("❌ PDF处理异常结束")
                    return []
    
    # 超时处理
    logger.error("PDF处理超时")
    st.error(f"⚠️ PDF处理超时 (>{timeout_seconds//60}分钟)，可能是文件过大或格式复杂")
    return []

def process_pdf_safe(uploaded_file) -> List[Dict]:
    """
    安全的PDF处理函数 - 多重fallback策略
    """
    chunks = []
    
    # 策略1: 尝试使用 pdfplumber (推荐)
    try:
        logger.info("尝试使用 pdfplumber 处理PDF")
        return process_with_pdfplumber(uploaded_file)
    except Exception as e:
        logger.warning(f"pdfplumber 处理失败: {e}")
    
    # 策略2: 回退到 PyPDF2
    try:
        logger.info("回退到 PyPDF2 处理PDF")
        return process_with_pypdf2(uploaded_file)
    except Exception as e:
        logger.warning(f"PyPDF2 处理失败: {e}")
    
    # 策略3: 最后尝试简单文本提取
    try:
        logger.info("使用基础文本提取")
        return process_basic_text(uploaded_file)
    except Exception as e:
        logger.error(f"所有PDF处理方法都失败: {e}")
        return []

def process_with_pdfplumber(uploaded_file) -> List[Dict]:
    """使用 pdfplumber 处理PDF"""
    import pdfplumber
    
    chunks = []
    text = ""
    empty_pages = 0
    total_pages = 0
    
    with pdfplumber.open(uploaded_file) as pdf:
        total_pages = len(pdf.pages)
        logger.info(f"PDF总页数: {total_pages}")
        
        # 注意：在工作线程中不创建Streamlit UI元素
        # progress_bar = st.progress(0)  
        # status_text = st.empty()
        
        for page_num, page in enumerate(pdf.pages, 1):
            try:
                # 记录进度（不使用Streamlit UI）
                progress = page_num / total_pages
                if page_num % 10 == 0:  # 每10页记录一次
                    logger.info(f"处理进度: {page_num}/{total_pages} 页 ({progress:.1%})")
                
                # 提取文本
                page_text = page.extract_text() or ""
                text_length = len(page_text.strip())
                # 只在有问题时记录，减少日志开销
                if text_length == 0:
                    logger.warning(f"第 {page_num} 页未提取到文本")
                
                if not page_text.strip():
                    empty_pages += 1
                    logger.warning(f"第 {page_num} 页未提取到文本内容")
                # 移除预览日志以提高性能
                
                text += f"\n[Page {page_num}]\n{page_text}\n"
                
                # 提取表格（限制处理时间）
                try:
                    tables = page.extract_tables()
                    if tables:
                        logger.info(f"第 {page_num} 页发现 {len(tables)} 个表格")
                        for table_idx, table in enumerate(tables[:3]):  # 限制最多3个表格
                            if table:
                                df = pd.DataFrame(table[1:], columns=table[0])
                                text += f"\n[Table on Page {page_num}]\n{df.to_string()}\n"
                except Exception as table_error:
                    logger.warning(f"第 {page_num} 页表格提取失败: {table_error}")
                
                # 每10页检查一次时间
                if page_num % 10 == 0:
                    time.sleep(0.1)  # 给UI更新时间
                
            except Exception as page_error:
                logger.error(f"处理第 {page_num} 页时出错: {page_error}")
                # 不在工作线程中使用st.warning
                continue
    
    # 统计信息
    logger.info(f"文档提取完成 - 总文本长度: {len(text)}, 空页面: {empty_pages}/{total_pages}")
    
    if empty_pages > 0:
        logger.warning(f"文档提取统计: 总页数 {total_pages}, 空页面 {empty_pages}, 有效页面 {total_pages - empty_pages}")
    
    if not text.strip():
        raise ValueError("未能从PDF中提取到任何文本内容")
    
    # 文本分块
    return create_text_chunks(text, uploaded_file.name)

def process_with_pypdf2(uploaded_file) -> List[Dict]:
    """使用 PyPDF2 处理PDF"""
    import PyPDF2
    
    text = ""
    
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        total_pages = len(pdf_reader.pages)
        logger.info(f"PyPDF2: PDF总页数: {total_pages}")
        
        for page_num, page in enumerate(pdf_reader.pages, 1):
            if page_num % 10 == 0:
                logger.info(f"PyPDF2处理进度: {page_num}/{total_pages}")
            
            try:
                page_text = page.extract_text()
                text += f"\n[Page {page_num}]\n{page_text}\n"
            except Exception as e:
                logger.warning(f"PyPDF2: 第 {page_num} 页提取失败: {e}")
                continue
        
        if not text.strip():
            raise ValueError("PyPDF2未能提取到文本")
        
        return create_text_chunks(text, uploaded_file.name)
        
    except Exception as e:
        logger.error(f"PyPDF2处理失败: {e}")
        raise

def process_basic_text(uploaded_file) -> List[Dict]:
    """基础文本提取（最后的fallback）"""
    logger.info("尝试基础文本提取")
    
    # 简单地尝试读取文件内容
    try:
        content = uploaded_file.read()
        if isinstance(content, bytes):
            # 尝试解码
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError:
                text = content.decode('latin-1', errors='ignore')
        else:
            text = str(content)
        
        if text.strip():
            return create_text_chunks(text, uploaded_file.name)
        else:
            raise ValueError("基础文本提取未获得内容")
            
    except Exception as e:
        logger.error(f"基础文本提取失败: {e}")
        raise

def create_text_chunks(text: str, filename: str) -> List[Dict]:
    """创建文本块"""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    
    # 过滤掉过短的chunks
    valid_chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]
    
    logger.info(f"文本分块完成: 原始 {len(chunks)} 块, 有效 {len(valid_chunks)} 块")
    
    # 移除chunk预览日志以提高性能
    
    return [{"content": chunk, "metadata": {"source": filename, "chunk_index": i}} 
            for i, chunk in enumerate(valid_chunks)]

# 检查所需的库是否可用
def check_dependencies():
    """检查依赖库"""
    missing_deps = []
    
    try:
        import pdfplumber
        logger.info("✅ pdfplumber 可用")
    except ImportError:
        missing_deps.append("pdfplumber")
        logger.warning("❌ pdfplumber 不可用")
    
    try:
        import PyPDF2
        logger.info("✅ PyPDF2 可用")
    except ImportError:
        missing_deps.append("PyPDF2")
        logger.warning("❌ PyPDF2 不可用")
    
    if missing_deps:
        logger.warning(f"缺少依赖库: {', '.join(missing_deps)}")
    
    return len(missing_deps) == 0

if __name__ == "__main__":
    # 测试依赖
    check_dependencies()