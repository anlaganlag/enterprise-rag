"""
高性能PDF处理器
解决所有性能瓶颈问题
"""
import logging
from typing import List, Dict
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue

logger = logging.getLogger(__name__)

def process_pdf_fast(uploaded_file, timeout_seconds=600) -> List[Dict]:
    """
    高性能PDF处理 - 解决性能问题的根本方案
    
    优化策略：
    1. 禁用表格提取（最耗时的操作）
    2. 使用更快的PDF库
    3. 批量处理页面
    4. 优化文本分块
    5. 减少不必要的操作
    """
    import PyPDF2  # PyPDF2比pdfplumber快得多
    
    logger.info(f"使用快速模式处理PDF: {uploaded_file.name}")
    start_time = time.time()
    
    try:
        # 使用PyPDF2（更快）
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        total_pages = len(pdf_reader.pages)
        logger.info(f"PDF总页数: {total_pages}")
        
        # 收集所有文本（不处理表格，大幅提速）
        all_text = []
        
        # 批量处理，每10页为一批
        batch_size = 10
        for batch_start in range(0, total_pages, batch_size):
            batch_end = min(batch_start + batch_size, total_pages)
            batch_text = []
            
            for page_num in range(batch_start, batch_end):
                try:
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text:
                        batch_text.append(f"[Page {page_num + 1}]\n{text}")
                except Exception as e:
                    logger.warning(f"页面 {page_num + 1} 提取失败: {e}")
                    continue
            
            all_text.extend(batch_text)
            
            # 每处理一批记录进度
            logger.info(f"处理进度: {batch_end}/{total_pages} 页")
        
        # 合并所有文本
        full_text = "\n".join(all_text)
        
        # 快速分块（简化的分块策略）
        chunks = fast_text_split(full_text, uploaded_file.name)
        
        elapsed = time.time() - start_time
        logger.info(f"PDF处理完成，耗时: {elapsed:.2f}秒，生成 {len(chunks)} 个文本块")
        
        return chunks
        
    except Exception as e:
        logger.error(f"快速PDF处理失败: {e}")
        return []

def fast_text_split(text: str, filename: str) -> List[Dict]:
    """
    快速文本分块 - 优化的分块策略
    """
    # 使用更大的chunk_size减少分块数量
    chunk_size = 1500  # 增大到1500
    chunk_overlap = 200  # 减少重叠到200
    
    chunks = []
    text_length = len(text)
    
    # 简单的分块，避免复杂的正则处理
    for start in range(0, text_length, chunk_size - chunk_overlap):
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        
        # 只保留有意义的块
        if len(chunk.strip()) > 100:
            chunks.append({
                "content": chunk,
                "metadata": {
                    "source": filename,
                    "chunk_index": len(chunks),
                    "start_char": start,
                    "end_char": end
                }
            })
    
    logger.info(f"快速分块完成: 生成 {len(chunks)} 个块")
    return chunks

def process_pdf_parallel(uploaded_file, timeout_seconds=600) -> List[Dict]:
    """
    并行PDF处理 - 使用多线程加速
    """
    import PyPDF2
    
    logger.info(f"使用并行模式处理PDF: {uploaded_file.name}")
    start_time = time.time()
    
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        total_pages = len(pdf_reader.pages)
        logger.info(f"PDF总页数: {total_pages}")
        
        # 使用线程池并行处理
        max_workers = min(4, total_pages // 10)  # 最多4个线程
        page_texts = [None] * total_pages
        
        def process_page(page_num):
            try:
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                return page_num, f"[Page {page_num + 1}]\n{text}" if text else ""
            except Exception as e:
                logger.warning(f"页面 {page_num + 1} 处理失败: {e}")
                return page_num, ""
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_page, i): i for i in range(total_pages)}
            
            for future in as_completed(futures):
                page_num, text = future.result()
                page_texts[page_num] = text
                
                # 每处理20页记录一次
                if (page_num + 1) % 20 == 0:
                    logger.info(f"并行处理进度: {page_num + 1}/{total_pages}")
        
        # 合并文本
        full_text = "\n".join(filter(None, page_texts))
        
        # 快速分块
        chunks = fast_text_split(full_text, uploaded_file.name)
        
        elapsed = time.time() - start_time
        logger.info(f"并行处理完成，耗时: {elapsed:.2f}秒")
        
        return chunks
        
    except Exception as e:
        logger.error(f"并行处理失败: {e}")
        return []

def choose_best_processor(file_size_mb: float, page_count: int = None) -> str:
    """
    根据文件特征选择最佳处理器
    """
    if file_size_mb < 5:
        return "fast"  # 小文件用快速模式
    elif file_size_mb < 20:
        return "parallel"  # 中等文件用并行模式
    else:
        return "fast"  # 大文件也用快速模式，避免内存问题

# 性能对比测试
def benchmark_processors(pdf_file):
    """
    性能测试不同的处理器
    """
    import time
    
    results = {}
    
    # 测试快速模式
    start = time.time()
    chunks_fast = process_pdf_fast(pdf_file)
    results['fast'] = {
        'time': time.time() - start,
        'chunks': len(chunks_fast)
    }
    
    # 测试并行模式
    pdf_file.seek(0)  # 重置文件指针
    start = time.time()
    chunks_parallel = process_pdf_parallel(pdf_file)
    results['parallel'] = {
        'time': time.time() - start,
        'chunks': len(chunks_parallel)
    }
    
    return results