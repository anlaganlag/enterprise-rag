"""
PDF处理模块 - 提取和处理PDF文档
"""
import logging
from pathlib import Path
from typing import List, Dict, Any
import PyPDF2
import pdfplumber
from tqdm import tqdm

logger = logging.getLogger(__name__)

class PDFProcessor:
    """PDF文档处理器"""
    
    def __init__(self):
        self.documents = []
        
    def extract_text_pypdf2(self, pdf_path: Path) -> str:
        """使用PyPDF2提取PDF文本"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                logger.info(f"正在处理 {pdf_path.name}, 共 {num_pages} 页")
                
                for page_num in tqdm(range(num_pages), desc=f"提取 {pdf_path.name}"):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- 第 {page_num + 1} 页 ---\n"
                        text += page_text
                        
        except Exception as e:
            logger.error(f"PyPDF2提取失败: {e}")
            
        return text
    
    def extract_tables_pdfplumber(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """使用pdfplumber提取表格数据"""
        tables_data = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    
                    for table_idx, table in enumerate(tables):
                        if table:
                            tables_data.append({
                                "page": page_num + 1,
                                "table_index": table_idx,
                                "data": table,
                                "source": pdf_path.name
                            })
                            
        except Exception as e:
            logger.error(f"pdfplumber表格提取失败: {e}")
            
        return tables_data
    
    def process_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """处理单个PDF文件"""
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
            
        logger.info(f"开始处理: {pdf_path}")
        
        # 提取文本
        text_content = self.extract_text_pypdf2(pdf_path)
        
        # 提取表格
        tables = self.extract_tables_pdfplumber(pdf_path)
        
        # 构建文档对象
        document = {
            "source": pdf_path.name,
            "text": text_content,
            "tables": tables,
            "metadata": {
                "file_path": str(pdf_path),
                "page_count": len(PyPDF2.PdfReader(open(pdf_path, 'rb')).pages),
                "has_tables": len(tables) > 0
            }
        }
        
        return document
    
    def process_multiple_pdfs(self, pdf_paths: List[Path]) -> List[Dict[str, Any]]:
        """处理多个PDF文件"""
        documents = []
        
        for pdf_path in pdf_paths:
            try:
                doc = self.process_pdf(pdf_path)
                documents.append(doc)
                logger.info(f"成功处理: {pdf_path.name}")
            except Exception as e:
                logger.error(f"处理 {pdf_path.name} 失败: {e}")
                
        return documents
    
    def create_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """将文本分割成块"""
        chunks = []
        text_length = len(text)
        
        for start in range(0, text_length, chunk_size - overlap):
            end = min(start + chunk_size, text_length)
            chunk = text[start:end]
            
            # 清理chunk
            chunk = chunk.strip()
            if chunk:
                chunks.append(chunk)
                
        return chunks
    
    def prepare_documents_for_indexing(self, documents: List[Dict[str, Any]], 
                                      chunk_size: int = 500, 
                                      overlap: int = 50) -> List[Dict[str, Any]]:
        """准备文档用于向量索引"""
        indexed_documents = []
        
        for doc in documents:
            # 处理文本内容
            if doc.get("text"):
                chunks = self.create_chunks(doc["text"], chunk_size, overlap)
                
                for chunk_idx, chunk in enumerate(chunks):
                    indexed_doc = {
                        "content": chunk,
                        "metadata": {
                            "source": doc["source"],
                            "chunk_index": chunk_idx,
                            "total_chunks": len(chunks),
                            **doc.get("metadata", {})
                        }
                    }
                    indexed_documents.append(indexed_doc)
            
            # 处理表格内容
            for table_info in doc.get("tables", []):
                # 将表格转换为文本格式
                table_text = self._table_to_text(table_info["data"])
                if table_text:
                    indexed_doc = {
                        "content": table_text,
                        "metadata": {
                            "source": doc["source"],
                            "type": "table",
                            "page": table_info["page"],
                            "table_index": table_info["table_index"]
                        }
                    }
                    indexed_documents.append(indexed_doc)
                    
        logger.info(f"准备了 {len(indexed_documents)} 个文档块用于索引")
        return indexed_documents
    
    def _table_to_text(self, table_data: List[List]) -> str:
        """将表格数据转换为文本"""
        if not table_data:
            return ""
            
        text_lines = []
        
        # 假设第一行是表头
        if len(table_data) > 0:
            headers = table_data[0]
            text_lines.append("表格内容:")
            text_lines.append(" | ".join(str(h) for h in headers if h))
            
            # 处理数据行
            for row in table_data[1:]:
                if row:
                    row_text = " | ".join(str(cell) for cell in row if cell)
                    if row_text.strip():
                        text_lines.append(row_text)
                        
        return "\n".join(text_lines)


if __name__ == "__main__":
    # 测试代码
    from config import PDF_FILES, CHUNK_SIZE, CHUNK_OVERLAP
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 初始化处理器
    processor = PDFProcessor()
    
    # 处理PDF文件
    pdf_paths = list(PDF_FILES.values())
    documents = processor.process_multiple_pdfs(pdf_paths)
    
    # 准备索引
    indexed_docs = processor.prepare_documents_for_indexing(
        documents, 
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )
    
    print(f"处理完成: {len(documents)} 个文档, {len(indexed_docs)} 个文档块")