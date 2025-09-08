"""
简化改进版RAG - 回归基础，专注准确性
"""
import os
import json
import re
from pathlib import Path
from datetime import datetime
import PyPDF2
import pdfplumber
import requests
from tqdm import tqdm
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# 34个标准问题及其精确映射
QUESTION_MAPPINGS = {
    "Insurer Entity Name": {
        "keywords": ["insurer", "company", "underwritten by", "issued by", "AIA", "Sun Life"],
        "pattern": r"(?:issued by|underwritten by|insurer)[:\s]*([\w\s]+(?:Limited|Ltd|Company|Insurance))"
    },
    "Product Name": {
        "keywords": ["product name", "plan name", "RoyalFortune", "FlexiAchiever", "活享"],
        "pattern": r"(?:product|plan)\s*(?:name)?[:\s]*([\w\s]+)"
    },
    "Minimum Premium": {
        "keywords": ["minimum premium", "min premium", "USD125,000", "HKD1,000,000"],
        "pattern": r"(?:minimum|min).*?premium.*?(?:USD|US\$|HKD|HK\$)?\s*([\d,]+)"
    },
    "Maximum Premium": {
        "keywords": ["maximum premium", "max premium", "no maximum"],
        "pattern": r"(?:maximum|max).*?premium.*?(?:USD|US\$|HKD|HK\$)?\s*([\d,]+|no maximum)"
    },
    "Policy Currency": {
        "keywords": ["USD", "HKD", "currency", "US Dollar", "Hong Kong Dollar"],
        "pattern": r"(?:currency|currencies)[:\s]*(USD|HKD|US Dollar|Hong Kong Dollar)"
    },
    "Policy Term": {
        "keywords": ["policy term", "whole life", "years", "lifetime"],
        "pattern": r"(?:policy term|term)[:\s]*([\w\s]+|whole life|\d+\s*years?)"
    },
    "Issue Age": {
        "keywords": ["issue age", "age 0", "age 80", "0-80", "15 days"],
        "pattern": r"(?:issue age|age)[:\s]*(\d+).*?(?:to|-).*?(\d+)"
    }
}

class SimpleImprovedRAG:
    """简化改进版RAG"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("请设置OPENAI_API_KEY")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        self.documents = []
        self.embeddings = []
        self.total_cost = 0
        
    def extract_pdf_smart(self, pdf_path):
        """智能提取PDF - 更大的块，保持完整性"""
        print(f"提取: {Path(pdf_path).name}")
        chunks = []
        
        # 1. 先尝试提取表格
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    for table in tables:
                        if table and len(table) > 1:
                            # 表格转文本，保持结构
                            table_text = self._format_table(table)
                            if table_text:
                                chunks.append({
                                    "text": table_text,
                                    "source": Path(pdf_path).name,
                                    "page": page_num + 1,
                                    "type": "table"
                                })
        except:
            pass
        
        # 2. 提取正文 - 更大的块
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                full_text = ""
                
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        # 按页保存，不过度分割
                        chunks.append({
                            "text": text,
                            "source": Path(pdf_path).name,
                            "page": page_num + 1,
                            "type": "page"
                        })
                        full_text += text + "\n"
                
                # 额外保存完整文档用于全局搜索
                if full_text:
                    chunks.append({
                        "text": full_text[:8000],  # 限制长度
                        "source": Path(pdf_path).name,
                        "page": 0,
                        "type": "full"
                    })
        except Exception as e:
            print(f"提取错误: {e}")
        
        return chunks
    
    def _format_table(self, table):
        """格式化表格为易搜索的文本"""
        if not table or len(table) < 2:
            return ""
        
        lines = []
        headers = table[0]
        
        for row in table[1:]:
            if row:
                for i, cell in enumerate(row):
                    if cell and i < len(headers) and headers[i]:
                        lines.append(f"{headers[i]}: {cell}")
        
        return "\n".join(lines)
    
    def direct_extract(self, field_name):
        """直接从文档中提取信息"""
        mapping = QUESTION_MAPPINGS.get(field_name, {})
        keywords = mapping.get("keywords", [])
        pattern = mapping.get("pattern", "")
        
        # 搜索所有文档
        for doc in self.documents:
            text = doc["text"]
            
            # 1. 关键词匹配
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    # 找到关键词，尝试提取周围信息
                    if pattern:
                        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                        if match:
                            return match.group(0), doc
                    
                    # 提取关键词周围的文本
                    idx = text.lower().find(keyword.lower())
                    if idx >= 0:
                        context = text[max(0, idx-50):min(len(text), idx+200)]
                        # 尝试从上下文提取值
                        if "minimum premium" in field_name.lower():
                            numbers = re.findall(r'(?:USD|US\$|HKD|HK\$)?\s*([\d,]+)', context)
                            if numbers:
                                return f"USD{numbers[0]}" if "USD" in context else f"HKD{numbers[0]}", doc
        
        return None, None
    
    def get_embedding(self, text):
        """获取向量嵌入"""
        try:
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers=self.headers,
                json={
                    "model": "text-embedding-3-small",
                    "input": text[:8000]
                },
                timeout=30
            )
            if response.status_code == 200:
                self.total_cost += 0.00002
                return response.json()['data'][0]['embedding']
        except:
            pass
        return None
    
    def semantic_search(self, query, k=5):
        """语义搜索"""
        query_emb = self.get_embedding(query)
        if not query_emb or not self.embeddings:
            return []
        
        # 计算相似度
        query_vec = np.array(query_emb)
        scores = []
        for emb in self.embeddings:
            if emb is not None:
                score = np.dot(query_vec, emb) / (np.linalg.norm(query_vec) * np.linalg.norm(emb))
                scores.append(score)
            else:
                scores.append(0)
        
        # 获取top-k
        top_indices = np.argsort(scores)[-k:][::-1]
        results = []
        for idx in top_indices:
            if scores[idx] > 0.5:  # 相似度阈值
                results.append({
                    "doc": self.documents[idx],
                    "score": scores[idx]
                })
        
        return results
    
    def answer_question(self, field_name):
        """回答单个问题"""
        # 1. 先尝试直接提取
        answer, source_doc = self.direct_extract(field_name)
        if answer:
            return {
                "answer": answer,
                "confidence": "High",
                "method": "direct",
                "source": source_doc.get("source", "") if source_doc else ""
            }
        
        # 2. 语义搜索
        results = self.semantic_search(field_name.lower(), k=3)
        if not results:
            return {"answer": "[Not Found]", "confidence": "Low", "method": "none"}
        
        # 3. 使用GPT提取
        context = "\n\n".join([r["doc"]["text"][:1000] for r in results[:2]])
        
        prompt = f"""从以下保险文档中提取 {field_name}。
只提供准确的信息，如果找不到就说 [Not Found]。

文档内容:
{context}

问题: What is the {field_name}?
答案 (简洁准确):"""
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=self.headers,
                json={
                    "model": "gpt-4-turbo-preview",
                    "messages": [
                        {"role": "system", "content": "你是保险文档分析专家，只提取准确信息。"},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0,
                    "max_tokens": 100
                },
                timeout=30
            )
            
            if response.status_code == 200:
                self.total_cost += 0.02
                answer = response.json()['choices'][0]['message']['content'].strip()
                return {
                    "answer": answer,
                    "confidence": "Medium" if results[0]["score"] > 0.7 else "Low",
                    "method": "gpt",
                    "source": results[0]["doc"]["source"]
                }
        except:
            pass
        
        return {"answer": "[Not Found]", "confidence": "Low", "method": "failed"}
    
    def process_all(self, pdf_files):
        """处理所有问题"""
        # 1. 提取文档
        print("\n=== 提取文档 ===")
        for pdf_file in pdf_files:
            if Path(pdf_file).exists():
                chunks = self.extract_pdf_smart(pdf_file)
                self.documents.extend(chunks)
        
        print(f"提取了 {len(self.documents)} 个文档块")
        
        # 2. 构建向量（只对主要文档）
        print("\n=== 构建向量 ===")
        for doc in tqdm(self.documents[:50], desc="向量化"):  # 限制数量控制成本
            emb = self.get_embedding(doc["text"])
            self.embeddings.append(emb)
        
        # 3. 回答问题
        print("\n=== 回答问题 ===")
        results = {}
        success = 0
        
        questions = [
            "Insurer Entity Name", "Product Name", "Minimum Premium",
            "Maximum Premium", "Policy Currency", "Policy Term",
            "Issue Age", "Premium Term(s)", "Number of Insured Lives",
            "Product Type", "Issuing Jurisdiction", "Withdrawal Features",
            "Death Settlement Feature(s)", "Additional Benefits",
            "Contract Governing Law", "Product Asset Mix",
            "Backdating Availability?", "Non-Medical Limit",
            "Prepayment Applicable?", "Policy Currency(ies)",
            "Issue Age and Age Methodology", "Change of Life Assured Feature(s)",
            "Day 1 GCV", "Total Surrender Value Components",
            "Total Death Benefit Components", "Product Base",
            "Product Asset Manager", "Product Asset Custodian",
            "Insurer Financial Strength Rating(s)"
        ]
        
        for q in questions[:27]:  # 只处理27个问题
            print(f"  {q}...", end="")
            result = self.answer_question(q)
            results[q] = result
            
            if "[Not Found]" not in result["answer"]:
                success += 1
                print(f" ✓ ({result['method']})")
            else:
                print(" ✗")
        
        # 4. 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output/simple_improved_{timestamp}.json"
        
        Path("output").mkdir(exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "results": results,
                "summary": {
                    "total": len(results),
                    "success": success,
                    "rate": f"{success/len(results)*100:.1f}%",
                    "cost": f"${self.total_cost:.4f}"
                }
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n=== 完成 ===")
        print(f"成功率: {success}/{len(results)} = {success/len(results)*100:.1f}%")
        print(f"成本: ${self.total_cost:.4f}")
        print(f"结果已保存: {output_path}")
        
        return results

def main():
    rag = SimpleImprovedRAG()
    
    pdf_files = [
        r"D:\桌面\RAG\保险产品\AIA FlexiAchieverSavingsPlan_tc-活享儲蓄計劃.pdf",
        r"D:\桌面\RAG\保险产品\RoyalFortune_Product Brochure_EN.pdf"
    ]
    
    rag.process_all(pdf_files)

if __name__ == "__main__":
    main()