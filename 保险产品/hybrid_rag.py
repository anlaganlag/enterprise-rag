"""
混合检索RAG系统 - 结合BM25和向量搜索
"""
import os
import re
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
import PyPDF2
import pdfplumber
import requests
from tqdm import tqdm
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
import jieba  # 用于中文分词

# 加载环境变量
load_dotenv()

# 34个标准问题
QUESTIONS = {
    "Product Information": [
        "Insurer Entity Name",
        "Insurer Financial Strength Rating(s)",
        "Issuing Jurisdiction", 
        "Product Name",
        "Product Base",
        "Product Type",
        "Product Asset Manager",
        "Product Asset Custodian",
        "Product Asset Mix"
    ],
    "Plan Details": [
        "Issue Age and Age Methodology",
        "Number of Insured Lives",
        "Change of Life Assured Feature(s)",
        "Minimum Premium / Sum Assured",
        "Maximum Premium / Sum Assured",
        "Policy Term",
        "Premium Term(s)",
        "Prepayment Applicable?",
        "Policy Currency(ies)",
        "Withdrawal Features",
        "Death Settlement Feature(s)"
    ],
    "For Participating Whole of Life": [
        "Day 1 GCV",
        "Total Surrender Value Components",
        "Total Death Benefit Components"
    ],
    "Other Details": [
        "Backdating Availability?",
        "Non-Medical Limit",
        "Additional Benefits",
        "Contract Governing Law"
    ]
}

class HybridRAG:
    """混合检索RAG系统"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("请设置OPENAI_API_KEY环境变量")
        
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        self.documents = []
        self.embeddings = []
        self.bm25 = None
        self.tokenized_docs = []
        self.total_tokens = 0
        self.total_cost = 0.0
        
    def extract_pdf_enhanced(self, pdf_path: str) -> List[Dict]:
        """增强的PDF提取，包括表格处理"""
        print(f"正在提取: {Path(pdf_path).name}")
        chunks = []
        
        # 1. 使用pdfplumber提取表格
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # 提取表格
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        if table:
                            # 转换表格为文本
                            table_text = self._table_to_text(table)
                            if table_text:
                                chunks.append({
                                    "text": table_text,
                                    "source": Path(pdf_path).name,
                                    "page": page_num + 1,
                                    "type": "table",
                                    "table_index": table_idx
                                })
        except Exception as e:
            print(f"pdfplumber提取表格失败: {e}")
        
        # 2. 使用PyPDF2提取文本
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num in tqdm(range(len(pdf_reader.pages)), desc="提取文本"):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    
                    if page_text:
                        # 智能分块 - 保持段落完整性
                        paragraphs = self._smart_split(page_text)
                        
                        for para_idx, paragraph in enumerate(paragraphs):
                            if len(paragraph.strip()) > 30:  # 忽略太短的段落
                                chunks.append({
                                    "text": paragraph.strip(),
                                    "source": Path(pdf_path).name,
                                    "page": page_num + 1,
                                    "type": "text",
                                    "paragraph": para_idx + 1
                                })
                        
        except Exception as e:
            print(f"PyPDF2提取失败: {e}")
        
        # 3. 滑动窗口分块（增加上下文）
        final_chunks = self._sliding_window_chunks(chunks)
        
        return final_chunks
    
    def _table_to_text(self, table_data: List[List]) -> str:
        """将表格转换为结构化文本"""
        if not table_data:
            return ""
        
        lines = []
        
        # 处理表头
        if len(table_data) > 0:
            headers = [str(cell) if cell else "" for cell in table_data[0]]
            lines.append("Table Headers: " + " | ".join(headers))
            
            # 处理数据行
            for row in table_data[1:]:
                if row:
                    row_items = []
                    for i, cell in enumerate(row):
                        if cell and i < len(headers):
                            row_items.append(f"{headers[i]}: {cell}")
                    if row_items:
                        lines.append(" , ".join(row_items))
        
        return "\n".join(lines)
    
    def _smart_split(self, text: str, max_chunk_size: int = 400) -> List[str]:
        """智能分割文本，保持语义完整性"""
        chunks = []
        
        # 按句子分割
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > max_chunk_size and current_chunk:
                # 保存当前块
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # 保存最后一块
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _sliding_window_chunks(self, chunks: List[Dict], overlap: int = 100) -> List[Dict]:
        """滑动窗口分块，增加上下文重叠"""
        enhanced_chunks = []
        
        for i, chunk in enumerate(chunks):
            text = chunk["text"]
            
            # 添加前文上下文
            if i > 0 and chunks[i-1]["page"] == chunk["page"]:
                prev_text = chunks[i-1]["text"]
                if len(prev_text) > overlap:
                    text = "..." + prev_text[-overlap:] + " " + text
            
            # 添加后文上下文
            if i < len(chunks) - 1 and chunks[i+1]["page"] == chunk["page"]:
                next_text = chunks[i+1]["text"]
                if len(next_text) > overlap:
                    text = text + " " + next_text[:overlap] + "..."
            
            enhanced_chunks.append({
                **chunk,
                "text": text,
                "original_index": i
            })
        
        return enhanced_chunks
    
    def tokenize_text(self, text: str) -> List[str]:
        """文本分词（支持中英文）"""
        # 检测是否包含中文
        if re.search(r'[\u4e00-\u9fff]', text):
            # 中文分词
            tokens = list(jieba.cut(text))
        else:
            # 英文分词
            tokens = text.lower().split()
        
        # 清理和过滤
        tokens = [t.strip() for t in tokens if t.strip() and len(t) > 1]
        return tokens
    
    def build_bm25_index(self):
        """构建BM25索引"""
        print("构建BM25索引...")
        
        # 对所有文档进行分词
        self.tokenized_docs = []
        for doc in self.documents:
            tokens = self.tokenize_text(doc["text"])
            self.tokenized_docs.append(tokens)
        
        # 创建BM25索引
        self.bm25 = BM25Okapi(self.tokenized_docs)
        print(f"BM25索引构建完成: {len(self.tokenized_docs)} 个文档")
    
    def get_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本嵌入向量"""
        url = f"{self.base_url}/embeddings"
        
        data = {
            "model": "text-embedding-3-small",
            "input": texts
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            result = response.json()
            
            # 统计token使用
            if 'usage' in result:
                tokens = result['usage']['total_tokens']
                self.total_tokens += tokens
                self.total_cost += tokens * 0.00002 / 1000
            
            return [item['embedding'] for item in result['data']]
        except Exception as e:
            print(f"获取嵌入失败: {e}")
            return None
    
    def build_index(self, pdf_files: List[str]):
        """构建混合索引"""
        print("\n=== 构建混合索引 ===")
        
        # 1. 提取所有PDF文本
        all_chunks = []
        for pdf_file in pdf_files:
            if Path(pdf_file).exists():
                chunks = self.extract_pdf_enhanced(pdf_file)
                all_chunks.extend(chunks)
        
        print(f"提取了 {len(all_chunks)} 个文本块")
        
        # 去重
        seen = set()
        unique_chunks = []
        for chunk in all_chunks:
            text_hash = hash(chunk["text"])
            if text_hash not in seen:
                seen.add(text_hash)
                unique_chunks.append(chunk)
        
        print(f"去重后: {len(unique_chunks)} 个文本块")
        self.documents = unique_chunks
        
        # 2. 构建BM25索引
        self.build_bm25_index()
        
        # 3. 获取向量嵌入
        print("获取嵌入向量...")
        batch_size = 20
        
        for i in tqdm(range(0, len(self.documents), batch_size), desc="向量化"):
            batch = self.documents[i:i + batch_size]
            texts = [doc["text"] for doc in batch]
            
            embeddings = self.get_embedding_batch(texts)
            if embeddings:
                self.embeddings.extend(embeddings)
        
        self.embeddings = np.array(self.embeddings)
        print(f"混合索引构建完成")
    
    def hybrid_search(self, query: str, k: int = 10, alpha: float = 0.4) -> List[Dict]:
        """混合搜索：结合BM25和向量搜索"""
        results = []
        
        # 1. BM25关键词搜索
        query_tokens = self.tokenize_text(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # 2. 向量搜索
        query_embeddings = self.get_embedding_batch([query])
        if query_embeddings:
            query_vec = np.array(query_embeddings[0])
            query_norm = query_vec / np.linalg.norm(query_vec)
            embeddings_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            vector_scores = np.dot(embeddings_norm, query_norm)
        else:
            vector_scores = np.zeros(len(self.documents))
        
        # 3. 归一化分数
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()
        if vector_scores.max() > 0:
            vector_scores = vector_scores / vector_scores.max()
        
        # 4. 混合评分 (alpha控制BM25权重)
        hybrid_scores = alpha * bm25_scores + (1 - alpha) * vector_scores
        
        # 5. 获取Top-K结果
        top_k_indices = np.argsort(hybrid_scores)[-k:][::-1]
        
        for idx in top_k_indices:
            results.append({
                **self.documents[idx],
                "bm25_score": float(bm25_scores[idx]),
                "vector_score": float(vector_scores[idx]),
                "hybrid_score": float(hybrid_scores[idx])
            })
        
        return results
    
    def extract_specific_info(self, text: str, field_name: str) -> str:
        """使用正则表达式提取特定信息"""
        patterns = {
            "Minimum Premium": r"(?:minimum|min).*?(?:premium|amount).*?(?:USD|HKD|US\$|HK\$)?[\s]*([\d,]+)",
            "Maximum Premium": r"(?:maximum|max).*?(?:premium|amount).*?(?:USD|HKD|US\$|HK\$)?[\s]*([\d,]+)",
            "Policy Currency": r"(?:currency|currencies).*?(?:USD|HKD|US\$|HK\$|United States Dollar|Hong Kong Dollar)",
            "Issue Age": r"(?:issue age|age).*?(\d+).*?(?:to|-).*?(\d+)",
            "Policy Term": r"(?:policy term|term).*?(\d+).*?(?:years?|year)",
        }
        
        pattern = patterns.get(field_name)
        if pattern:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    def format_question_variants(self, field_name: str) -> List[str]:
        """为每个字段生成多个查询变体"""
        variants = {
            "Insurer Entity Name": [
                "insurance company name",
                "insurer name",
                "company name",
                "underwriter"
            ],
            "Product Name": [
                "product name",
                "plan name",
                "insurance product",
                "policy name"
            ],
            "Minimum Premium": [
                "minimum premium",
                "min premium",
                "lowest premium",
                "minimum payment"
            ],
            "Policy Currency": [
                "currency",
                "payment currency",
                "premium currency",
                "USD HKD"
            ],
            # 添加更多变体...
        }
        
        return variants.get(field_name, [field_name.lower()])
    
    def answer_question_enhanced(self, field_name: str) -> Dict:
        """增强的问答，使用多种策略"""
        
        # 1. 生成查询变体
        query_variants = self.format_question_variants(field_name)
        
        # 2. 对每个变体进行混合搜索
        all_results = []
        for variant in query_variants[:2]:  # 限制变体数量以控制成本
            results = self.hybrid_search(variant, k=5)
            all_results.extend(results)
        
        # 3. 去重和重排序
        seen = set()
        unique_results = []
        for result in sorted(all_results, key=lambda x: x['hybrid_score'], reverse=True):
            doc_id = (result['source'], result['page'], result.get('original_index', 0))
            if doc_id not in seen:
                seen.add(doc_id)
                unique_results.append(result)
                if len(unique_results) >= 5:
                    break
        
        if not unique_results:
            return {
                "answer": "[Not Found]",
                "confidence": "Low",
                "sources": []
            }
        
        # 4. 尝试直接提取信息
        for doc in unique_results[:3]:
            extracted = self.extract_specific_info(doc['text'], field_name)
            if extracted:
                return {
                    "answer": extracted,
                    "confidence": "High",
                    "sources": [{
                        "source": doc['source'],
                        "page": doc['page'],
                        "score": doc['hybrid_score']
                    }]
                }
        
        # 5. 使用GPT生成答案
        context = "\n\n".join([f"[{doc['source']} P{doc['page']}] {doc['text']}" for doc in unique_results[:3]])
        
        url = f"{self.base_url}/chat/completions"
        
        prompt = f"""Extract the {field_name} from the following insurance document excerpts. 
Be very specific and precise. If the exact information is not found, say "[Not Found]".

Context:
{context}

Question: What is the {field_name}?

Answer (be concise and exact):"""
        
        data = {
            "model": "gpt-4-turbo-preview",
            "messages": [
                {"role": "system", "content": "You are an insurance document analyst. Extract specific information precisely."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0,
            "max_tokens": 100
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            result = response.json()
            
            if 'usage' in result:
                tokens = result['usage']['total_tokens']
                self.total_tokens += tokens
                self.total_cost += tokens * 0.02 / 1000
            
            answer = result['choices'][0]['message']['content'].strip()
            
            # 判断置信度
            best_score = unique_results[0]['hybrid_score']
            confidence = "High" if best_score > 0.7 else "Medium" if best_score > 0.5 else "Low"
            
            return {
                "answer": answer,
                "confidence": confidence,
                "sources": [{
                    "source": doc['source'],
                    "page": doc['page'],
                    "score": doc['hybrid_score']
                } for doc in unique_results[:3]]
            }
            
        except Exception as e:
            return {
                "answer": f"[Error: {str(e)}]",
                "confidence": "Low",
                "sources": []
            }
    
    def answer_all_questions(self) -> Dict:
        """回答所有34个问题"""
        print("\n=== 回答34个问题（混合检索）===")
        
        results = {}
        answered = 0
        not_found = 0
        
        for category, fields in QUESTIONS.items():
            print(f"\n处理类别: {category}")
            
            for field in fields:
                print(f"  问题: {field}...")
                
                result = self.answer_question_enhanced(field)
                results[field] = {
                    "category": category,
                    **result
                }
                
                if "[Not Found]" not in result["answer"] and "[Error" not in result["answer"]:
                    answered += 1
                else:
                    not_found += 1
        
        # 汇总统计
        total_questions = len(results)
        summary = {
            "total_questions": total_questions,
            "answered": answered,
            "not_found": not_found,
            "success_rate": f"{(answered/total_questions*100):.1f}%",
            "total_tokens": self.total_tokens,
            "total_cost": f"${self.total_cost:.4f}"
        }
        
        return {"results": results, "summary": summary}
    
    def save_results(self, results: Dict, output_dir: str = "output"):
        """保存结果"""
        Path(output_dir).mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存JSON
        json_path = Path(output_dir) / f"hybrid_qa_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"JSON结果已保存: {json_path}")
        
        # 保存Markdown
        md_path = Path(output_dir) / f"hybrid_qa_results_{timestamp}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# 保险产品信息提取结果（混合检索）\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            summary = results.get('summary', {})
            f.write("## 执行摘要\n\n")
            f.write(f"- **总问题数**: {summary.get('total_questions', 0)}\n")
            f.write(f"- **成功回答**: {summary.get('answered', 0)}\n")
            f.write(f"- **未找到**: {summary.get('not_found', 0)}\n")
            f.write(f"- **成功率**: {summary.get('success_rate', 'N/A')}\n")
            f.write(f"- **检索方式**: 混合检索（BM25 + 向量）\n\n")
            
            for category in QUESTIONS:
                f.write(f"## {category}\n\n")
                
                for field in QUESTIONS[category]:
                    if field in results['results']:
                        item = results['results'][field]
                        f.write(f"### {field}\n")
                        f.write(f"- **答案**: {item['answer']}\n")
                        f.write(f"- **置信度**: {item['confidence']}\n")
                        if item.get('sources'):
                            f.write(f"- **最佳来源**: {item['sources'][0]['source']} (页 {item['sources'][0]['page']}, 分数 {item['sources'][0]['score']:.3f})\n")
                        f.write("\n")
        
        print(f"Markdown结果已保存: {md_path}")
    
    def save_index(self, path: str = "hybrid_index.pkl"):
        """保存索引"""
        with open(path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embeddings': self.embeddings.tolist() if isinstance(self.embeddings, np.ndarray) else self.embeddings,
                'tokenized_docs': self.tokenized_docs
            }, f)
        print(f"索引已保存: {path}")
    
    def load_index(self, path: str = "hybrid_index.pkl"):
        """加载索引"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.embeddings = np.array(data['embeddings'])
            self.tokenized_docs = data['tokenized_docs']
            
            # 重建BM25索引
            self.bm25 = BM25Okapi(self.tokenized_docs)
        
        print(f"索引已加载: {len(self.documents)} 个文档")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='混合检索RAG系统')
    parser.add_argument('command', nargs='?', default='all',
                      choices=['build', 'run', 'test', 'all'],
                      help='执行命令')
    
    args = parser.parse_args()
    
    print("=== 混合检索RAG系统 ===\n")
    
    # 初始化系统
    rag = HybridRAG()
    
    # PDF文件路径
    pdf_files = [
        r"D:\桌面\RAG\保险产品\AIA FlexiAchieverSavingsPlan_tc-活享儲蓄計劃.pdf",
        r"D:\桌面\RAG\保险产品\RoyalFortune_Product Brochure_EN.pdf"
    ]
    
    index_path = "hybrid_index.pkl"
    
    # 根据命令执行
    if args.command in ['build', 'all']:
        print("步骤1: 构建混合索引")
        rag.build_index(pdf_files)
        rag.save_index(index_path)
    
    if args.command in ['run', 'all', 'test']:
        # 加载索引
        if Path(index_path).exists() and args.command != 'all':
            print("加载现有索引...")
            rag.load_index(index_path)
        elif args.command != 'all':
            print("索引不存在，请先运行: python hybrid_rag.py build")
            return
    
    if args.command == 'test':
        # 测试混合搜索
        test_queries = [
            "minimum premium",
            "product name",
            "currency USD HKD"
        ]
        
        print("\n测试混合搜索:")
        for q in test_queries:
            results = rag.hybrid_search(q, k=3)
            print(f"\n查询: {q}")
            for i, r in enumerate(results, 1):
                print(f"  {i}. [{r['source']}] BM25:{r['bm25_score']:.3f} Vector:{r['vector_score']:.3f} Hybrid:{r['hybrid_score']:.3f}")
                print(f"     {r['text'][:100]}...")
    
    if args.command in ['run', 'all']:
        print("\n步骤2: 回答所有问题")
        results = rag.answer_all_questions()
        
        print("\n步骤3: 保存结果")
        rag.save_results(results)
        
        # 打印摘要
        summary = results['summary']
        print("\n=== 执行完成 ===")
        print(f"成功率: {summary['success_rate']}")
        print(f"总成本: {summary['total_cost']}")

if __name__ == "__main__":
    main()