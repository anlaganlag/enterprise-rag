"""
增强版独立RAG系统 - 完整功能版本
"""
import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import PyPDF2
import requests
from tqdm import tqdm
from dotenv import load_dotenv

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

class EnhancedStandaloneRAG:
    """增强版独立RAG系统"""
    
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
        self.total_tokens = 0
        self.total_cost = 0.0
        
    def extract_pdf_text(self, pdf_path: str) -> List[Dict]:
        """提取PDF文本并保留元数据"""
        print(f"正在提取: {Path(pdf_path).name}")
        chunks = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num in tqdm(range(len(pdf_reader.pages)), desc="提取页面"):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    
                    if page_text:
                        # 按段落分块
                        paragraphs = page_text.split('\n\n')
                        for para_idx, paragraph in enumerate(paragraphs):
                            if len(paragraph.strip()) > 50:  # 忽略太短的段落
                                chunks.append({
                                    "text": paragraph.strip(),
                                    "source": Path(pdf_path).name,
                                    "page": page_num + 1,
                                    "paragraph": para_idx + 1
                                })
                        
        except Exception as e:
            print(f"提取PDF失败: {e}")
            
        # 如果段落分块太大，进一步分割
        final_chunks = []
        max_chunk_size = 500
        
        for chunk in chunks:
            if len(chunk["text"]) > max_chunk_size:
                # 分割大段落
                text = chunk["text"]
                words = text.split()
                for i in range(0, len(words), max_chunk_size // 5):
                    sub_chunk = ' '.join(words[i:i + max_chunk_size // 5])
                    if sub_chunk.strip():
                        final_chunks.append({
                            **chunk,
                            "text": sub_chunk.strip()
                        })
            else:
                final_chunks.append(chunk)
                
        return final_chunks
    
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
                # text-embedding-3-small 价格约 $0.00002/1K tokens
                self.total_cost += tokens * 0.00002 / 1000
            
            return [item['embedding'] for item in result['data']]
        except Exception as e:
            print(f"获取嵌入失败: {e}")
            return None
    
    def build_index(self, pdf_files: List[str]):
        """构建向量索引"""
        print("\n=== 构建向量索引 ===")
        
        # 1. 提取所有PDF文本
        all_chunks = []
        for pdf_file in pdf_files:
            if Path(pdf_file).exists():
                chunks = self.extract_pdf_text(pdf_file)
                all_chunks.extend(chunks)
        
        print(f"提取了 {len(all_chunks)} 个文本块")
        
        # 2. 批量获取嵌入向量
        print("获取嵌入向量...")
        batch_size = 20  # OpenAI API 批量限制
        
        for i in tqdm(range(0, len(all_chunks), batch_size), desc="向量化"):
            batch = all_chunks[i:i + batch_size]
            texts = [chunk["text"] for chunk in batch]
            
            embeddings = self.get_embedding_batch(texts)
            if embeddings:
                for chunk, embedding in zip(batch, embeddings):
                    self.documents.append(chunk)
                    self.embeddings.append(embedding)
        
        # 转换为numpy数组以加速搜索
        self.embeddings = np.array(self.embeddings)
        print(f"索引构建完成: {len(self.documents)} 个文档")
        
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """搜索相似文档"""
        # 获取查询向量
        query_embeddings = self.get_embedding_batch([query])
        if not query_embeddings:
            return []
        
        query_vec = np.array(query_embeddings[0])
        
        # 计算余弦相似度
        # 归一化向量
        query_norm = query_vec / np.linalg.norm(query_vec)
        embeddings_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        # 计算相似度
        similarities = np.dot(embeddings_norm, query_norm)
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            results.append({
                **self.documents[idx],
                "score": float(similarities[idx])
            })
        
        return results
    
    def format_question(self, field_name: str) -> str:
        """将字段名转换为问题"""
        question_map = {
            "Insurer Entity Name": "What is the name of the insurance company or insurer entity?",
            "Product Name": "What is the name of the insurance product?",
            "Minimum Premium / Sum Assured": "What is the minimum premium or sum assured?",
            "Maximum Premium / Sum Assured": "What is the maximum premium or sum assured?",
            "Policy Currency(ies)": "What currencies are available for this policy?",
            "Policy Term": "What is the policy term or duration?",
            # 添加更多映射...
        }
        return question_map.get(field_name, f"What is the {field_name}?")
    
    def answer_question(self, question: str, field_name: str = None) -> Dict:
        """回答问题并返回结构化结果"""
        # 1. 搜索相关文档
        docs = self.search(question, k=5)
        
        if not docs:
            return {
                "answer": "[Not Found]",
                "confidence": "Low",
                "sources": []
            }
        
        # 2. 构建上下文
        context_parts = []
        sources = []
        
        for doc in docs[:3]:  # 只使用前3个最相关的
            context_parts.append(f"[Page {doc['page']}] {doc['text']}")
            sources.append({
                "source": doc['source'],
                "page": doc['page'],
                "score": doc['score']
            })
        
        context = "\n\n".join(context_parts)
        
        # 3. 调用GPT生成答案
        url = f"{self.base_url}/chat/completions"
        
        prompt = f"""You are an insurance product expert. Based on the following document excerpts, answer the question accurately.

Context from insurance documents:
{context}

Question: {question}

Instructions:
1. If the answer is clearly in the documents, provide it precisely
2. For numerical values, quote exactly from the documents
3. If the information is not found, say "[Not Found]"
4. Be concise and specific

Answer:"""
        
        data = {
            "model": "gpt-4-turbo-preview",
            "messages": [
                {"role": "system", "content": "You are an insurance product analyst. Provide accurate, concise answers based solely on the given documents."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0,
            "max_tokens": 200
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            result = response.json()
            
            # 统计token使用
            if 'usage' in result:
                tokens = result['usage']['total_tokens']
                self.total_tokens += tokens
                # GPT-4 Turbo 价格约 $0.01/1K tokens (input) + $0.03/1K tokens (output)
                self.total_cost += tokens * 0.02 / 1000  # 平均价格
            
            answer = result['choices'][0]['message']['content'].strip()
            
            # 判断置信度
            confidence = "High" if docs[0]['score'] > 0.8 else "Medium" if docs[0]['score'] > 0.6 else "Low"
            
            return {
                "answer": answer,
                "confidence": confidence,
                "sources": sources
            }
            
        except Exception as e:
            return {
                "answer": f"[Error: {str(e)}]",
                "confidence": "Low",
                "sources": []
            }
    
    def answer_all_questions(self) -> Dict:
        """回答所有34个问题"""
        print("\n=== 回答34个问题 ===")
        
        results = {}
        answered = 0
        not_found = 0
        
        for category, fields in QUESTIONS.items():
            print(f"\n处理类别: {category}")
            
            for field in fields:
                question = self.format_question(field)
                print(f"  问题: {field}...")
                
                result = self.answer_question(question, field)
                results[field] = {
                    "category": category,
                    **result
                }
                
                if "[Not Found]" not in result["answer"] and "[Error" not in result["answer"]:
                    answered += 1
                else:
                    not_found += 1
        
        # 汇总统计
        summary = {
            "total_questions": len(results),
            "answered": answered,
            "not_found": not_found,
            "success_rate": f"{(answered/len(results)*100):.1f}%",
            "total_tokens": self.total_tokens,
            "total_cost": f"${self.total_cost:.4f}"
        }
        
        return {"results": results, "summary": summary}
    
    def save_results(self, results: Dict, output_dir: str = "output"):
        """保存结果为JSON和Markdown格式"""
        # 创建输出目录
        Path(output_dir).mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存JSON
        json_path = Path(output_dir) / f"insurance_qa_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"JSON结果已保存: {json_path}")
        
        # 保存Markdown
        md_path = Path(output_dir) / f"insurance_qa_results_{timestamp}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# 保险产品信息提取结果\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 摘要
            summary = results.get('summary', {})
            f.write("## 执行摘要\n\n")
            f.write(f"- **总问题数**: {summary.get('total_questions', 0)}\n")
            f.write(f"- **成功回答**: {summary.get('answered', 0)}\n")
            f.write(f"- **未找到**: {summary.get('not_found', 0)}\n")
            f.write(f"- **成功率**: {summary.get('success_rate', 'N/A')}\n")
            f.write(f"- **Token使用**: {summary.get('total_tokens', 0)}\n")
            f.write(f"- **预估成本**: {summary.get('total_cost', 'N/A')}\n\n")
            
            # 详细结果
            for category in QUESTIONS:
                f.write(f"## {category}\n\n")
                
                for field in QUESTIONS[category]:
                    if field in results['results']:
                        item = results['results'][field]
                        f.write(f"### {field}\n")
                        f.write(f"- **答案**: {item['answer']}\n")
                        f.write(f"- **置信度**: {item['confidence']}\n")
                        if item.get('sources'):
                            f.write(f"- **来源**: {item['sources'][0]['source']} (页 {item['sources'][0]['page']})\n")
                        f.write("\n")
        
        print(f"Markdown结果已保存: {md_path}")
    
    def save_index(self, path: str = "enhanced_index.pkl"):
        """保存索引"""
        with open(path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embeddings': self.embeddings.tolist() if isinstance(self.embeddings, np.ndarray) else self.embeddings
            }, f)
        print(f"索引已保存: {path}")
    
    def load_index(self, path: str = "enhanced_index.pkl"):
        """加载索引"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.embeddings = np.array(data['embeddings'])
        print(f"索引已加载: {len(self.documents)} 个文档")

def main():
    """主函数 - 等同于 python main.py all"""
    import argparse
    
    parser = argparse.ArgumentParser(description='增强版独立RAG系统')
    parser.add_argument('command', nargs='?', default='all',
                      choices=['build', 'run', 'test', 'all'],
                      help='执行命令: build(构建索引), run(运行问答), test(测试), all(完整流程)')
    
    args = parser.parse_args()
    
    print("=== 增强版独立RAG系统 ===\n")
    
    # 初始化系统
    rag = EnhancedStandaloneRAG()
    
    # PDF文件路径
    pdf_files = [
        r"D:\桌面\RAG\保险产品\AIA FlexiAchieverSavingsPlan_tc-活享儲蓄計劃.pdf",
        r"D:\桌面\RAG\保险产品\RoyalFortune_Product Brochure_EN.pdf"
    ]
    
    index_path = "enhanced_index.pkl"
    
    # 根据命令执行
    if args.command in ['build', 'all']:
        print("步骤1: 构建向量索引")
        rag.build_index(pdf_files)
        rag.save_index(index_path)
    
    if args.command in ['run', 'all', 'test']:
        # 加载索引
        if Path(index_path).exists() and args.command != 'all':
            print("加载现有索引...")
            rag.load_index(index_path)
        elif args.command != 'all':
            print("索引不存在，请先运行: python standalone_rag_enhanced.py build")
            return
    
    if args.command == 'test':
        # 测试几个问题
        test_questions = [
            "What is the product name?",
            "What is the minimum premium?",
            "What currencies are available?"
        ]
        
        print("\n测试问答:")
        for q in test_questions:
            result = rag.answer_question(q)
            print(f"\n问题: {q}")
            print(f"答案: {result['answer']}")
            print(f"置信度: {result['confidence']}")
    
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