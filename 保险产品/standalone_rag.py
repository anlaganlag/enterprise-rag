"""
独立的RAG系统 - 完全不依赖LangChain
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

class StandaloneRAG:
    """完全独立的RAG系统"""
    
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
        
    def extract_pdf_text(self, pdf_path: str) -> List[str]:
        """提取PDF文本"""
        print(f"正在提取: {pdf_path}")
        chunks = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                # 简单分块
                chunk_size = 500
                words = text.split()
                
                for i in range(0, len(words), chunk_size // 5):  # 假设平均每个词5个字符
                    chunk = ' '.join(words[i:i + chunk_size // 5])
                    if chunk.strip():
                        chunks.append(chunk)
                        
        except Exception as e:
            print(f"提取PDF失败: {e}")
            
        return chunks
    
    def get_embedding(self, text: str) -> List[float]:
        """获取文本嵌入向量"""
        url = f"{self.base_url}/embeddings"
        
        data = {
            "model": "text-embedding-3-small",
            "input": text
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result['data'][0]['embedding']
        except Exception as e:
            print(f"获取嵌入失败: {e}")
            return None
    
    def build_index(self, pdf_files: List[str]):
        """构建向量索引"""
        print("=== 构建向量索引 ===")
        
        # 1. 提取所有PDF文本
        all_chunks = []
        for pdf_file in pdf_files:
            if Path(pdf_file).exists():
                chunks = self.extract_pdf_text(pdf_file)
                for chunk in chunks:
                    all_chunks.append({
                        "text": chunk,
                        "source": Path(pdf_file).name
                    })
        
        print(f"提取了 {len(all_chunks)} 个文本块")
        
        # 2. 获取嵌入向量
        print("获取嵌入向量...")
        for chunk in tqdm(all_chunks[:100]):  # 限制为前100个以节省成本
            embedding = self.get_embedding(chunk["text"])
            if embedding:
                self.documents.append(chunk)
                self.embeddings.append(embedding)
        
        # 转换为numpy数组
        self.embeddings = np.array(self.embeddings)
        print(f"索引构建完成: {len(self.documents)} 个文档")
        
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """搜索相似文档"""
        # 获取查询向量
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []
        
        query_vec = np.array(query_embedding)
        
        # 计算余弦相似度
        similarities = np.dot(self.embeddings, query_vec)
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            results.append({
                "text": self.documents[idx]["text"],
                "source": self.documents[idx]["source"],
                "score": float(similarities[idx])
            })
        
        return results
    
    def answer_question(self, question: str) -> str:
        """回答问题"""
        # 1. 搜索相关文档
        docs = self.search(question, k=3)
        
        if not docs:
            return "未找到相关信息"
        
        # 2. 构建上下文
        context = "\n\n".join([f"来源: {doc['source']}\n内容: {doc['text']}" for doc in docs])
        
        # 3. 调用GPT生成答案
        url = f"{self.base_url}/chat/completions"
        
        prompt = f"""基于以下保险产品文档内容回答问题。如果信息不在文档中，请说"信息未找到"。

文档内容:
{context}

问题: {question}

请提供准确的答案:"""
        
        data = {
            "model": "gpt-4-turbo-preview",
            "messages": [
                {"role": "system", "content": "你是保险产品专家，基于提供的文档准确回答问题。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0,
            "max_tokens": 500
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            return f"生成答案失败: {e}"
    
    def save_index(self, path: str):
        """保存索引"""
        with open(path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embeddings': self.embeddings
            }, f)
        print(f"索引已保存到: {path}")
    
    def load_index(self, path: str):
        """加载索引"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.embeddings = data['embeddings']
        print(f"索引已加载: {len(self.documents)} 个文档")

def main():
    """主函数"""
    print("=== 独立RAG系统 ===\n")
    
    # 初始化系统
    rag = StandaloneRAG()
    
    # PDF文件
    pdf_files = [
        r"D:\桌面\RAG\保险产品\AIA FlexiAchieverSavingsPlan_tc-活享儲蓄計劃.pdf",
        r"D:\桌面\RAG\保险产品\RoyalFortune_Product Brochure_EN.pdf"
    ]
    
    # 检查是否已有索引
    index_path = "standalone_index.pkl"
    
    if Path(index_path).exists():
        print("加载现有索引...")
        rag.load_index(index_path)
    else:
        print("构建新索引...")
        rag.build_index(pdf_files)
        rag.save_index(index_path)
    
    # 测试问题
    questions = [
        "What is the product name?",
        "What is the minimum premium?",
        "What currencies are available?",
        "What is the policy term?"
    ]
    
    print("\n=== 测试问答 ===")
    for q in questions:
        print(f"\n问题: {q}")
        answer = rag.answer_question(q)
        print(f"答案: {answer[:200]}...")
    
    # 回答所有34个问题
    print("\n是否回答所有34个问题？(y/n): ", end="")
    if input().lower() == 'y':
        from main import QUESTIONS
        
        results = {}
        for category, qs in QUESTIONS.items():
            print(f"\n处理类别: {category}")
            for field in qs:
                question = f"What is the {field}?"
                answer = rag.answer_question(question)
                results[field] = answer
                print(f"  - {field}: {answer[:50]}...")
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"standalone_results_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {output_file}")

if __name__ == "__main__":
    main()