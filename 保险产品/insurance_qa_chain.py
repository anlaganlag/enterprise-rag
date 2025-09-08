"""
保险产品问答链 - 核心RAG逻辑
"""
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.callbacks.manager import get_openai_callback

logger = logging.getLogger(__name__)

class InsuranceQAChain:
    """保险产品问答链"""
    
    def __init__(self, vector_store_manager, llm_model: str = "gpt-4-turbo-preview"):
        """初始化问答链"""
        self.vector_store_manager = vector_store_manager
        # 修复：显式设置OpenAI API密钥
        import os
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=0,
            max_tokens=1000,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.qa_chain = None
        self._setup_qa_chain()
        
    def _setup_qa_chain(self):
        """设置问答链"""
        # 自定义提示模板
        prompt_template = """You are an expert insurance product analyst. Use the following pieces of context to answer the question about insurance products. 
        
If the answer is not in the context, say "Information not found in the provided documents" and explain what specific information is missing.

For numerical values (premiums, ages, percentages), be very precise and quote exactly from the context.

Context:
{context}

Question: {question}

Please provide a structured answer with:
1. Direct answer to the question
2. Source reference (which document and approximate location)
3. Confidence level (High/Medium/Low)

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # 创建检索问答链
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store_manager.get_retriever(
                search_kwargs={"k": 5}
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        logger.info("问答链设置完成")
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """回答单个问题"""
        if self.qa_chain is None:
            raise ValueError("问答链尚未初始化")
            
        try:
            with get_openai_callback() as cb:
                result = self.qa_chain({"query": question})
                
                # 解析结果
                answer = result.get("result", "No answer generated")
                source_docs = result.get("source_documents", [])
                
                # 构建响应
                response = {
                    "question": question,
                    "answer": answer,
                    "sources": [
                        {
                            "content": doc.page_content[:200] + "...",
                            "metadata": doc.metadata
                        } for doc in source_docs[:3]  # 只返回前3个源
                    ],
                    "tokens_used": {
                        "prompt_tokens": cb.prompt_tokens,
                        "completion_tokens": cb.completion_tokens,
                        "total_tokens": cb.total_tokens,
                        "total_cost": cb.total_cost
                    }
                }
                
                logger.info(f"问题已回答，使用 {cb.total_tokens} tokens")
                return response
                
        except Exception as e:
            logger.error(f"回答问题时出错: {e}")
            return {
                "question": question,
                "answer": f"Error: {str(e)}",
                "sources": [],
                "error": True
            }
    
    def answer_all_questions(self, questions_dict: Dict[str, List[str]]) -> Dict[str, Any]:
        """回答所有34个问题"""
        results = {}
        total_tokens = 0
        total_cost = 0
        
        # 扁平化问题列表
        all_questions = []
        for category, questions in questions_dict.items():
            for q in questions:
                all_questions.append({
                    "category": category,
                    "field": q,
                    "question": self._format_question(q)
                })
        
        logger.info(f"开始回答 {len(all_questions)} 个问题")
        
        # 逐个回答问题
        for item in all_questions:
            print(f"\n处理: {item['field']}...")
            response = self.answer_question(item['question'])
            
            # 记录结果
            results[item['field']] = {
                "category": item['category'],
                "answer": self._extract_answer_value(response['answer']),
                "raw_response": response['answer'],
                "sources": response.get('sources', []),
                "confidence": self._extract_confidence(response['answer'])
            }
            
            # 累计token使用
            if 'tokens_used' in response:
                total_tokens += response['tokens_used']['total_tokens']
                total_cost += response['tokens_used']['total_cost']
        
        # 汇总统计
        summary = {
            "total_questions": len(all_questions),
            "answered": sum(1 for r in results.values() if r['answer'] != "[Not Found]"),
            "not_found": sum(1 for r in results.values() if r['answer'] == "[Not Found]"),
            "total_tokens": total_tokens,
            "total_cost": total_cost
        }
        
        return {
            "results": results,
            "summary": summary
        }
    
    def _format_question(self, field_name: str) -> str:
        """将字段名转换为问题"""
        # 映射字段到具体问题
        question_map = {
            "Insurer Entity Name": "What is the name of the insurance company or insurer entity?",
            "Insurer Financial Strength Rating(s)": "What are the financial strength ratings of the insurer?",
            "Issuing Jurisdiction": "What is the issuing jurisdiction for this insurance product?",
            "Product Name": "What is the name of the insurance product?",
            "Product Base": "What is the base type of this product?",
            "Product Type": "What type of insurance product is this (e.g., savings, whole life)?",
            "Product Asset Manager": "Who is the asset manager for this product?",
            "Product Asset Custodian": "Who is the asset custodian for this product?",
            "Product Asset Mix": "What is the asset mix or asset allocation for this product?",
            "Issue Age and Age Methodology": "What are the issue age requirements and age calculation methodology?",
            "Number of Insured Lives": "How many insured lives are allowed on this policy?",
            "Change of Life Assured Feature(s)": "Are there features for changing the life assured?",
            "Minimum Premium / Sum Assured": "What is the minimum premium or sum assured?",
            "Maximum Premium / Sum Assured": "What is the maximum premium or sum assured?",
            "Policy Term": "What is the policy term or duration?",
            "Premium Term(s)": "What are the available premium payment terms?",
            "Prepayment Applicable?": "Is prepayment applicable for this product?",
            "Policy Currency(ies)": "What currencies are available for this policy?",
            "Withdrawal Features": "What are the withdrawal features or partial surrender options?",
            "Death Settlement Feature(s)": "What are the death benefit settlement features?",
            "Day 1 GCV": "What is the Day 1 Guaranteed Cash Value as percentage of single premium?",
            "Total Surrender Value Components": "What are the components of the total surrender value?",
            "Total Death Benefit Components": "What are the components of the total death benefit?",
            "Backdating Availability?": "Is backdating available for this policy?",
            "Non-Medical Limit": "What is the non-medical limit for this product?",
            "Additional Benefits": "What additional benefits are available (e.g., terminal illness, accidental death)?",
            "Contract Governing Law": "What is the governing law for this insurance contract?"
        }
        
        return question_map.get(field_name, f"What is the {field_name}?")
    
    def _extract_answer_value(self, raw_answer: str) -> str:
        """从原始回答中提取具体答案值"""
        # 简单的提取逻辑，可以根据需要改进
        if "not found" in raw_answer.lower() or "no information" in raw_answer.lower():
            return "[Not Found]"
        
        # 尝试提取第一行作为答案
        lines = raw_answer.split('\n')
        for line in lines:
            if line.strip() and not line.startswith('Source') and not line.startswith('Confidence'):
                return line.strip()
        
        return raw_answer.split('.')[0] if raw_answer else "[Not Found]"
    
    def _extract_confidence(self, raw_answer: str) -> str:
        """从原始回答中提取置信度"""
        if "High" in raw_answer:
            return "High"
        elif "Medium" in raw_answer:
            return "Medium"
        elif "Low" in raw_answer:
            return "Low"
        else:
            return "Unknown"
    
    def save_results(self, results: Dict[str, Any], output_path: Path):
        """保存结果到文件"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存JSON格式
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存Markdown格式
        md_path = output_path.with_suffix('.md')
        self._save_as_markdown(results, md_path)
        
        logger.info(f"结果已保存到: {json_path} 和 {md_path}")
    
    def _save_as_markdown(self, results: Dict[str, Any], md_path: Path):
        """保存结果为Markdown格式"""
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# 保险产品信息提取结果\n\n")
            
            # 写入摘要
            summary = results.get('summary', {})
            f.write("## 摘要\n")
            f.write(f"- 总问题数: {summary.get('total_questions', 0)}\n")
            f.write(f"- 已回答: {summary.get('answered', 0)}\n")
            f.write(f"- 未找到: {summary.get('not_found', 0)}\n")
            f.write(f"- Token使用: {summary.get('total_tokens', 0)}\n")
            f.write(f"- 总成本: ${summary.get('total_cost', 0):.4f}\n\n")
            
            # 按类别组织结果
            categories = {}
            for field, data in results.get('results', {}).items():
                category = data.get('category', 'Other')
                if category not in categories:
                    categories[category] = []
                categories[category].append((field, data))
            
            # 写入各类别结果
            for category, items in categories.items():
                f.write(f"## {category}\n\n")
                for field, data in items:
                    answer = data.get('answer', '[Not Found]')
                    confidence = data.get('confidence', 'Unknown')
                    f.write(f"### {field}\n")
                    f.write(f"- **答案**: {answer}\n")
                    f.write(f"- **置信度**: {confidence}\n\n")


if __name__ == "__main__":
    # 测试代码
    from config import VECTOR_STORE_DIR, LLM_MODEL, OUTPUT_DIR
    from vector_store import VectorStoreManager
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 加载向量存储
    vector_manager = VectorStoreManager()
    vector_manager.load_vector_store(VECTOR_STORE_DIR / "insurance_vectors")
    
    # 初始化问答链
    qa_chain = InsuranceQAChain(vector_manager, llm_model=LLM_MODEL)
    
    # 测试单个问题
    test_question = "What is the minimum premium?"
    result = qa_chain.answer_question(test_question)
    
    print(f"问题: {test_question}")
    print(f"答案: {result['answer']}")
    print(f"Token使用: {result.get('tokens_used', {})}")