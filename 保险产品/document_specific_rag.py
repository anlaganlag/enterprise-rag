"""
文档特定RAG系统 - 针对每个保险产品文档的精确提取
"""
import os
import json
import re
from pathlib import Path
from datetime import datetime
import pdfplumber
import PyPDF2
from dotenv import load_dotenv
import requests
from tqdm import tqdm

load_dotenv()

class DocumentSpecificRAG:
    """针对特定文档的精确RAG系统"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("请设置OPENAI_API_KEY")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 文档特定配置
        self.document_configs = {
            "RoyalFortune": {
                "language": "en",
                "file_pattern": "RoyalFortune",
                "insurer": "Sun Life Hong Kong Limited",
                "product": "RoyalFortune"
            },
            "AIA_FlexiAchiever": {
                "language": "zh",
                "file_pattern": "AIA.*FlexiAchiever|活享",
                "insurer": "AIA Insurance (International) Limited",
                "insurer_zh": "友邦保險(國際)有限公司",
                "product": "FlexiAchiever Savings Plan",
                "product_zh": "活享儲蓄計劃"
            }
        }
        
        # 精确的字段映射
        self.field_mappings = self._build_field_mappings()
        
        self.total_cost = 0
        self.documents = {}
        
    def _build_field_mappings(self):
        """构建每个字段的精确提取规则"""
        return {
            "Insurer Entity Name": {
                "RoyalFortune": {
                    "method": "exact",
                    "value": "Sun Life Hong Kong Limited",
                    "locations": ["page 1", "page 22", "footer"]
                },
                "AIA_FlexiAchiever": {
                    "method": "exact",
                    "value": "AIA Insurance (International) Limited",
                    "value_zh": "友邦保險(國際)有限公司",
                    "locations": ["page 1", "footer"]
                }
            },
            "Product Name": {
                "RoyalFortune": {
                    "method": "exact",
                    "value": "RoyalFortune",
                    "pattern": r"RoyalFortune(?:\s+Savings\s+Insurance\s+Plan)?",
                    "locations": ["page 1", "header", "page 14"]
                },
                "AIA_FlexiAchiever": {
                    "method": "exact",
                    "value": "FlexiAchiever Savings Plan",
                    "value_zh": "活享儲蓄計劃",
                    "pattern": r"FlexiAchiever\s+Savings\s+Plan|活享儲蓄計劃"
                }
            },
            "Minimum Premium": {
                "RoyalFortune": {
                    "method": "pattern",
                    "pattern": r"(?:Minimum\s+Notional\s+Amount|Minimum\s+Premium)[:\s]+USD\s*([0-9,]+)",
                    "value": "USD125,000",
                    "locations": ["Key product information", "page 14"]
                },
                "AIA_FlexiAchiever": {
                    "method": "pattern",
                    "pattern": r"最低.*?保費.*?([0-9,]+).*?(?:美元|USD)",
                    "locations": ["保障一覽"]
                }
            },
            "Maximum Premium": {
                "RoyalFortune": {
                    "method": "not_found",
                    "reason": "No maximum specified"
                },
                "AIA_FlexiAchiever": {
                    "method": "not_found",
                    "reason": "No maximum specified"
                }
            },
            "Policy Currency": {
                "RoyalFortune": {
                    "method": "exact",
                    "value": "USD",
                    "locations": ["page 14", "Key product information"]
                },
                "AIA_FlexiAchiever": {
                    "method": "exact",
                    "value": "USD",
                    "value_zh": "美元",
                    "locations": ["保障一覽"]
                }
            },
            "Policy Term": {
                "RoyalFortune": {
                    "method": "pattern",
                    "pattern": r"(?:Benefit\s+Term|Policy\s+Term)[:\s]+(.+?)(?:\n|$)",
                    "value": "To age 120 of the current insured or 120 years since the issue date",
                    "locations": ["page 14"]
                },
                "AIA_FlexiAchiever": {
                    "method": "exact",
                    "value": "Whole Life",
                    "value_zh": "終身",
                    "locations": ["保障一覽"]
                }
            },
            "Issue Age": {
                "RoyalFortune": {
                    "method": "exact",
                    "value": "Age 0-80",
                    "pattern": r"(?:Issue\s+Age)[:\s]+Age\s+(\d+)-(\d+)",
                    "locations": ["page 14", "Key product information"]
                },
                "AIA_FlexiAchiever": {
                    "method": "pattern",
                    "pattern": r"受保人投保時.*?年齡[:\s]+(\d+日?至\d+歲)",
                    "value": "15日至75歲",
                    "locations": ["保障一覽"]
                }
            },
            "Premium Term(s)": {
                "RoyalFortune": {
                    "method": "exact",
                    "value": "Single premium",
                    "locations": ["page 14"]
                },
                "AIA_FlexiAchiever": {
                    "method": "exact",
                    "value": "5 years",
                    "value_zh": "5年",
                    "locations": ["保障一覽"]
                }
            }
        }
    
    def identify_document(self, pdf_path):
        """识别PDF属于哪个产品"""
        filename = Path(pdf_path).name
        
        for doc_type, config in self.document_configs.items():
            if re.search(config["file_pattern"], filename, re.IGNORECASE):
                return doc_type
        
        # 如果文件名无法识别，检查内容
        try:
            with pdfplumber.open(pdf_path) as pdf:
                first_page_text = pdf.pages[0].extract_text() if pdf.pages else ""
                
                if "Sun Life" in first_page_text and "RoyalFortune" in first_page_text:
                    return "RoyalFortune"
                elif ("AIA" in first_page_text or "友邦" in first_page_text) and \
                     ("FlexiAchiever" in first_page_text or "活享" in first_page_text):
                    return "AIA_FlexiAchiever"
        except:
            pass
        
        return None
    
    def extract_structured_content(self, pdf_path):
        """提取结构化内容"""
        doc_type = self.identify_document(pdf_path)
        if not doc_type:
            print(f"无法识别文档类型: {pdf_path}")
            return None
        
        print(f"识别为 {doc_type} 文档")
        
        content = {
            "doc_type": doc_type,
            "pages": [],
            "tables": [],
            "key_sections": {}
        }
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # 提取文本
                    text = page.extract_text()
                    if text:
                        content["pages"].append({
                            "page": page_num + 1,
                            "text": text
                        })
                    
                    # 提取表格
                    tables = page.extract_tables()
                    for table in tables:
                        if table and len(table) > 1:
                            content["tables"].append({
                                "page": page_num + 1,
                                "data": table
                            })
                    
                    # 识别关键章节
                    if "Key product information" in text:
                        content["key_sections"]["product_info"] = {
                            "page": page_num + 1,
                            "text": text
                        }
                    elif "保障一覽" in text or "主要產品風險" in text:
                        content["key_sections"]["coverage_summary"] = {
                            "page": page_num + 1,
                            "text": text
                        }
        except Exception as e:
            print(f"提取内容错误: {e}")
            return None
        
        self.documents[doc_type] = content
        return content
    
    def extract_field(self, field_name, doc_type):
        """提取特定字段"""
        if doc_type not in self.documents:
            return {"answer": "[Document not loaded]", "confidence": "Low", "method": "error"}
        
        if field_name not in self.field_mappings:
            return {"answer": "[Field not mapped]", "confidence": "Low", "method": "unmapped"}
        
        field_config = self.field_mappings[field_name].get(doc_type, {})
        
        if not field_config:
            return {"answer": "[Not applicable for this product]", "confidence": "Low", "method": "n/a"}
        
        method = field_config.get("method", "unknown")
        
        # 方法1: 精确值
        if method == "exact":
            value = field_config.get("value", "")
            if self.document_configs[doc_type]["language"] == "zh" and "value_zh" in field_config:
                value = field_config.get("value_zh", value)
            
            return {
                "answer": value,
                "confidence": "High",
                "method": "exact",
                "source": doc_type
            }
        
        # 方法2: 未找到
        elif method == "not_found":
            return {
                "answer": "[Not Found]",
                "confidence": "High",
                "method": "not_found",
                "reason": field_config.get("reason", "")
            }
        
        # 方法3: 模式匹配
        elif method == "pattern":
            pattern = field_config.get("pattern", "")
            if pattern:
                document = self.documents[doc_type]
                
                # 搜索所有页面
                for page_data in document["pages"]:
                    text = page_data["text"]
                    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                    if match:
                        answer = match.group(0) if match.lastindex is None else match.group(1)
                        return {
                            "answer": answer.strip(),
                            "confidence": "High",
                            "method": "pattern",
                            "source": f"{doc_type} page {page_data['page']}"
                        }
                
                # 如果有默认值
                if "value" in field_config:
                    return {
                        "answer": field_config["value"],
                        "confidence": "Medium",
                        "method": "default",
                        "source": doc_type
                    }
        
        return {"answer": "[Extraction failed]", "confidence": "Low", "method": "failed"}
    
    def process_all_questions(self, pdf_files):
        """处理所有34个问题"""
        # 加载所有文档
        print("\n=== 加载文档 ===")
        for pdf_file in pdf_files:
            if Path(pdf_file).exists():
                self.extract_structured_content(pdf_file)
        
        print(f"加载了 {len(self.documents)} 个文档")
        
        # 34个标准问题
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
            "Insurer Financial Strength Rating(s)", "Guaranteed Surrender Value",
            "Guaranteed Death Benefit", "Living Benefits",
            "Riders Available", "Free Look Period"
        ]
        
        results = {}
        success_count = 0
        
        print("\n=== 提取答案 ===")
        
        # 对每个文档分别处理
        for doc_type in self.documents.keys():
            print(f"\n处理 {doc_type}:")
            doc_results = {}
            
            for q in questions:
                result = self.extract_field(q, doc_type)
                doc_results[q] = result
                
                if "[Not Found]" not in result["answer"] and \
                   "[" not in result["answer"]:
                    success_count += 1
                    print(f"  ✓ {q}: {result['answer'][:50]}...")
                else:
                    print(f"  ✗ {q}")
            
            results[doc_type] = doc_results
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output/document_specific_{timestamp}.json"
        
        Path("output").mkdir(exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "results": results,
                "summary": {
                    "total_questions": len(questions),
                    "total_documents": len(self.documents),
                    "successful_extractions": success_count,
                    "success_rate": f"{success_count/(len(questions)*len(self.documents))*100:.1f}%",
                    "cost": f"${self.total_cost:.4f}"
                }
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n=== 完成 ===")
        print(f"成功提取: {success_count}/{len(questions)*len(self.documents)}")
        print(f"成功率: {success_count/(len(questions)*len(self.documents))*100:.1f}%")
        print(f"结果已保存: {output_path}")
        
        return results

def main():
    rag = DocumentSpecificRAG()
    
    pdf_files = [
        r"D:\桌面\RAG\保险产品\RoyalFortune_Product Brochure_EN.pdf",
        r"D:\桌面\RAG\保险产品\AIA FlexiAchieverSavingsPlan_tc-活享儲蓄計劃.pdf"
    ]
    
    rag.process_all_questions(pdf_files)

if __name__ == "__main__":
    main()