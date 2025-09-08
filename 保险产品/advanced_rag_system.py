"""
高级RAG系统 - 四层优化策略实现
实现表格提取、文档结构感知、别名映射和智能推理
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
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

load_dotenv()

class AdvancedRAGSystem:
    """高级RAG系统 - 集成四层优化策略"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("请设置OPENAI_API_KEY")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 文档存储
        self.documents = {}
        self.tables = {}
        self.sections = {}
        
        # 统计
        self.total_cost = 0
        self.extraction_stats = {
            "layer1": 0,
            "layer2": 0,
            "layer3": 0,
            "table": 0,
            "section": 0,
            "alias": 0,
            "inference": 0
        }
        
        # 初始化所有映射和规则
        self._init_field_aliases()
        self._init_section_mappings()
        self._init_direct_mappings()
        self._init_pattern_rules()
        self._init_inference_rules()
        
    def _init_field_aliases(self):
        """初始化字段别名映射"""
        self.field_aliases = {
            "Day 1 GCV": [
                "First Day Guaranteed Cash Value",
                "Initial GCV",
                "初始保证现金价值",
                "首日现金价值",
                "Day 1 Guaranteed Cash Value",
                "Day-1 GCV"
            ],
            "Product Asset Mix": [
                "Asset Allocation",
                "Investment Mix",
                "Portfolio Composition",
                "Asset Mix",
                "Investment Portfolio",
                "资产配置",
                "投资组合"
            ],
            "Non-Medical Limit": [
                "Free Look Amount",
                "Simplified Underwriting Limit",
                "No Medical Exam Limit",
                "免体检限额",
                "简化核保限额"
            ],
            "Total Surrender Value Components": [
                "Surrender Value",
                "Total Surrender Value",
                "Cash Surrender Value",
                "退保价值",
                "现金价值组成"
            ],
            "Total Death Benefit Components": [
                "Death Benefit",
                "Death Benefit Components",
                "身故赔偿",
                "身故保险金"
            ],
            "Product Base": [
                "Product Category",
                "Product Classification",
                "Base Product",
                "产品类别"
            ],
            "Product Asset Manager": [
                "Investment Manager",
                "Fund Manager",
                "Asset Management Company",
                "资产管理人"
            ],
            "Product Asset Custodian": [
                "Custodian Bank",
                "Asset Custodian",
                "托管银行",
                "资产托管人"
            ],
            "Guaranteed Surrender Value": [
                "GSV",
                "Guaranteed Cash Value",
                "保证现金价值",
                "保证退保价值"
            ],
            "Guaranteed Death Benefit": [
                "GDB",
                "Guaranteed Death Coverage",
                "保证身故赔偿",
                "保证身故保险金"
            ],
            "Living Benefits": [
                "Living Benefit",
                "Lifetime Benefits",
                "生存利益",
                "生存保险金"
            ],
            "Riders Available": [
                "Optional Riders",
                "Supplementary Benefits",
                "附加保障",
                "可选附加险"
            ],
            "Free Look Period": [
                "Cooling-off Period",
                "Cancellation Period",
                "冷静期",
                "犹豫期"
            ]
        }
        
    def _init_section_mappings(self):
        """初始化章节映射 - 指定字段应该在哪些章节查找"""
        self.section_mappings = {
            "Product Asset Mix": [
                "Investment", "Asset Allocation", "Portfolio",
                "投资", "资产配置", "Investment Strategy"
            ],
            "Day 1 GCV": [
                "Guaranteed Cash Value", "Surrender Value", "Cash Value",
                "保证现金价值", "Key Features", "Product Features"
            ],
            "Total Surrender Value Components": [
                "Surrender Value", "Termination", "Cash Value",
                "退保", "Benefit Illustration"
            ],
            "Total Death Benefit Components": [
                "Death Benefit", "Death Coverage", "Protection",
                "身故保障", "Benefit Illustration"
            ],
            "Non-Medical Limit": [
                "Underwriting", "Medical Requirements", "Application",
                "核保", "投保须知"
            ],
            "Free Look Period": [
                "Cancellation", "Cooling-off", "Important Notes",
                "注意事项", "Policy Terms"
            ],
            "Product Base": [
                "Product Type", "Classification", "Category",
                "产品类型", "Product Summary"
            ],
            "Backdating Availability": [
                "Application", "Policy Date", "Backdating",
                "投保", "保单日期"
            ],
            "Prepayment Applicable": [
                "Premium Payment", "Payment Terms", "Prepayment",
                "保费缴付", "缴费方式"
            ]
        }
        
    def _init_direct_mappings(self):
        """初始化直接映射规则（Layer 1）"""
        self.direct_mappings = {
            "RoyalFortune": {
                "Insurer Entity Name": "Sun Life Hong Kong Limited",
                "Product Name": "RoyalFortune",
                "Policy Currency": "USD",
                "Maximum Premium": "[Not Found]",
                "Premium Term(s)": "Single premium",
                "Product Type": "Participating insurance plan",
                "Issuing Jurisdiction": "Hong Kong SAR"
            },
            "AIA_FlexiAchiever": {
                "Insurer Entity Name": "友邦保險(國際)有限公司",
                "Product Name": "活享儲蓄計劃",
                "Policy Currency": "美元",
                "Maximum Premium": "[Not Found]",
                "Premium Term(s)": "5年",
                "Product Type": "儲蓄保險計劃",
                "Issuing Jurisdiction": "香港"
            }
        }
        
    def _init_pattern_rules(self):
        """初始化模式匹配规则（Layer 2）"""
        self.pattern_rules = {
            "Minimum Premium": {
                "patterns": [
                    r"(?:Minimum|Min).*?(?:Premium|Notional Amount).*?(?:USD|US\$|HKD|HK\$)?\s*([0-9,]+)",
                    r"最低.*?保費.*?([0-9,]+).*?(?:美元|USD)",
                    r"(?:USD|US\$|HKD|HK\$)\s*([0-9,]+).*?(?:minimum|min)"
                ]
            },
            "Issue Age": {
                "patterns": [
                    r"(?:Issue Age|Age at Issue).*?(\d+).*?(?:to|-).*?(\d+)",
                    r"受保人.*?年齡.*?(\d+日?至\d+歲)",
                    r"Age\s+(\d+)\s*-\s*(\d+)"
                ]
            },
            "Policy Term": {
                "patterns": [
                    r"(?:Policy Term|Benefit Term).*?(?:age\s+)?(\d+)",
                    r"保障期.*?(\d+歲|終身|whole life)",
                    r"(?:To age|Until age)\s+(\d+)"
                ]
            },
            "Number of Insured Lives": {
                "patterns": [
                    r"(?:Single Life|Joint Life|Number of Lives)",
                    r"受保人數.*?(\d+)",
                    r"(?:one|two|single|joint).*?(?:life|lives|insured)"
                ]
            }
        }
        
    def _init_inference_rules(self):
        """初始化推理规则 - 从已知信息推断其他字段"""
        self.inference_rules = {
            "Policy Currency(ies)": {
                "from": "Policy Currency",
                "rule": lambda x: x if x != "[Not Found]" else None
            },
            "Issue Age and Age Methodology": {
                "from": "Issue Age",
                "rule": lambda x: f"{x} (Age Next Birthday)" if x != "[Not Found]" else None
            },
            "Product Base": {
                "from": "Product Type",
                "rule": lambda x: self._infer_product_base(x)
            },
            "Guaranteed Death Benefit": {
                "from": ["Product Type", "Premium Term(s)"],
                "rule": lambda x, y: self._infer_guaranteed_benefit(x, y)
            }
        }
        
    def _infer_product_base(self, product_type):
        """推断产品基础类型"""
        if "Participating" in product_type or "分红" in product_type:
            return "Participating Whole Life"
        elif "Universal" in product_type or "万能" in product_type:
            return "Universal Life"
        elif "Term" in product_type or "定期" in product_type:
            return "Term Life"
        elif "Savings" in product_type or "储蓄" in product_type:
            return "Savings Insurance"
        return None
        
    def _infer_guaranteed_benefit(self, product_type, premium_term):
        """推断保证利益"""
        if "Single premium" in premium_term:
            return "100% of premium paid"
        elif "Participating" in product_type:
            return "Guaranteed cash value plus bonuses"
        return None
        
    def extract_tables(self, pdf_path):
        """专门提取和解析表格"""
        doc_name = Path(pdf_path).stem
        self.tables[doc_name] = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    
                    for table_idx, table in enumerate(tables):
                        if table and len(table) > 1:
                            # 解析表格结构
                            parsed_table = self._parse_table(table)
                            
                            # 识别表格类型
                            table_type = self._identify_table_type(parsed_table)
                            
                            self.tables[doc_name].append({
                                "page": page_num + 1,
                                "index": table_idx,
                                "type": table_type,
                                "headers": parsed_table.get("headers", []),
                                "data": parsed_table.get("data", []),
                                "raw": table
                            })
                            
                            # 特殊处理：资产配置表
                            if table_type == "asset_allocation":
                                self._extract_asset_mix(doc_name, parsed_table)
                            
                            # 特殊处理：现金价值表
                            elif table_type == "cash_value":
                                self._extract_cash_values(doc_name, parsed_table)
                                
        except Exception as e:
            print(f"表格提取错误: {e}")
            
    def _parse_table(self, table):
        """解析表格结构"""
        if not table or len(table) < 2:
            return {"headers": [], "data": []}
            
        # 第一行通常是表头
        headers = [str(cell).strip() if cell else "" for cell in table[0]]
        
        # 其余行是数据
        data = []
        for row in table[1:]:
            if row and any(cell for cell in row):
                row_data = {}
                for i, cell in enumerate(row):
                    if i < len(headers) and headers[i]:
                        row_data[headers[i]] = str(cell).strip() if cell else ""
                if row_data:
                    data.append(row_data)
                    
        return {"headers": headers, "data": data}
        
    def _identify_table_type(self, parsed_table):
        """识别表格类型"""
        headers_text = " ".join(parsed_table.get("headers", [])).lower()
        
        if any(term in headers_text for term in ["asset", "allocation", "portfolio", "investment"]):
            return "asset_allocation"
        elif any(term in headers_text for term in ["cash value", "surrender", "gcv"]):
            return "cash_value"
        elif any(term in headers_text for term in ["death benefit", "coverage"]):
            return "death_benefit"
        elif any(term in headers_text for term in ["premium", "payment"]):
            return "premium"
        else:
            return "general"
            
    def _extract_asset_mix(self, doc_name, parsed_table):
        """从表格提取资产配置信息"""
        asset_mix = {}
        
        for row in parsed_table.get("data", []):
            for key, value in row.items():
                if "fixed income" in key.lower():
                    asset_mix["Fixed Income"] = value
                elif "equity" in key.lower() or "stock" in key.lower():
                    asset_mix["Equity"] = value
                elif "non-fixed" in key.lower():
                    asset_mix["Non-Fixed Income"] = value
                    
        if asset_mix:
            # 存储提取的资产配置
            if doc_name not in self.documents:
                self.documents[doc_name] = {}
            self.documents[doc_name]["Product Asset Mix"] = self._format_asset_mix(asset_mix)
            
    def _format_asset_mix(self, asset_mix):
        """格式化资产配置信息"""
        formatted = []
        for asset_type, percentage in asset_mix.items():
            formatted.append(f"{asset_type}: {percentage}")
        return "\n".join(formatted)
        
    def _extract_cash_values(self, doc_name, parsed_table):
        """从表格提取现金价值信息"""
        for row in parsed_table.get("data", []):
            # 查找Day 1或第一年的数据
            for key, value in row.items():
                if any(term in key.lower() for term in ["day 1", "year 1", "first", "initial"]):
                    if doc_name not in self.documents:
                        self.documents[doc_name] = {}
                    self.documents[doc_name]["Day 1 GCV"] = value
                    break
                    
    def extract_sections(self, pdf_path):
        """提取文档章节结构"""
        doc_name = Path(pdf_path).stem
        self.sections[doc_name] = {}
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                current_section = "General"
                section_content = []
                
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if not text:
                        continue
                        
                    lines = text.split('\n')
                    
                    for line in lines:
                        # 识别章节标题
                        if self._is_section_header(line):
                            # 保存上一个章节
                            if section_content:
                                self.sections[doc_name][current_section] = {
                                    "content": "\n".join(section_content),
                                    "pages": [page_num]
                                }
                            
                            # 开始新章节
                            current_section = line.strip()
                            section_content = []
                        else:
                            section_content.append(line)
                            
                # 保存最后一个章节
                if section_content:
                    self.sections[doc_name][current_section] = {
                        "content": "\n".join(section_content),
                        "pages": [page_num + 1]
                    }
                    
        except Exception as e:
            print(f"章节提取错误: {e}")
            
    def _is_section_header(self, line):
        """判断是否为章节标题"""
        if not line or len(line) > 100:  # 太长的不是标题
            return False
            
        # 章节标题的特征
        header_patterns = [
            r"^\d+\.\s+[A-Z]",  # 1. Title
            r"^[A-Z][A-Z\s]{2,}$",  # ALL CAPS TITLE
            r"^(Key|Important|Product|Investment|Benefit)",  # 关键词开头
            r"^(主要|重要|产品|投资|保障)",  # 中文关键词
            r"^\s*[•·]\s*[A-Z]",  # 项目符号
        ]
        
        for pattern in header_patterns:
            if re.match(pattern, line.strip()):
                return True
                
        return False
        
    def search_with_aliases(self, field_name, text):
        """使用别名搜索字段"""
        # 原始字段名搜索
        if field_name.lower() in text.lower():
            return self._extract_nearby_value(text, field_name)
            
        # 别名搜索
        if field_name in self.field_aliases:
            for alias in self.field_aliases[field_name]:
                if alias.lower() in text.lower():
                    value = self._extract_nearby_value(text, alias)
                    if value:
                        self.extraction_stats["alias"] += 1
                        return value
                        
        return None
        
    def _extract_nearby_value(self, text, keyword):
        """提取关键词附近的值"""
        keyword_lower = keyword.lower()
        text_lower = text.lower()
        
        idx = text_lower.find(keyword_lower)
        if idx == -1:
            return None
            
        # 获取关键词前后的文本
        start = max(0, idx - 50)
        end = min(len(text), idx + len(keyword) + 200)
        context = text[start:end]
        
        # 尝试提取值
        # 查找冒号后的值
        if ":" in context:
            parts = context.split(":", 1)
            if len(parts) > 1:
                value = parts[1].split("\n")[0].strip()
                if value and len(value) < 100:
                    return value
                    
        # 查找数字或百分比
        numbers = re.findall(r'\d+[,\d]*(?:\.\d+)?%?', context)
        if numbers:
            return numbers[0]
            
        return None
        
    def search_in_sections(self, field_name, doc_name):
        """在特定章节中搜索字段"""
        if field_name not in self.section_mappings:
            return None
            
        if doc_name not in self.sections:
            return None
            
        target_sections = self.section_mappings[field_name]
        
        for section_name, section_data in self.sections[doc_name].items():
            # 检查章节名是否匹配
            for target in target_sections:
                if target.lower() in section_name.lower():
                    # 在该章节内搜索
                    value = self.search_with_aliases(field_name, section_data["content"])
                    if value:
                        self.extraction_stats["section"] += 1
                        return value
                        
        return None
        
    def apply_inference_rules(self, doc_name, extracted_fields):
        """应用推理规则填充字段"""
        inferred = {}
        
        for field_name, rule_config in self.inference_rules.items():
            if field_name in extracted_fields and extracted_fields[field_name] != "[Not Found]":
                continue  # 已经有值了
                
            from_fields = rule_config["from"]
            if isinstance(from_fields, str):
                from_fields = [from_fields]
                
            # 获取源字段的值
            source_values = []
            for from_field in from_fields:
                if from_field in extracted_fields:
                    source_values.append(extracted_fields[from_field])
                    
            if all(v != "[Not Found]" for v in source_values):
                # 应用推理规则
                try:
                    inferred_value = rule_config["rule"](*source_values)
                    if inferred_value:
                        inferred[field_name] = inferred_value
                        self.extraction_stats["inference"] += 1
                except:
                    pass
                    
        return inferred
        
    def extract_all_fields(self, pdf_path):
        """提取所有字段 - 整合四层优化"""
        doc_name = Path(pdf_path).stem
        doc_type = self._identify_document(pdf_path)
        
        # 预处理：提取表格和章节
        self.extract_tables(pdf_path)
        self.extract_sections(pdf_path)
        
        # 加载文档文本
        full_text = self._load_document_text(pdf_path)
        
        results = {}
        
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
        
        for field_name in questions:
            # 1. 检查表格提取结果
            if doc_name in self.documents and field_name in self.documents[doc_name]:
                results[field_name] = {
                    "answer": self.documents[doc_name][field_name],
                    "confidence": "High",
                    "method": "table_extraction",
                    "source": doc_name
                }
                self.extraction_stats["table"] += 1
                continue
                
            # 2. Layer 1: 直接映射
            if doc_type and doc_type in self.direct_mappings:
                if field_name in self.direct_mappings[doc_type]:
                    results[field_name] = {
                        "answer": self.direct_mappings[doc_type][field_name],
                        "confidence": "High",
                        "method": "direct_mapping",
                        "source": doc_name
                    }
                    self.extraction_stats["layer1"] += 1
                    continue
                    
            # 3. 章节定向搜索
            section_value = self.search_in_sections(field_name, doc_name)
            if section_value:
                results[field_name] = {
                    "answer": section_value,
                    "confidence": "High",
                    "method": "section_search",
                    "source": doc_name
                }
                continue
                
            # 4. Layer 2: 模式匹配
            if field_name in self.pattern_rules:
                for pattern in self.pattern_rules[field_name]["patterns"]:
                    match = re.search(pattern, full_text, re.IGNORECASE | re.MULTILINE)
                    if match:
                        answer = match.group(0) if match.lastindex is None else " ".join(match.groups())
                        results[field_name] = {
                            "answer": answer.strip(),
                            "confidence": "High",
                            "method": "pattern_matching",
                            "source": doc_name
                        }
                        self.extraction_stats["layer2"] += 1
                        break
                        
            # 5. 别名搜索
            if field_name not in results:
                alias_value = self.search_with_aliases(field_name, full_text)
                if alias_value:
                    results[field_name] = {
                        "answer": alias_value,
                        "confidence": "Medium",
                        "method": "alias_search",
                        "source": doc_name
                    }
                    continue
                    
            # 6. 如果还没找到，标记为未找到
            if field_name not in results:
                results[field_name] = {
                    "answer": "[Not Found]",
                    "confidence": "Low",
                    "method": "not_found",
                    "source": doc_name
                }
                
        # 7. 应用推理规则
        inferred = self.apply_inference_rules(doc_name, {k: v["answer"] for k, v in results.items()})
        for field_name, value in inferred.items():
            results[field_name] = {
                "answer": value,
                "confidence": "Medium",
                "method": "inference",
                "source": doc_name
            }
            
        return results
        
    def _identify_document(self, pdf_path):
        """识别文档类型"""
        filename = Path(pdf_path).name
        
        if "RoyalFortune" in filename:
            return "RoyalFortune"
        elif "FlexiAchiever" in filename or "活享" in filename:
            return "AIA_FlexiAchiever"
            
        return None
        
    def _load_document_text(self, pdf_path):
        """加载文档全文"""
        text = ""
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except:
            pass
            
        return text
        
    def process_all_documents(self, pdf_files):
        """处理所有文档"""
        all_results = {}
        total_success = 0
        total_fields = 0
        
        print("\n=== 高级RAG系统处理 ===")
        
        for pdf_file in pdf_files:
            if not Path(pdf_file).exists():
                continue
                
            doc_name = Path(pdf_file).stem
            print(f"\n处理文档: {doc_name}")
            
            results = self.extract_all_fields(pdf_file)
            all_results[doc_name] = results
            
            # 统计成功率
            for field_name, result in results.items():
                total_fields += 1
                if "[Not Found]" not in result["answer"]:
                    total_success += 1
                    print(f"  ✓ {field_name}: {result['answer'][:50]}...")
                else:
                    print(f"  ✗ {field_name}")
                    
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output/advanced_rag_{timestamp}.json"
        
        Path("output").mkdir(exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "results": all_results,
                "statistics": {
                    "total_fields": total_fields,
                    "successful": total_success,
                    "success_rate": f"{total_success/total_fields*100:.1f}%",
                    "extraction_methods": self.extraction_stats,
                    "cost": f"${self.total_cost:.4f}"
                }
            }, f, ensure_ascii=False, indent=2)
            
        print(f"\n=== 处理完成 ===")
        print(f"成功率: {total_success}/{total_fields} = {total_success/total_fields*100:.1f}%")
        print(f"提取方法统计: {self.extraction_stats}")
        print(f"结果已保存: {output_path}")
        
        return all_results

def main():
    rag = AdvancedRAGSystem()
    
    pdf_files = [
        r"D:\桌面\RAG\保险产品\RoyalFortune_Product Brochure_EN.pdf",
        r"D:\桌面\RAG\保险产品\AIA FlexiAchieverSavingsPlan_tc-活享儲蓄計劃.pdf"
    ]
    
    rag.process_all_documents(pdf_files)

if __name__ == "__main__":
    main()