"""
中文查询优化器
提供查询预处理、关键词映射、同义词扩展等功能
"""
import re
from typing import Dict, List, Tuple

class ChineseQueryOptimizer:
    def __init__(self):
        # 保险术语中英对照表
        self.term_mapping = {
            # 产品相关
            "最低保费": ["minimum premium", "min premium", "最少保费"],
            "最高保费": ["maximum premium", "max premium", "最大保费"],
            "保费": ["premium", "保险费", "费用"],
            "保单": ["policy", "保险单"],
            "产品": ["product", "保险产品", "险种"],
            
            # 保障相关
            "保证现金价值": ["guaranteed cash value", "GCV", "保证价值"],
            "现金价值": ["cash value", "CV", "现价"],
            "身故赔付": ["death benefit", "death settlement", "死亡赔付"],
            "身故保障": ["death coverage", "死亡保障"],
            "退保": ["surrender", "退保价值", "退保金"],
            
            # 年龄相关
            "投保年龄": ["issue age", "投保年纪", "承保年龄"],
            "年龄": ["age", "年纪"],
            "被保险人": ["insured", "被保人", "受保人"],
            
            # 期限相关
            "保险期限": ["policy term", "保障期限", "保单期限"],
            "缴费期": ["premium term", "付款期", "缴费期限"],
            "保障期": ["coverage period", "保障期间"],
            
            # 货币相关
            "美元": ["USD", "dollar", "美金"],
            "港币": ["HKD", "港元"],
            "人民币": ["RMB", "CNY", "元"],
            
            # 公司相关
            "永明": ["Sun Life", "永明人寿", "永明金融"],
            "友邦": ["AIA", "友邦保险"],
            "保险公司": ["insurer", "insurance company", "承保公司"],
            
            # 功能特性
            "提取": ["withdrawal", "部分提取", "提款"],
            "贷款": ["loan", "保单贷款"],
            "红利": ["bonus", "分红", "dividend"],
            "收益": ["return", "回报", "收益率"],
            "保本": ["breakeven", "回本", "保证回本"],
            
            # 其他
            "优势": ["advantage", "benefit", "优点"],
            "特点": ["feature", "特性", "特色"],
            "适合": ["suitable", "适用", "合适"],
            "购买": ["purchase", "buy", "投保"],
            "申请": ["apply", "application", "申请流程"]
        }
        
        # 问题模板映射
        self.question_patterns = {
            r".*是什么.*": "definition",
            r".*有什么.*特点.*": "features",
            r".*多少.*": "amount",
            r".*怎么.*|.*如何.*": "how_to",
            r".*为什么.*": "why",
            r".*适合.*": "suitability",
            r".*区别.*|.*不同.*": "comparison"
        }
    
    def detect_language(self, text: str) -> str:
        """检测文本语言"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text.replace(" ", ""))
        
        if total_chars == 0:
            return "unknown"
        
        chinese_ratio = chinese_chars / total_chars
        if chinese_ratio > 0.3:
            return "chinese"
        else:
            return "english"
    
    def expand_query(self, query: str) -> List[str]:
        """扩展查询词"""
        expanded_terms = [query]  # 保留原始查询
        query_lower = query.lower()
        
        # 查找并替换术语
        for chinese_term, variations in self.term_mapping.items():
            if chinese_term in query:
                # 添加所有变体
                for variant in variations:
                    expanded_query = query.replace(chinese_term, variant)
                    if expanded_query not in expanded_terms:
                        expanded_terms.append(expanded_query)
        
        # 如果是中文查询，也尝试关键词的英文版本
        if self.detect_language(query) == "chinese":
            english_keywords = []
            for chinese_term, variations in self.term_mapping.items():
                if chinese_term in query:
                    # 使用第一个英文翻译（通常是标准术语）
                    if variations and not re.search(r'[\u4e00-\u9fff]', variations[0]):
                        english_keywords.append(variations[0])
            
            if english_keywords:
                expanded_terms.append(" ".join(english_keywords))
        
        return expanded_terms
    
    def extract_key_info(self, query: str) -> Dict[str, any]:
        """提取查询中的关键信息"""
        info = {
            "query_type": None,
            "product_name": None,
            "amount": None,
            "currency": None,
            "feature": None
        }
        
        # 识别问题类型
        for pattern, q_type in self.question_patterns.items():
            if re.match(pattern, query):
                info["query_type"] = q_type
                break
        
        # 提取产品名
        if "royalfortune" in query.lower() or "皇室财富" in query:
            info["product_name"] = "RoyalFortune"
        elif "aia" in query.lower() or "友邦" in query:
            info["product_name"] = "AIA"
        
        # 提取金额
        amount_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:万|千|百万|million|k|m)', query)
        if amount_match:
            info["amount"] = amount_match.group(1)
        
        # 提取货币
        for currency_cn, currency_variants in [("美元", ["USD", "dollar"]), 
                                               ("港币", ["HKD"]), 
                                               ("人民币", ["RMB", "CNY"])]:
            if currency_cn in query or any(v in query for v in currency_variants):
                info["currency"] = currency_variants[0] if currency_variants else currency_cn
                break
        
        return info
    
    def optimize_for_chinese(self, query: str) -> Tuple[List[str], Dict]:
        """
        综合优化中文查询
        返回: (扩展后的查询列表, 提取的信息)
        """
        # 1. 检测语言
        language = self.detect_language(query)
        
        # 2. 提取关键信息
        key_info = self.extract_key_info(query)
        
        # 3. 扩展查询
        expanded_queries = self.expand_query(query)
        
        # 4. 添加语言标记
        key_info["language"] = language
        
        return expanded_queries, key_info


# 测试函数
def test_optimizer():
    optimizer = ChineseQueryOptimizer()
    
    test_queries = [
        "RoyalFortune的最低保费是多少？",
        "这个产品的保证现金价值怎么样？",
        "我有100万美元，适合买什么产品？",
        "身故赔付有哪些特点？",
        "永明的产品优势是什么？"
    ]
    
    print("中文查询优化测试")
    print("="*60)
    
    for query in test_queries:
        print(f"\n原始查询: {query}")
        expanded, info = optimizer.optimize_for_chinese(query)
        
        print(f"语言: {info['language']}")
        print(f"查询类型: {info['query_type']}")
        print(f"扩展查询:")
        for i, eq in enumerate(expanded[:3], 1):
            print(f"  {i}. {eq}")
        print(f"提取信息: {info}")
        print("-"*40)


if __name__ == "__main__":
    test_optimizer()