"""
Prompt工程优化 - 具体代码示例
包含完整的Prompt管理器实现和测试框架
"""

import json
import time
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import yaml
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """Prompt模板数据类"""
    name: str
    base_prompt: str
    few_shot_examples: List[Dict]
    validation_rules: Dict
    chain_of_thought: str
    
    
class InsuranceDomainKnowledge:
    """保险领域知识库"""
    
    def __init__(self):
        self.glossary = {
            "基础术语": {
                "保费": "premium - 投保人支付给保险公司的费用",
                "保额": "sum insured - 保险公司承担赔偿的最高限额",
                "保障": "coverage - 保险提供的保护范围",
                "理赔": "claim - 保险事故后的赔偿申请",
                "保单": "policy - 保险合同文件",
                "免赔额": "deductible - 保险公司不予赔付的部分"
            },
            "产品类型": {
                "储蓄险": "savings insurance - 兼具保障和储蓄功能的保险产品",
                "寿险": "life insurance - 以生命为标的的保险",
                "意外险": "accident insurance - 意外事故保障",
                "重疾险": "critical illness insurance - 重大疾病保障",
                "年金险": "annuity insurance - 定期给付年金的保险"
            },
            "技术术语": {
                "现金价值": "cash value - 保单的现金价值",
                "退保": "surrender - 提前终止保险合同",
                "分红": "dividend - 保险公司的利润分配",
                "GCV": "Guaranteed Cash Value - 保证现金价值",
                "NAV": "Net Asset Value - 净资产价值"
            }
        }
        
        self.validation_rules = {
            "数值范围": {
                "年龄": {"min": 0, "max": 100, "unit": "岁"},
                "保费": {"min": 0, "max": float('inf'), "unit": "USD/HKD"},
                "百分比": {"min": 0, "max": 100, "unit": "%"}
            },
            "格式要求": {
                "货币": r"^\d+(\.\d{2})?$",
                "日期": r"^\d{4}-\d{2}-\d{2}$",
                "百分比": r"^\d+(\.\d+)?%$"
            }
        }
    
    def get_relevant_terms(self, query: str) -> Dict[str, str]:
        """获取查询相关的专业术语"""
        relevant_terms = {}
        query_lower = query.lower()
        
        for category, terms in self.glossary.items():
            for term, explanation in terms.items():
                if term in query_lower:
                    relevant_terms[term] = explanation
                    
        return relevant_terms


class PromptOptimizer:
    """Prompt优化管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path) if config_path else {}
        self.domain_knowledge = InsuranceDomainKnowledge()
        self.prompt_templates = self._initialize_templates()
        self.performance_history = []
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return {}
    
    def _initialize_templates(self) -> Dict[str, PromptTemplate]:
        """初始化Prompt模板"""
        templates = {
            "basic_info": PromptTemplate(
                name="基础信息提取",
                base_prompt="""
你是专业的保险产品信息提取专家。

任务：从文档中提取基础产品信息
要求：
1. 产品名称：完整、准确的产品名称
2. 保险公司：官方注册名称
3. 产品类型：储蓄险/寿险/意外险等
4. 货币类型：USD/HKD/CNY等

文档内容：
{context}

问题：{question}
                """,
                few_shot_examples=[
                    {
                        "question": "产品名称是什么？",
                        "answer": "Royal Fortune储蓄保险计划",
                        "confidence": 0.95
                    }
                ],
                validation_rules={
                    "required_fields": ["product_name", "insurer", "type"],
                    "confidence_threshold": 0.8
                },
                chain_of_thought="1.识别文档类型 2.定位关键信息 3.提取并验证 4.格式化输出"
            ),
            
            "numerical": PromptTemplate(
                name="数值参数提取",
                base_prompt="""
你是专业的保险产品信息提取专家，擅长处理数值信息。

任务：精确提取数值参数
要求：
1. 数值必须精确，包含单位
2. 区分最低值和最高值
3. 验证数值合理性
4. 处理范围类型数据

思维链步骤：
{chain_of_thought}

文档内容：
{context}

相关术语：
{terminology}

问题：{question}
                """,
                few_shot_examples=[
                    {
                        "question": "最低保费是多少？",
                        "answer": "USD 10,000",
                        "confidence": 0.90,
                        "source": "第3页，保费说明部分"
                    }
                ],
                validation_rules={
                    "number_format": r"^\d+(\.\d+)?$",
                    "unit_required": True
                },
                chain_of_thought="1.定位数值信息 2.提取数值和单位 3.验证范围合理性 4.格式化输出"
            ),
            
            "conceptual": PromptTemplate(
                name="概念理解",
                base_prompt="""
你是专业的保险产品专家，深入理解保险概念和条款。

任务：理解和解释保险产品概念
要求：
1. 准确理解产品功能
2. 清晰解释复杂概念
3. 保持客观准确
4. 标注不确定部分

推理步骤：
{chain_of_thought}

文档内容：
{context}

专业术语解释：
{terminology}

问题：{question}

请按以下格式回答：
1. 概念解释：
2. 关键特性：
3. 置信度评分：
4. 信息来源：
                """,
                few_shot_examples=[
                    {
                        "question": "什么是保单的提取功能？",
                        "answer": "保单提取功能允许投保人在保单有效期内，从保单的现金价值中提取部分资金，而不需要终止保单。",
                        "confidence": 0.85
                    }
                ],
                validation_rules={
                    "min_length": 50,
                    "max_length": 500
                },
                chain_of_thought="1.识别概念类型 2.查找相关定义 3.结合上下文理解 4.验证理解准确性 5.组织答案"
            ),
            
            "comparison": PromptTemplate(
                name="对比分析",
                base_prompt="""
你是专业的保险产品分析师，擅长产品对比。

任务：对比分析保险产品
要求：
1. 识别对比维度
2. 提取对比数据
3. 分析差异和相似点
4. 提供客观评价

对比框架：
{chain_of_thought}

文档内容：
{context}

问题：{question}

输出格式：
## 对比维度
- 维度1: [说明]
- 维度2: [说明]

## 产品A
[数据]

## 产品B
[数据]

## 主要差异
[分析]

## 相似点
[分析]
                """,
                few_shot_examples=[],
                validation_rules={
                    "comparison_dimensions": ["price", "coverage", "terms"],
                    "require_both_products": True
                },
                chain_of_thought="1.识别对比产品 2.确定对比维度 3.提取各产品数据 4.计算差异 5.总结分析"
            )
        }
        
        return templates
    
    def detect_question_type(self, question: str) -> str:
        """自动检测问题类型"""
        question_lower = question.lower()
        
        # 关键词映射
        type_keywords = {
            'basic_info': ['名称', '公司', '类型', '是什么', '哪家'],
            'numerical': ['多少', '金额', '年龄', '范围', '最低', '最高', '数值'],
            'conceptual': ['什么是', '如何', '为什么', '解释', '含义', '功能'],
            'comparison': ['比较', '区别', '不同', '对比', '哪个更']
        }
        
        for q_type, keywords in type_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                return q_type
                
        return 'basic_info'  # 默认类型
    
    def optimize_prompt(self, 
                       question: str, 
                       context: str,
                       question_type: Optional[str] = None) -> str:
        """生成优化后的Prompt"""
        
        # 自动检测问题类型
        if question_type is None:
            question_type = self.detect_question_type(question)
            logger.info(f"检测到问题类型: {question_type}")
        
        # 获取对应模板
        template = self.prompt_templates.get(question_type, self.prompt_templates['basic_info'])
        
        # 获取相关术语
        terminology = self.domain_knowledge.get_relevant_terms(question)
        terminology_str = "\n".join([f"- {term}: {desc}" for term, desc in terminology.items()])
        
        # 构建few-shot示例
        few_shot_str = self._build_few_shot_examples(template.few_shot_examples)
        
        # 组装最终Prompt
        final_prompt = template.base_prompt.format(
            context=context,
            question=question,
            chain_of_thought=template.chain_of_thought,
            terminology=terminology_str if terminology_str else "无需特殊术语解释",
            few_shot=few_shot_str
        )
        
        # 记录性能
        self._track_performance(question, question_type)
        
        return final_prompt
    
    def _build_few_shot_examples(self, examples: List[Dict]) -> str:
        """构建few-shot示例字符串"""
        if not examples:
            return ""
            
        few_shot_parts = ["示例："]
        for i, example in enumerate(examples, 1):
            few_shot_parts.append(f"""
示例{i}:
Q: {example['question']}
A: {example['answer']}
置信度: {example.get('confidence', 'N/A')}
""")
        
        return "\n".join(few_shot_parts)
    
    def _track_performance(self, question: str, question_type: str):
        """记录性能跟踪"""
        self.performance_history.append({
            "timestamp": time.time(),
            "question": question,
            "type": question_type
        })
    
    def validate_response(self, response: Dict, question_type: str) -> Dict:
        """验证响应质量"""
        template = self.prompt_templates.get(question_type)
        if not template:
            return {"valid": False, "errors": ["未知的问题类型"]}
            
        errors = []
        
        # 检查必需字段
        if "required_fields" in template.validation_rules:
            for field in template.validation_rules["required_fields"]:
                if field not in response:
                    errors.append(f"缺少必需字段: {field}")
        
        # 检查置信度阈值
        if "confidence_threshold" in template.validation_rules:
            threshold = template.validation_rules["confidence_threshold"]
            if response.get("confidence", 0) < threshold:
                errors.append(f"置信度低于阈值: {response.get('confidence')} < {threshold}")
        
        # 检查文本长度
        if "answer" in response:
            answer_length = len(response["answer"])
            if "min_length" in template.validation_rules:
                if answer_length < template.validation_rules["min_length"]:
                    errors.append(f"答案过短: {answer_length} < {template.validation_rules['min_length']}")
            if "max_length" in template.validation_rules:
                if answer_length > template.validation_rules["max_length"]:
                    errors.append(f"答案过长: {answer_length} > {template.validation_rules['max_length']}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }


class ABTestFramework:
    """A/B测试框架"""
    
    def __init__(self):
        self.results = []
        
    def run_test(self, 
                 old_prompt: str,
                 new_prompt: str,
                 test_questions: List[str],
                 llm_client: Any) -> Dict:
        """运行A/B测试"""
        
        results = {
            "old_prompt_results": [],
            "new_prompt_results": [],
            "improvements": [],
            "summary": {}
        }
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            for question in test_questions:
                # 并行测试两个Prompt
                old_future = executor.submit(
                    self._test_single_prompt, 
                    question, old_prompt, "old", llm_client
                )
                new_future = executor.submit(
                    self._test_single_prompt,
                    question, new_prompt, "new", llm_client
                )
                
                old_result = old_future.result()
                new_result = new_future.result()
                
                results["old_prompt_results"].append(old_result)
                results["new_prompt_results"].append(new_result)
                
                # 计算改进
                improvement = self._calculate_improvement(old_result, new_result)
                results["improvements"].append(improvement)
        
        # 生成汇总
        results["summary"] = self._generate_summary(results)
        
        return results
    
    def _test_single_prompt(self, 
                           question: str,
                           prompt: str,
                           prompt_type: str,
                           llm_client: Any) -> Dict:
        """测试单个Prompt"""
        start_time = time.time()
        
        try:
            # 模拟LLM调用
            response = llm_client.generate(
                prompt=prompt.format(question=question),
                temperature=0.1,
                max_tokens=500
            )
            
            end_time = time.time()
            
            result = {
                "question": question,
                "prompt_type": prompt_type,
                "answer": response.get("answer"),
                "confidence": response.get("confidence", 0),
                "response_time": end_time - start_time,
                "success": True
            }
            
        except Exception as e:
            result = {
                "question": question,
                "prompt_type": prompt_type,
                "error": str(e),
                "confidence": 0,
                "response_time": 0,
                "success": False
            }
            
        return result
    
    def _calculate_improvement(self, old_result: Dict, new_result: Dict) -> Dict:
        """计算改进指标"""
        improvement = {
            "question": old_result["question"],
            "confidence_delta": new_result.get("confidence", 0) - old_result.get("confidence", 0),
            "time_delta": new_result.get("response_time", 0) - old_result.get("response_time", 0),
            "success_improvement": new_result.get("success", False) and not old_result.get("success", False)
        }
        
        # 计算改进百分比
        if old_result.get("confidence", 0) > 0:
            improvement["confidence_improvement_pct"] = (
                (new_result.get("confidence", 0) - old_result.get("confidence", 0)) 
                / old_result.get("confidence", 0) * 100
            )
        
        return improvement
    
    def _generate_summary(self, results: Dict) -> Dict:
        """生成测试汇总"""
        old_confidences = [r.get("confidence", 0) for r in results["old_prompt_results"]]
        new_confidences = [r.get("confidence", 0) for r in results["new_prompt_results"]]
        
        old_times = [r.get("response_time", 0) for r in results["old_prompt_results"] if r.get("success")]
        new_times = [r.get("response_time", 0) for r in results["new_prompt_results"] if r.get("success")]
        
        summary = {
            "total_questions": len(results["old_prompt_results"]),
            "old_avg_confidence": np.mean(old_confidences) if old_confidences else 0,
            "new_avg_confidence": np.mean(new_confidences) if new_confidences else 0,
            "confidence_improvement": np.mean(new_confidences) - np.mean(old_confidences) if old_confidences else 0,
            "old_avg_response_time": np.mean(old_times) if old_times else 0,
            "new_avg_response_time": np.mean(new_times) if new_times else 0,
            "time_improvement": np.mean(old_times) - np.mean(new_times) if old_times and new_times else 0,
            "old_success_rate": sum(1 for r in results["old_prompt_results"] if r.get("success")) / len(results["old_prompt_results"]),
            "new_success_rate": sum(1 for r in results["new_prompt_results"] if r.get("success")) / len(results["new_prompt_results"])
        }
        
        # 判断是否改进
        summary["overall_improvement"] = (
            summary["new_avg_confidence"] > summary["old_avg_confidence"] and
            summary["new_success_rate"] >= summary["old_success_rate"]
        )
        
        return summary


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = []
        self.alert_thresholds = {
            "response_time": 3.0,  # 秒
            "confidence": 0.6,
            "error_rate": 0.15
        }
        
    def record_metric(self, metric_type: str, value: float, metadata: Dict = None):
        """记录性能指标"""
        metric = {
            "timestamp": time.time(),
            "type": metric_type,
            "value": value,
            "metadata": metadata or {}
        }
        
        self.metrics.append(metric)
        
        # 检查是否需要告警
        self._check_alert(metric_type, value)
        
    def _check_alert(self, metric_type: str, value: float):
        """检查告警条件"""
        if metric_type == "response_time" and value > self.alert_thresholds["response_time"]:
            self._send_alert(f"响应时间过长: {value}秒")
        elif metric_type == "confidence" and value < self.alert_thresholds["confidence"]:
            self._send_alert(f"置信度过低: {value}")
        elif metric_type == "error_rate" and value > self.alert_thresholds["error_rate"]:
            self._send_alert(f"错误率过高: {value}")
    
    def _send_alert(self, message: str):
        """发送告警"""
        logger.warning(f"性能告警: {message}")
        # 这里可以集成实际的告警系统，如邮件、短信等
        
    def get_statistics(self, metric_type: str, time_window: int = 3600) -> Dict:
        """获取指定时间窗口的统计信息"""
        current_time = time.time()
        window_start = current_time - time_window
        
        # 筛选时间窗口内的指标
        window_metrics = [
            m for m in self.metrics 
            if m["type"] == metric_type and m["timestamp"] >= window_start
        ]
        
        if not window_metrics:
            return {"count": 0}
            
        values = [m["value"] for m in window_metrics]
        
        return {
            "count": len(values),
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "median": np.median(values)
        }


# 使用示例
if __name__ == "__main__":
    # 初始化优化器
    optimizer = PromptOptimizer()
    
    # 测试问题
    test_questions = [
        "产品名称是什么？",
        "最低保费是多少？",
        "什么是保单的提取功能？",
        "Royal Fortune和AIA产品有什么区别？"
    ]
    
    # 模拟文档内容
    context = """
    Royal Fortune储蓄保险计划
    由ABC保险公司提供
    最低保费：USD 10,000
    最高保费：USD 1,000,000
    投保年龄：0-70岁
    保单提取功能：允许从现金价值中提取资金
    """
    
    # 生成优化后的Prompt
    for question in test_questions:
        print(f"\n问题: {question}")
        print("-" * 50)
        
        optimized_prompt = optimizer.optimize_prompt(question, context)
        print(f"优化后的Prompt:\n{optimized_prompt[:500]}...")  # 只显示前500字符
        
        # 检测问题类型
        q_type = optimizer.detect_question_type(question)
        print(f"检测到的问题类型: {q_type}")
    
    # 初始化性能监控
    monitor = PerformanceMonitor()
    
    # 模拟记录性能指标
    monitor.record_metric("response_time", 1.5)
    monitor.record_metric("confidence", 0.85)
    monitor.record_metric("error_rate", 0.05)
    
    # 获取统计信息
    stats = monitor.get_statistics("confidence")
    print(f"\n置信度统计: {stats}")