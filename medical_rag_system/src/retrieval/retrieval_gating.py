"""
检索门控系统
智能判断何时需要进行医疗文档检索
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import re

from ..knowledge_base.schema import MedicalQuery, MedicalEntity
from ..knowledge_base.medical_terminology import MedicalTerminology


@dataclass
class GatingDecision:
    """门控决策结果"""
    requires_retrieval: bool
    confidence: float
    reason: str
    medical_entities: List[MedicalEntity]
    uncertainty_score: float = 0.0


class RetrievalGating:
    """检索门控系统"""
    
    def __init__(self, terminology: MedicalTerminology):
        self.terminology = terminology
        self.logger = logging.getLogger(__name__)
        
        # 非医疗问题模式
        self.non_medical_patterns = [
            r'天气|温度|下雨|晴天',
            r'时间|几点|日期|今天|明天',
            r'你好|再见|谢谢|不客气',
            r'计算|数学|加减|乘除',
            r'新闻|政治|体育|娱乐'
        ]
        
        # 简单医疗问题模式（可能不需要检索）
        self.simple_medical_patterns = [
            r'什么是|什么是|定义|含义',
            r'正常值|正常范围|标准值',
            r'单位|单位换算|毫克|克|毫升'
        ]
    
    def should_retrieve(self, query: MedicalQuery) -> GatingDecision:
        """判断是否需要检索"""
        
        # 1. 检查是否为非医疗问题
        if self._is_non_medical_question(query.question):
            return GatingDecision(
                requires_retrieval=False,
                confidence=0.9,
                reason="非医疗问题",
                medical_entities=[]
            )
        
        # 2. 检查是否包含医疗术语
        medical_entities = self._extract_medical_entities(query.question)
        if not medical_entities:
            return GatingDecision(
                requires_retrieval=False,
                confidence=0.7,
                reason="未检测到医疗术语",
                medical_entities=[]
            )
        
        # 3. 检查是否为简单医疗问题
        if self._is_simple_medical_question(query.question):
            return GatingDecision(
                requires_retrieval=False,
                confidence=0.6,
                reason="简单医疗问题，可用模型知识回答",
                medical_entities=medical_entities
            )
        
        # 4. 语义不确定性检测
        uncertainty_score = self._calculate_uncertainty(query.question)
        if uncertainty_score > 0.7:  # 高不确定性
            return GatingDecision(
                requires_retrieval=True,
                confidence=0.8,
                reason=f"高不确定性，需要检索 (uncertainty: {uncertainty_score:.2f})",
                medical_entities=medical_entities,
                uncertainty_score=uncertainty_score
            )
        
        # 5. 默认需要检索
        return GatingDecision(
            requires_retrieval=True,
            confidence=0.7,
            reason="包含医疗术语，需要检索",
            medical_entities=medical_entities
        )
    
    def _is_non_medical_question(self, question: str) -> bool:
        """判断是否为非医疗问题"""
        question_lower = question.lower()
        
        for pattern in self.non_medical_patterns:
            if re.search(pattern, question_lower):
                return True
        
        return False
    
    def _is_simple_medical_question(self, question: str) -> bool:
        """判断是否为简单医疗问题"""
        question_lower = question.lower()
        
        for pattern in self.simple_medical_patterns:
            if re.search(pattern, question_lower):
                return True
        
        return False
    
    def _extract_medical_entities(self, text: str) -> List[MedicalEntity]:
        """提取医疗实体"""
        entities = []
        found_terms = self.terminology.find_terms_in_text(text)
        
        for term in found_terms:
            # 在文本中查找术语位置
            start_pos = text.find(term.term)
            if start_pos != -1:
                entity = MedicalEntity(
                    text=term.term,
                    entity_type=term.category,
                    start_pos=start_pos,
                    end_pos=start_pos + len(term.term),
                    confidence=term.confidence,
                    normalized_form=term.normalized_form
                )
                entities.append(entity)
        
        return entities
    
    def _calculate_uncertainty(self, question: str) -> float:
        """计算语义不确定性"""
        # 这里实现语义不确定性检测
        # 简化版本：基于问题长度和复杂度
        
        # 问题长度因子
        length_factor = min(len(question) / 100, 1.0)
        
        # 医疗术语密度
        medical_terms = self.terminology.find_terms_in_text(question)
        term_density = len(medical_terms) / max(len(question.split()), 1)
        
        # 问题复杂度（问号数量、连接词等）
        complexity_indicators = ['?', '？', '如何', '怎么', '为什么', '是否', '能否']
        complexity_score = sum(1 for indicator in complexity_indicators if indicator in question)
        complexity_factor = min(complexity_score / 3, 1.0)
        
        # 综合不确定性分数
        uncertainty = (length_factor + term_density + complexity_factor) / 3
        
        return min(uncertainty, 1.0)
    
    def paraphrase_question(self, question: str, num_paraphrases: int = 3) -> List[str]:
        """生成问题的多种表述"""
        paraphrases = [question]  # 原始问题
        
        # 简单的同义词替换
        synonym_replacements = {
            '症状': ['表现', '临床表现', '征象'],
            '治疗': ['治疗方法', '治疗方案', '医治'],
            '诊断': ['诊断标准', '诊断依据', '确诊'],
            '疾病': ['病症', '疾患', '病患'],
            '药物': ['药品', '药物', '药剂']
        }
        
        for original, synonyms in synonym_replacements.items():
            if original in question:
                for synonym in synonyms[:2]:  # 最多取2个同义词
                    paraphrased = question.replace(original, synonym)
                    if paraphrased not in paraphrases:
                        paraphrases.append(paraphrased)
                    if len(paraphrases) >= num_paraphrases:
                        break
                if len(paraphrases) >= num_paraphrases:
                    break
        
        return paraphrases[:num_paraphrases]
    
    def calculate_answer_uncertainty(self, answers: List[str]) -> float:
        """计算答案的不确定性"""
        if len(answers) < 2:
            return 0.0
        
        # 简化的答案相似度计算
        # 实际应用中应该使用embedding计算语义相似度
        
        # 基于长度差异
        lengths = [len(answer) for answer in answers]
        length_variance = np.var(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0
        
        # 基于关键词差异
        all_words = set()
        for answer in answers:
            all_words.update(answer.split())
        
        word_overlap_scores = []
        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                words1 = set(answers[i].split())
                words2 = set(answers[j].split())
                overlap = len(words1 & words2) / len(words1 | words2) if words1 | words2 else 0
                word_overlap_scores.append(overlap)
        
        avg_overlap = np.mean(word_overlap_scores) if word_overlap_scores else 0
        
        # 综合不确定性（0-1，越高越不确定）
        uncertainty = (length_variance + (1 - avg_overlap)) / 2
        
        return min(uncertainty, 1.0)


class MedicalQuestionClassifier:
    """医疗问题分类器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 问题类型模式
        self.question_patterns = {
            'definition': [r'什么是', r'定义', r'含义', r'概念'],
            'symptoms': [r'症状', r'表现', r'征象', r'临床表现'],
            'diagnosis': [r'诊断', r'确诊', r'诊断标准', r'如何诊断'],
            'treatment': [r'治疗', r'医治', r'治疗方法', r'如何治疗'],
            'prevention': [r'预防', r'预防措施', r'如何预防'],
            'prognosis': [r'预后', r'预后情况', r'预后如何'],
            'drug': [r'药物', r'药品', r'用药', r'剂量'],
            'examination': [r'检查', r'检验', r'检测', r'化验']
        }
    
    def classify_question(self, question: str) -> Dict[str, float]:
        """分类医疗问题"""
        question_lower = question.lower()
        scores = {}
        
        for category, patterns in self.question_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    score += 1
            scores[category] = score / len(patterns)
        
        return scores
    
    def get_primary_category(self, question: str) -> str:
        """获取主要问题类型"""
        scores = self.classify_question(question)
        if not scores:
            return 'general'
        
        return max(scores.items(), key=lambda x: x[1])[0]

