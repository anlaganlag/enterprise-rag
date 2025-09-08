"""
医疗术语词库和触发规则
建立医疗领域专业术语的标准化词库
"""

from typing import Dict, List, Set, Optional
from dataclasses import dataclass
import json
import logging


@dataclass
class MedicalTerm:
    """医疗术语"""
    term: str
    category: str  # 疾病、药物、症状等
    synonyms: List[str]  # 同义词
    icd_code: Optional[str] = None
    drug_code: Optional[str] = None
    normalized_form: Optional[str] = None
    confidence: float = 1.0


class MedicalTerminology:
    """医疗术语管理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.terms: Dict[str, MedicalTerm] = {}
        self.synonym_map: Dict[str, str] = {}  # 同义词到标准词的映射
        self.category_index: Dict[str, Set[str]] = {}  # 按类别索引
        self._load_default_terms()
    
    def _load_default_terms(self):
        """加载默认医疗术语"""
        # 常见疾病
        diseases = [
            "高血压", "糖尿病", "冠心病", "脑卒中", "肺炎", "肝炎",
            "肾炎", "胃炎", "关节炎", "哮喘", "癫痫", "抑郁症",
            "焦虑症", "癌症", "肿瘤", "白血病", "淋巴瘤"
        ]
        
        # 常见症状
        symptoms = [
            "头痛", "发热", "咳嗽", "胸痛", "腹痛", "恶心", "呕吐",
            "腹泻", "便秘", "失眠", "疲劳", "乏力", "心悸", "气短",
            "呼吸困难", "水肿", "皮疹", "瘙痒", "麻木", "疼痛"
        ]
        
        # 常见药物
        drugs = [
            "阿司匹林", "布洛芬", "对乙酰氨基酚", "青霉素", "头孢菌素",
            "阿莫西林", "甲硝唑", "奥美拉唑", "雷尼替丁", "硝苯地平",
            "美托洛尔", "卡托普利", "氢氯噻嗪", "二甲双胍", "胰岛素"
        ]
        
        # 医疗程序
        procedures = [
            "手术", "化疗", "放疗", "透析", "输血", "输液", "注射",
            "穿刺", "活检", "内镜检查", "X光检查", "CT检查", "MRI检查",
            "超声检查", "心电图", "血压测量", "血糖检测"
        ]
        
        # 添加术语到词库
        for disease in diseases:
            self.add_term(MedicalTerm(
                term=disease,
                category="disease",
                synonyms=[],
                confidence=1.0
            ))
        
        for symptom in symptoms:
            self.add_term(MedicalTerm(
                term=symptom,
                category="symptom",
                synonyms=[],
                confidence=1.0
            ))
        
        for drug in drugs:
            self.add_term(MedicalTerm(
                term=drug,
                category="drug",
                synonyms=[],
                confidence=1.0
            ))
        
        for procedure in procedures:
            self.add_term(MedicalTerm(
                term=procedure,
                category="procedure",
                synonyms=[],
                confidence=1.0
            ))
    
    def add_term(self, term: MedicalTerm):
        """添加医疗术语"""
        self.terms[term.term] = term
        
        # 更新同义词映射
        for synonym in term.synonyms:
            self.synonym_map[synonym] = term.term
        
        # 更新类别索引
        if term.category not in self.category_index:
            self.category_index[term.category] = set()
        self.category_index[term.category].add(term.term)
        
        self.logger.debug(f"添加医疗术语: {term.term}")
    
    def get_term(self, term: str) -> Optional[MedicalTerm]:
        """获取医疗术语"""
        # 直接查找
        if term in self.terms:
            return self.terms[term]
        
        # 通过同义词查找
        if term in self.synonym_map:
            standard_term = self.synonym_map[term]
            return self.terms.get(standard_term)
        
        return None
    
    def find_terms_in_text(self, text: str) -> List[MedicalTerm]:
        """在文本中查找医疗术语"""
        found_terms = []
        
        for term_name, term in self.terms.items():
            if term_name in text:
                found_terms.append(term)
            
            # 检查同义词
            for synonym in term.synonyms:
                if synonym in text:
                    found_terms.append(term)
        
        return found_terms
    
    def get_terms_by_category(self, category: str) -> List[MedicalTerm]:
        """按类别获取术语"""
        if category not in self.category_index:
            return []
        
        return [
            self.terms[term_name] 
            for term_name in self.category_index[category]
        ]
    
    def is_medical_term(self, term: str) -> bool:
        """判断是否为医疗术语"""
        return self.get_term(term) is not None
    
    def get_trigger_terms(self) -> Set[str]:
        """获取触发检索的术语"""
        # 返回所有术语和同义词
        trigger_terms = set(self.terms.keys())
        trigger_terms.update(self.synonym_map.keys())
        return trigger_terms
    
    def load_from_file(self, file_path: str):
        """从文件加载术语"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for term_data in data:
                term = MedicalTerm(**term_data)
                self.add_term(term)
            
            self.logger.info(f"从文件加载术语: {file_path}")
        except Exception as e:
            self.logger.error(f"加载术语文件失败: {e}")
    
    def save_to_file(self, file_path: str):
        """保存术语到文件"""
        try:
            data = []
            for term in self.terms.values():
                data.append({
                    'term': term.term,
                    'category': term.category,
                    'synonyms': term.synonyms,
                    'icd_code': term.icd_code,
                    'drug_code': term.drug_code,
                    'normalized_form': term.normalized_form,
                    'confidence': term.confidence
                })
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"保存术语到文件: {file_path}")
        except Exception as e:
            self.logger.error(f"保存术语文件失败: {e}")


class ICD10CodeManager:
    """ICD-10编码管理器"""
    
    def __init__(self):
        self.codes: Dict[str, str] = {}  # 编码到描述的映射
        self.descriptions: Dict[str, str] = {}  # 描述到编码的映射
        self._load_common_codes()
    
    def _load_common_codes(self):
        """加载常见ICD-10编码"""
        common_codes = {
            "I10": "原发性高血压",
            "E11": "2型糖尿病",
            "I25": "慢性缺血性心脏病",
            "I63": "脑梗死",
            "J18": "肺炎，病原体未特指",
            "K59": "其他功能性肠疾患",
            "M79": "其他软组织疾患",
            "F32": "抑郁发作",
            "F41": "其他焦虑障碍",
            "C78": "继发性恶性肿瘤",
            "C91": "淋巴细胞白血病"
        }
        
        for code, description in common_codes.items():
            self.codes[code] = description
            self.descriptions[description] = code
    
    def get_description(self, code: str) -> Optional[str]:
        """根据编码获取描述"""
        return self.codes.get(code)
    
    def get_code(self, description: str) -> Optional[str]:
        """根据描述获取编码"""
        return self.descriptions.get(description)
    
    def is_valid_code(self, code: str) -> bool:
        """验证编码是否有效"""
        return code in self.codes


class DrugCodeManager:
    """药物编码管理器"""
    
    def __init__(self):
        self.drugs: Dict[str, Dict[str, str]] = {}  # 药物名称到编码信息的映射
        self._load_common_drugs()
    
    def _load_common_drugs(self):
        """加载常见药物编码"""
        common_drugs = {
            "阿司匹林": {
                "atc_code": "N02BA01",
                "generic_name": "acetylsalicylic acid",
                "category": "解热镇痛药"
            },
            "布洛芬": {
                "atc_code": "M01AE01",
                "generic_name": "ibuprofen",
                "category": "非甾体抗炎药"
            },
            "青霉素": {
                "atc_code": "J01CE01",
                "generic_name": "penicillin",
                "category": "抗生素"
            }
        }
        
        self.drugs.update(common_drugs)
    
    def get_drug_info(self, drug_name: str) -> Optional[Dict[str, str]]:
        """获取药物信息"""
        return self.drugs.get(drug_name)
    
    def is_valid_drug(self, drug_name: str) -> bool:
        """验证药物名称是否有效"""
        return drug_name in self.drugs

