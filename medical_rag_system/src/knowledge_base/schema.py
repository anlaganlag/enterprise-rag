"""
医疗知识库数据模型定义
支持多种医疗文档格式的统一存储和检索
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime


class DocumentType(Enum):
    """医疗文档类型"""
    CLINICAL_GUIDE = "clinical_guide"  # 临床指南
    DRUG_MANUAL = "drug_manual"        # 药物手册
    MEDICAL_TEXTBOOK = "textbook"      # 医学教科书
    RESEARCH_PAPER = "research_paper"  # 研究论文
    DIAGNOSTIC_CRITERIA = "diagnostic" # 诊断标准
    TREATMENT_PROTOCOL = "treatment"   # 治疗方案


class MedicalEntityType(Enum):
    """医疗实体类型"""
    DISEASE = "disease"           # 疾病
    SYMPTOM = "symptom"          # 症状
    DRUG = "drug"                # 药物
    PROCEDURE = "procedure"      # 医疗程序
    ANATOMY = "anatomy"          # 解剖结构
    ICD_CODE = "icd_code"        # ICD编码
    DRUG_CODE = "drug_code"      # 药物编码


@dataclass
class MedicalEntity:
    """医疗实体"""
    text: str
    entity_type: MedicalEntityType
    start_pos: int
    end_pos: int
    confidence: float
    normalized_form: Optional[str] = None  # 标准化形式
    icd_code: Optional[str] = None        # ICD编码
    drug_code: Optional[str] = None       # 药物编码


@dataclass
class DocumentChunk:
    """医疗文档分块"""
    chunk_id: str
    doc_id: str
    content: str
    chunk_type: str  # 如：disease_definition, symptoms, diagnosis, treatment, prognosis
    start_position: int
    end_position: int
    token_count: int
    medical_entities: List[MedicalEntity]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass
class MedicalDocument:
    """医疗文档"""
    doc_id: str
    title: str
    doc_type: DocumentType
    source: str  # 来源URL或文件路径
    content: str
    chunks: List[DocumentChunk]
    medical_entities: List[MedicalEntity]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    version: str = "1.0"
    language: str = "zh-CN"


@dataclass
class RetrievalResult:
    """检索结果"""
    chunk: DocumentChunk
    relevance_score: float
    retrieval_method: str  # "bm25", "vector", "hybrid"
    rank: int


@dataclass
class MedicalAnswer:
    """医疗答案"""
    answer: str
    confidence: float
    sources: List[RetrievalResult]
    medical_entities: List[MedicalEntity]
    disclaimer: str
    last_updated: datetime
    answer_type: str  # "direct", "inferred", "not_found"
    requires_human_review: bool = False


@dataclass
class MedicalQuery:
    """医疗查询"""
    query_id: str
    question: str
    user_id: Optional[str] = None
    context: Optional[str] = None
    medical_entities: List[MedicalEntity] = None
    requires_retrieval: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.medical_entities is None:
            self.medical_entities = []

