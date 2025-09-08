"""
医疗文档处理器
支持PDF、Word、文本等多种格式的医疗文档解析和分块
"""

import re
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from .schema import (
    MedicalDocument, DocumentChunk, MedicalEntity, 
    DocumentType, MedicalEntityType
)


class MedicalDocumentProcessor:
    """医疗文档处理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 医疗分块模式
        self.chunk_patterns = {
            'disease_definition': r'(?:疾病定义|定义|概述|简介)',
            'symptoms': r'(?:症状|临床表现|症状表现)',
            'diagnosis': r'(?:诊断|诊断标准|诊断依据)',
            'treatment': r'(?:治疗|治疗方案|治疗方法)',
            'prognosis': r'(?:预后|预后情况|预后评估)',
            'prevention': r'(?:预防|预防措施|预防方法)',
            'complications': r'(?:并发症|并发症情况)',
            'contraindications': r'(?:禁忌症|禁忌|注意事项)'
        }
        
        # 医疗实体识别模式
        self.entity_patterns = {
            MedicalEntityType.DISEASE: [
                r'[a-zA-Z\u4e00-\u9fff]+病',
                r'[a-zA-Z\u4e00-\u9fff]+症',
                r'[a-zA-Z\u4e00-\u9fff]+综合征'
            ],
            MedicalEntityType.DRUG: [
                r'[a-zA-Z\u4e00-\u9fff]+片',
                r'[a-zA-Z\u4e00-\u9fff]+胶囊',
                r'[a-zA-Z\u4e00-\u9fff]+注射液',
                r'[a-zA-Z\u4e00-\u9fff]+颗粒'
            ],
            MedicalEntityType.SYMPTOM: [
                r'[a-zA-Z\u4e00-\u9fff]+痛',
                r'[a-zA-Z\u4e00-\u9fff]+热',
                r'[a-zA-Z\u4e00-\u9fff]+肿',
                r'[a-zA-Z\u4e00-\u9fff]+炎'
            ],
            MedicalEntityType.ICD_CODE: [
                r'[A-Z]\d{2}(?:\.\d{1,2})?',  # ICD-10编码
                r'[A-Z]\d{2}\.\d{1,2}'       # 更精确的ICD-10编码
            ]
        }
    
    def process_document(self, 
                        content: str, 
                        title: str, 
                        doc_type: DocumentType,
                        source: str,
                        metadata: Optional[Dict[str, Any]] = None) -> MedicalDocument:
        """处理医疗文档"""
        
        # 生成文档ID
        doc_id = self._generate_doc_id(content, title)
        
        # 提取医疗实体
        medical_entities = self._extract_medical_entities(content)
        
        # 分块处理
        chunks = self._chunk_document(content, doc_id, medical_entities)
        
        # 创建文档对象
        document = MedicalDocument(
            doc_id=doc_id,
            title=title,
            doc_type=doc_type,
            source=source,
            content=content,
            chunks=chunks,
            medical_entities=medical_entities,
            metadata=metadata or {},
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.logger.info(f"处理文档完成: {doc_id}, 分块数: {len(chunks)}")
        return document
    
    def _generate_doc_id(self, content: str, title: str) -> str:
        """生成文档唯一ID"""
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
        title_slug = re.sub(r'[^\w\u4e00-\u9fff]', '_', title)[:20]
        return f"{title_slug}_{content_hash}"
    
    def _extract_medical_entities(self, content: str) -> List[MedicalEntity]:
        """提取医疗实体"""
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    entity = MedicalEntity(
                        text=match.group(),
                        entity_type=entity_type,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.8,  # 基础置信度
                        normalized_form=self._normalize_entity(match.group(), entity_type)
                    )
                    entities.append(entity)
        
        # 去重
        unique_entities = self._deduplicate_entities(entities)
        return unique_entities
    
    def _normalize_entity(self, text: str, entity_type: MedicalEntityType) -> str:
        """标准化医疗实体"""
        # 这里可以实现更复杂的标准化逻辑
        # 例如：药物名称标准化、疾病名称标准化等
        return text.strip()
    
    def _deduplicate_entities(self, entities: List[MedicalEntity]) -> List[MedicalEntity]:
        """去重医疗实体"""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            key = (entity.text, entity.start_pos, entity.end_pos)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _chunk_document(self, 
                       content: str, 
                       doc_id: str, 
                       entities: List[MedicalEntity]) -> List[DocumentChunk]:
        """分块处理文档"""
        chunks = []
        
        # 按段落分割
        paragraphs = self._split_into_paragraphs(content)
        
        chunk_index = 0
        for para in paragraphs:
            if len(para.strip()) < 50:  # 跳过太短的段落
                continue
                
            # 确定分块类型
            chunk_type = self._determine_chunk_type(para)
            
            # 计算token数量（简单估算）
            token_count = len(para.split())
            
            # 提取该段落中的医疗实体
            para_entities = self._extract_entities_in_range(
                entities, 0, len(para)
            )
            
            # 创建分块
            chunk = DocumentChunk(
                chunk_id=f"{doc_id}_chunk_{chunk_index}",
                doc_id=doc_id,
                content=para,
                chunk_type=chunk_type,
                start_position=0,  # 在段落中的位置
                end_position=len(para),
                token_count=token_count,
                medical_entities=para_entities,
                metadata={
                    'paragraph_index': chunk_index,
                    'chunk_type': chunk_type
                },
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            chunks.append(chunk)
            chunk_index += 1
        
        return chunks
    
    def _split_into_paragraphs(self, content: str) -> List[str]:
        """将内容分割为段落"""
        # 按双换行符分割
        paragraphs = re.split(r'\n\s*\n', content)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _determine_chunk_type(self, content: str) -> str:
        """确定分块类型"""
        content_lower = content.lower()
        
        for chunk_type, pattern in self.chunk_patterns.items():
            if re.search(pattern, content_lower):
                return chunk_type
        
        return 'general'  # 默认类型
    
    def _extract_entities_in_range(self, 
                                  entities: List[MedicalEntity], 
                                  start: int, 
                                  end: int) -> List[MedicalEntity]:
        """提取指定范围内的医疗实体"""
        return [
            entity for entity in entities
            if start <= entity.start_pos < end
        ]


class PDFProcessor(MedicalDocumentProcessor):
    """PDF文档处理器"""
    
    def process_pdf(self, pdf_path: str) -> MedicalDocument:
        """处理PDF文件"""
        # 这里需要集成PDF解析库，如PyPDF2或pdfplumber
        # 暂时返回示例
        raise NotImplementedError("PDF处理功能待实现")


class WordProcessor(MedicalDocumentProcessor):
    """Word文档处理器"""
    
    def process_word(self, word_path: str) -> MedicalDocument:
        """处理Word文件"""
        # 这里需要集成python-docx库
        # 暂时返回示例
        raise NotImplementedError("Word处理功能待实现")

