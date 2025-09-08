"""
ElasticSearch配置 - 简化版（不依赖IK分词器）
使用ES内置的standard分词器
"""

# ES索引配置 - 简化版
DOCUMENT_INDEX_SETTINGS = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "chinese_analyzer": {
                    "type": "standard"  # 使用标准分词器
                }
            }
        }
    },
    "mappings": {
        "properties": {
            # 文档元数据
            "doc_id": {"type": "keyword"},
            "doc_name": {"type": "keyword"},
            "doc_type": {"type": "keyword"},
            "upload_time": {"type": "date"},
            "page_num": {"type": "integer"},
            
            # 文本内容
            "content": {
                "type": "text",
                "analyzer": "standard",
                "fields": {
                    "keyword": {"type": "keyword"}
                }
            },
            
            # 向量嵌入
            "content_vector": {
                "type": "dense_vector",
                "dims": 1536,
                "index": True,
                "similarity": "cosine"
            },
            
            # 结构化信息
            "chunk_id": {"type": "keyword"},
            "chunk_type": {"type": "keyword"},
            "chunk_metadata": {"type": "object"}
        }
    }
}

# 提取字段索引配置
EXTRACTED_FIELDS_INDEX_SETTINGS = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "doc_id": {"type": "keyword"},
            "doc_name": {"type": "keyword"},
            "extraction_time": {"type": "date"},
            
            # 保险产品核心字段
            "insurer_entity_name": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword"}}
            },
            "product_name": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword"}}
            },
            "minimum_premium": {"type": "float"},
            "maximum_premium": {"type": "float"},
            "policy_currency": {"type": "keyword"},
            "policy_term": {"type": "keyword"},
            "issue_age": {"type": "keyword"},
            "premium_terms": {"type": "keyword"},
            "product_type": {"type": "keyword"},
            "issuing_jurisdiction": {"type": "keyword"},
            
            # 详细信息
            "withdrawal_features": {"type": "text"},
            "death_settlement_features": {"type": "text"},
            "additional_benefits": {"type": "text"},
            "product_asset_mix": {"type": "text"},
            "day_1_gcv": {"type": "keyword"},
            "surrender_value_components": {"type": "text"},
            "death_benefit_components": {"type": "text"},
            
            # 置信度和元数据
            "confidence_scores": {"type": "object"},
            "extraction_method": {"type": "object"},
            "last_updated": {"type": "date"}
        }
    }
}