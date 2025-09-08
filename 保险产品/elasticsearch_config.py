"""
ElasticSearch配置和索引模板定义
包含中文分词器配置和保险行业专用设置
"""

# ES索引配置
DOCUMENT_INDEX_SETTINGS = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "char_filter": {
                "traditional_simplified": {
                    "type": "mapping",
                    "mappings": [
                        "壽=>寿", "險=>险", "醫=>医", "藥=>药",
                        "費=>费", "賠=>赔", "償=>偿", "產=>产"
                    ]
                }
            },
            "filter": {
                "insurance_synonym": {
                    "type": "synonym",
                    "synonyms": [
                        "保费,保險費,premium",
                        "保单,保單,policy",
                        "现金价值,現金價值,cash value,CV",
                        "身故赔偿,身故賠償,死亡给付,death benefit",
                        "投保人,保单持有人,policy holder",
                        "受保人,被保险人,insured",
                        "退保,surrender",
                        "GCV,保证现金价值,guaranteed cash value",
                        "红利,分红,bonus,dividend",
                        "AIA,友邦,友邦保险",
                        "Sun Life,永明,太阳人寿"
                    ]
                },
                "insurance_stop": {
                    "type": "stop",
                    "stopwords": ["的", "了", "在", "是", "和", "与", "及", "或", "将", "会"]
                }
            },
            "analyzer": {
                "ik_insurance_max": {
                    "type": "custom",
                    "char_filter": ["traditional_simplified"],
                    "tokenizer": "ik_max_word",
                    "filter": ["lowercase", "insurance_synonym", "insurance_stop"]
                },
                "ik_insurance_smart": {
                    "type": "custom",
                    "char_filter": ["traditional_simplified"],
                    "tokenizer": "ik_smart",
                    "filter": ["lowercase", "insurance_synonym", "insurance_stop"]
                },
                "standard_insurance": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "insurance_synonym"]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            # 文档元数据
            "doc_id": {"type": "keyword"},
            "doc_name": {"type": "keyword"},
            "doc_type": {"type": "keyword"},  # PDF类型：RoyalFortune, AIA等
            "upload_time": {"type": "date"},
            "page_num": {"type": "integer"},
            
            # 文本内容 - 多字段类型支持不同分析器
            "content": {
                "type": "text",
                "analyzer": "ik_insurance_max",
                "search_analyzer": "ik_insurance_smart",
                "fields": {
                    "keyword": {"type": "keyword"},
                    "english": {"type": "text", "analyzer": "standard_insurance"}
                }
            },
            
            # 向量嵌入
            "content_vector": {
                "type": "dense_vector",
                "dims": 1536,  # OpenAI embedding dimension
                "index": True,
                "similarity": "cosine"
            },
            
            # 结构化信息
            "chunk_id": {"type": "keyword"},
            "chunk_type": {"type": "keyword"},  # paragraph, table, list等
            "chunk_metadata": {"type": "object"},
            
            # 用于高亮显示
            "highlighted_content": {
                "type": "text",
                "analyzer": "ik_insurance_smart",
                "term_vector": "with_positions_offsets"
            }
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

# 查询模板
HYBRID_SEARCH_QUERY = {
    "size": 10,
    "query": {
        "bool": {
            "should": [
                # BM25文本搜索
                {
                    "multi_match": {
                        "query": "",  # 将被实际查询替换
                        "fields": ["content^2", "content.english"],
                        "type": "best_fields",
                        "operator": "or",
                        "boost": 1.0
                    }
                },
                # 向量相似度搜索
                {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'content_vector') + 1.0",
                            "params": {
                                "query_vector": []  # 将被实际向量替换
                            }
                        }
                    }
                }
            ]
        }
    },
    "highlight": {
        "fields": {
            "content": {
                "fragment_size": 150,
                "number_of_fragments": 3,
                "pre_tags": ["<mark>"],
                "post_tags": ["</mark>"]
            }
        }
    },
    "_source": ["doc_name", "page_num", "content", "chunk_type"],
    "track_scores": True
}

# 聚合查询模板（用于统计分析）
AGGREGATION_QUERY = {
    "size": 0,
    "aggs": {
        "by_insurer": {
            "terms": {
                "field": "insurer_entity_name.keyword",
                "size": 20
            },
            "aggs": {
                "products": {
                    "terms": {
                        "field": "product_name.keyword",
                        "size": 50
                    }
                },
                "avg_premium": {
                    "avg": {
                        "field": "minimum_premium"
                    }
                }
            }
        },
        "by_currency": {
            "terms": {
                "field": "policy_currency",
                "size": 10
            }
        },
        "premium_ranges": {
            "range": {
                "field": "minimum_premium",
                "ranges": [
                    {"to": 10000},
                    {"from": 10000, "to": 50000},
                    {"from": 50000, "to": 100000},
                    {"from": 100000}
                ]
            }
        }
    }
}

# 中文分词器测试用例
IK_TEST_CASES = [
    "友邦保險的活享儲蓄計劃提供終身保障",
    "Sun Life RoyalFortune最低保費USD125,000",
    "保證現金價值GCV為首日保費的80%",
    "身故賠償包括復歸紅利和終期分紅"
]