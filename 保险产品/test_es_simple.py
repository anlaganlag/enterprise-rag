"""
ElasticSearch简化测试 - 不依赖IK分词器
"""
import time
from elasticsearch import Elasticsearch
from elasticsearch_config_simple import DOCUMENT_INDEX_SETTINGS, EXTRACTED_FIELDS_INDEX_SETTINGS

def test_es_basic():
    """测试ES基础功能"""
    print("="*60)
    print("ElasticSearch 基础功能测试（简化版）")
    print("="*60)
    
    # 连接ES
    es = Elasticsearch(['http://localhost:9200'])
    
    # 1. 测试连接
    print("\n1. 测试连接...")
    if es.ping():
        info = es.info()
        print(f"   ✓ 连接成功")
        print(f"   版本: {info['version']['number']}")
        print(f"   集群: {info['cluster_name']}")
    else:
        print("   ✗ 连接失败")
        return
    
    # 2. 创建索引
    print("\n2. 创建索引...")
    
    # 删除旧索引
    if es.indices.exists(index="insurance_documents"):
        es.indices.delete(index="insurance_documents")
        print("   已删除旧索引")
    
    # 创建新索引
    es.indices.create(index="insurance_documents", body=DOCUMENT_INDEX_SETTINGS)
    print("   ✓ 文档索引创建成功")
    
    # 3. 插入测试文档
    print("\n3. 插入测试文档...")
    test_docs = [
        {
            "doc_id": "test_001",
            "doc_name": "RoyalFortune_Test",
            "content": "Sun Life RoyalFortune minimum premium USD 125,000",
            "content_vector": [0.1] * 1536,
            "page_num": 1
        },
        {
            "doc_id": "test_002",
            "doc_name": "AIA_Test",
            "content": "友邦保險活享儲蓄計劃最低保費要求",
            "content_vector": [0.2] * 1536,
            "page_num": 1
        }
    ]
    
    for doc in test_docs:
        es.index(index="insurance_documents", body=doc, refresh=True)
    print(f"   ✓ 插入 {len(test_docs)} 个文档")
    
    # 4. 测试搜索
    print("\n4. 测试搜索...")
    
    # 英文搜索
    result = es.search(
        index="insurance_documents",
        body={"query": {"match": {"content": "minimum premium"}}}
    )
    print(f"   搜索 'minimum premium': {result['hits']['total']['value']} 个结果")
    
    # 中文搜索
    result = es.search(
        index="insurance_documents",
        body={"query": {"match": {"content": "保費"}}}
    )
    print(f"   搜索 '保費': {result['hits']['total']['value']} 个结果")
    
    # 向量搜索
    print("\n5. 测试向量搜索...")
    result = es.search(
        index="insurance_documents",
        body={
            "size": 2,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'content_vector') + 1.0",
                        "params": {
                            "query_vector": [0.15] * 1536
                        }
                    }
                }
            }
        }
    )
    print(f"   ✓ 向量搜索返回 {len(result['hits']['hits'])} 个结果")
    for hit in result['hits']['hits']:
        print(f"      - {hit['_source']['doc_name']}: 分数 {hit['_score']:.3f}")
    
    # 6. 创建字段索引
    print("\n6. 创建字段索引...")
    if es.indices.exists(index="insurance_fields"):
        es.indices.delete(index="insurance_fields")
    
    es.indices.create(index="insurance_fields", body=EXTRACTED_FIELDS_INDEX_SETTINGS)
    print("   ✓ 字段索引创建成功")
    
    # 7. 测试聚合
    print("\n7. 测试聚合...")
    result = es.search(
        index="insurance_documents",
        body={
            "size": 0,
            "aggs": {
                "by_doc": {
                    "terms": {"field": "doc_name.keyword"}
                }
            }
        }
    )
    print(f"   ✓ 聚合成功，找到 {len(result['aggregations']['by_doc']['buckets'])} 个文档")
    
    print("\n" + "="*60)
    print("测试完成！ES基础功能正常")
    print("="*60)
    
    return True

def main():
    # 等待ES启动
    print("等待ElasticSearch启动...")
    es = Elasticsearch(['http://localhost:9200'])
    
    for i in range(30):
        try:
            if es.ping():
                print("ElasticSearch已就绪")
                break
        except:
            pass
        time.sleep(2)
    
    # 运行测试
    test_es_basic()

if __name__ == "__main__":
    main()