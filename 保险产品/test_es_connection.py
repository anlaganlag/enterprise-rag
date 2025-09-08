"""
ElasticSearch连接和基础操作测试
测试ES环境是否正确配置，包括中文分词器
"""
import unittest
import time
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError
import json
from elasticsearch_config import (
    DOCUMENT_INDEX_SETTINGS,
    EXTRACTED_FIELDS_INDEX_SETTINGS,
    IK_TEST_CASES
)


class TestESConnection(unittest.TestCase):
    """测试ES连接和基础功能"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        cls.es = Elasticsearch(
            ['http://localhost:9200'],
            verify_certs=False,
            request_timeout=30
        )
        cls.test_index = "test_insurance_docs"
        cls.test_fields_index = "test_extracted_fields"
        
    def test_01_connection(self):
        """测试1: ES连接"""
        try:
            info = self.es.info()
            print(f"✓ ES连接成功")
            print(f"  版本: {info['version']['number']}")
            print(f"  集群名: {info['cluster_name']}")
            self.assertTrue(info['version']['number'].startswith('8.'))
        except ConnectionError as e:
            self.fail(f"✗ ES连接失败: {e}")
            
    def test_02_cluster_health(self):
        """测试2: 集群健康状态"""
        health = self.es.cluster.health()
        print(f"✓ 集群状态: {health['status']}")
        print(f"  节点数: {health['number_of_nodes']}")
        self.assertIn(health['status'], ['yellow', 'green'])
        
    def test_03_create_document_index(self):
        """测试3: 创建文档索引"""
        # 删除已存在的测试索引
        if self.es.indices.exists(index=self.test_index):
            self.es.indices.delete(index=self.test_index)
            
        # 创建索引
        response = self.es.indices.create(
            index=self.test_index,
            body=DOCUMENT_INDEX_SETTINGS
        )
        
        self.assertTrue(response['acknowledged'])
        print(f"✓ 文档索引创建成功: {self.test_index}")
        
        # 验证映射
        mapping = self.es.indices.get_mapping(index=self.test_index)
        self.assertIn('content_vector', 
                     mapping[self.test_index]['mappings']['properties'])
        print(f"  向量字段配置正确")
        
    def test_04_create_fields_index(self):
        """测试4: 创建字段索引"""
        if self.es.indices.exists(index=self.test_fields_index):
            self.es.indices.delete(index=self.test_fields_index)
            
        response = self.es.indices.create(
            index=self.test_fields_index,
            body=EXTRACTED_FIELDS_INDEX_SETTINGS
        )
        
        self.assertTrue(response['acknowledged'])
        print(f"✓ 字段索引创建成功: {self.test_fields_index}")
        
    def test_05_chinese_analyzer(self):
        """测试5: 中文分词器"""
        # 测试分词效果
        for text in IK_TEST_CASES:
            response = self.es.indices.analyze(
                index=self.test_index,
                body={
                    "analyzer": "ik_insurance_max",
                    "text": text
                }
            )
            
            tokens = [t['token'] for t in response['tokens']]
            print(f"✓ 分词测试: '{text[:30]}...'")
            print(f"  分词结果: {tokens[:10]}")
            
            # 验证分词结果
            self.assertGreater(len(tokens), 0)
            
    def test_06_insert_test_document(self):
        """测试6: 插入测试文档"""
        test_doc = {
            "doc_id": "test_001",
            "doc_name": "RoyalFortune_Test",
            "doc_type": "RoyalFortune",
            "page_num": 1,
            "content": "Sun Life RoyalFortune提供最低保費USD125,000的終身保障計劃",
            "content_vector": [0.1] * 1536,  # 模拟向量
            "chunk_type": "paragraph"
        }
        
        response = self.es.index(
            index=self.test_index,
            body=test_doc,
            refresh=True
        )
        
        self.assertEqual(response['result'], 'created')
        print(f"✓ 文档插入成功: ID={response['_id']}")
        
    def test_07_search_chinese(self):
        """测试7: 中文搜索"""
        # 搜索测试
        search_queries = ["保費", "Sun Life", "終身保障", "125000"]
        
        for query in search_queries:
            response = self.es.search(
                index=self.test_index,
                body={
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["content", "content.english"]
                        }
                    }
                }
            )
            
            hits = response['hits']['total']['value']
            print(f"✓ 搜索 '{query}': 找到 {hits} 个结果")
            
            if hits > 0:
                score = response['hits']['hits'][0]['_score']
                print(f"  最高相关度分数: {score:.2f}")
                
    def test_08_bulk_insert(self):
        """测试8: 批量插入"""
        bulk_data = []
        
        # 准备批量数据
        for i in range(10):
            bulk_data.append({"index": {"_index": self.test_index}})
            bulk_data.append({
                "doc_id": f"bulk_{i}",
                "doc_name": f"Test_Doc_{i}",
                "content": f"這是第{i}個測試文檔，包含保險條款和保費信息",
                "page_num": i
            })
            
        response = self.es.bulk(body=bulk_data, refresh=True)
        
        self.assertFalse(response['errors'])
        print(f"✓ 批量插入成功: {len(response['items'])} 个文档")
        
    def test_09_aggregation(self):
        """测试9: 聚合查询"""
        # 插入一些结构化数据用于聚合
        test_fields = [
            {
                "doc_id": "agg_1",
                "insurer_entity_name": "Sun Life",
                "product_name": "RoyalFortune",
                "minimum_premium": 125000,
                "policy_currency": "USD"
            },
            {
                "doc_id": "agg_2", 
                "insurer_entity_name": "AIA",
                "product_name": "FlexiAchiever",
                "minimum_premium": 50000,
                "policy_currency": "USD"
            }
        ]
        
        for doc in test_fields:
            self.es.index(
                index=self.test_fields_index,
                body=doc,
                refresh=True
            )
            
        # 执行聚合查询
        response = self.es.search(
            index=self.test_fields_index,
            body={
                "size": 0,
                "aggs": {
                    "by_insurer": {
                        "terms": {
                            "field": "insurer_entity_name.keyword"
                        }
                    },
                    "avg_premium": {
                        "avg": {
                            "field": "minimum_premium"
                        }
                    }
                }
            }
        )
        
        aggs = response['aggregations']
        print(f"✓ 聚合查询成功")
        print(f"  保险公司数: {len(aggs['by_insurer']['buckets'])}")
        print(f"  平均保费: {aggs['avg_premium']['value']:.2f}")
        
    def test_10_highlight(self):
        """测试10: 高亮显示"""
        response = self.es.search(
            index=self.test_index,
            body={
                "query": {
                    "match": {
                        "content": "保費"
                    }
                },
                "highlight": {
                    "fields": {
                        "content": {}
                    },
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"]
                }
            }
        )
        
        if response['hits']['total']['value'] > 0:
            hit = response['hits']['hits'][0]
            if 'highlight' in hit:
                print(f"✓ 高亮显示成功")
                print(f"  高亮片段: {hit['highlight']['content'][0]}")
                
    @classmethod
    def tearDownClass(cls):
        """清理测试环境"""
        # 保留测试索引供后续使用
        print("\n测试索引保留供后续测试使用")
        print(f"  - {cls.test_index}")
        print(f"  - {cls.test_fields_index}")
        

def run_tests():
    """运行所有测试"""
    print("="*60)
    print("ElasticSearch 基础功能测试")
    print("="*60)
    
    # 等待ES启动
    print("等待ElasticSearch启动...")
    es = Elasticsearch(['http://localhost:9200'])
    
    max_retries = 30
    for i in range(max_retries):
        try:
            if es.ping():
                print("ElasticSearch已就绪\n")
                break
        except:
            pass
        time.sleep(2)
        if i == max_retries - 1:
            print("ElasticSearch连接超时")
            return
            
    # 运行测试
    suite = unittest.TestLoader().loadTestsFromTestCase(TestESConnection)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*60)
    print(f"测试结果: 运行 {result.testsRun} 个测试")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print("="*60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)