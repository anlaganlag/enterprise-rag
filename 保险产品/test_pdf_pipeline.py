"""
PDF处理管道测试用例
测试PDF解析、分块、向量化、索引的完整流程
"""
import unittest
import asyncio
import os
from pathlib import Path
import hashlib
from datetime import datetime
from es_pdf_pipeline import PDFProcessor, VectorEncoder, ESIndexer, PDFPipeline
from elasticsearch import Elasticsearch
import time


class TestPDFPipeline(unittest.TestCase):
    """PDF处理管道测试"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        cls.es = Elasticsearch(['http://localhost:9200'])
        cls.processor = PDFProcessor()
        cls.test_pdf = r"D:\桌面\RAG\保险产品\RoyalFortune_Product Brochure_EN.pdf"
        cls.test_index = "test_pipeline_docs"
        
        # 确保测试索引存在
        if not cls.es.indices.exists(index=cls.test_index):
            from elasticsearch_config import DOCUMENT_INDEX_SETTINGS
            cls.es.indices.create(index=cls.test_index, body=DOCUMENT_INDEX_SETTINGS)
            
    def test_01_pdf_extraction(self):
        """测试1: PDF文本提取"""
        if not Path(self.test_pdf).exists():
            self.skipTest(f"测试PDF不存在: {self.test_pdf}")
            
        doc_info = self.processor.extract_text_from_pdf(self.test_pdf)
        
        print(f"✓ PDF提取成功")
        print(f"  文件名: {doc_info['file_name']}")
        print(f"  总页数: {doc_info['total_pages']}")
        print(f"  表格数: {len(doc_info['tables'])}")
        
        # 验证提取结果
        self.assertGreater(doc_info['total_pages'], 0)
        self.assertGreater(len(doc_info['pages']), 0)
        
        # 检查关键内容
        full_text = "\n".join(p['text'] for p in doc_info['pages'])
        self.assertIn("Sun Life", full_text)
        self.assertIn("RoyalFortune", full_text)
        
    def test_02_text_chunking(self):
        """测试2: 文本分块"""
        doc_info = self.processor.extract_text_from_pdf(self.test_pdf)
        chunks = self.processor.create_chunks(doc_info)
        
        print(f"✓ 文本分块成功")
        print(f"  总分块数: {len(chunks)}")
        
        # 统计不同类型的块
        chunk_types = {}
        for chunk in chunks:
            chunk_type = chunk['chunk_type']
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
        for ctype, count in chunk_types.items():
            print(f"  {ctype}: {count} 个")
            
        # 验证分块结果
        self.assertGreater(len(chunks), 10)
        self.assertTrue(any(c['chunk_type'] == 'paragraph' for c in chunks))
        
        # 检查分块内容
        sample_chunk = chunks[0]
        self.assertIn('doc_id', sample_chunk)
        self.assertIn('chunk_id', sample_chunk)
        self.assertIn('content', sample_chunk)
        self.assertGreater(len(sample_chunk['content']), 20)
        
    def test_03_table_extraction(self):
        """测试3: 表格提取"""
        doc_info = self.processor.extract_text_from_pdf(self.test_pdf)
        
        if doc_info['tables']:
            print(f"✓ 表格提取成功")
            print(f"  表格总数: {len(doc_info['tables'])}")
            
            # 检查第一个表格
            first_table = doc_info['tables'][0]
            print(f"  第一个表格所在页: {first_table['page']}")
            print(f"  表格内容预览: {first_table['content'][:200]}...")
            
            self.assertIn('content', first_table)
            self.assertIn('page', first_table)
        else:
            print("○ 文档中未发现表格")
            
    def test_04_vector_encoding(self):
        """测试4: 向量编码"""
        if not os.getenv("OPENAI_API_KEY"):
            self.skipTest("未设置OPENAI_API_KEY")
            
        encoder = VectorEncoder()
        test_texts = [
            "Sun Life insurance premium",
            "最低保費USD125,000",
            "Guaranteed cash value"
        ]
        
        # 测试同步编码
        vector = encoder.encode_sync(test_texts[0])
        
        print(f"✓ 向量编码成功")
        print(f"  向量维度: {len(vector)}")
        print(f"  向量样本: [{vector[0]:.4f}, {vector[1]:.4f}, ...]")
        
        self.assertEqual(len(vector), 1536)
        self.assertTrue(all(isinstance(v, float) for v in vector))
        
        # 测试批量编码
        async def test_batch():
            vectors = await encoder.encode_batch(test_texts)
            return vectors
            
        vectors = asyncio.run(test_batch())
        
        print(f"  批量编码: {len(vectors)} 个向量")
        self.assertEqual(len(vectors), len(test_texts))
        
    def test_05_es_indexing(self):
        """测试5: ES索引"""
        indexer = ESIndexer()
        
        # 准备测试数据
        test_chunks = [
            {
                "doc_id": "test_doc_001",
                "doc_name": "test.pdf",
                "chunk_id": "test_chunk_001",
                "chunk_type": "paragraph",
                "content": "This is a test paragraph about insurance premium",
                "page_num": 1,
                "metadata": {}
            }
        ]
        
        test_vectors = [[0.1] * 1536]
        
        # 异步索引
        async def test_index():
            return await indexer.index_chunks(test_chunks, test_vectors)
            
        success = asyncio.run(test_index())
        
        print(f"✓ ES索引测试")
        print(f"  索引结果: {'成功' if success else '失败'}")
        
        self.assertTrue(success)
        
        # 验证索引
        time.sleep(1)  # 等待索引刷新
        result = self.es.get(index=indexer.doc_index, id="test_chunk_001", ignore=404)
        
        if result.get('found'):
            print(f"  文档已索引: {result['_id']}")
            self.assertEqual(result['_source']['content'], test_chunks[0]['content'])
            
    def test_06_search_functionality(self):
        """测试6: 搜索功能"""
        indexer = ESIndexer()
        
        # 先索引一些测试数据
        test_data = [
            {"chunk_id": "search_test_1", "content": "minimum premium is USD 125,000"},
            {"chunk_id": "search_test_2", "content": "保費最低要求是125000美元"},
            {"chunk_id": "search_test_3", "content": "Sun Life RoyalFortune product"}
        ]
        
        for data in test_data:
            self.es.index(
                index=indexer.doc_index,
                id=data["chunk_id"],
                body={
                    **data,
                    "doc_id": "test_search",
                    "doc_name": "test.pdf",
                    "chunk_type": "test",
                    "page_num": 1,
                    "content_vector": [0.1] * 1536
                },
                refresh=True
            )
            
        # 测试搜索
        queries = ["premium", "保費", "Sun Life"]
        
        print(f"✓ 搜索功能测试")
        for query in queries:
            results = indexer.search(query, size=5)
            hits = results['hits']['total']['value']
            
            print(f"  搜索 '{query}': {hits} 个结果")
            
            if hits > 0:
                top_hit = results['hits']['hits'][0]
                print(f"    最佳匹配: {top_hit['_source']['content'][:50]}...")
                print(f"    相关度分数: {top_hit['_score']:.2f}")
                
    def test_07_complete_pipeline(self):
        """测试7: 完整管道流程"""
        if not Path(self.test_pdf).exists():
            self.skipTest("测试PDF不存在")
            
        pipeline = PDFPipeline()
        
        async def run_pipeline():
            return await pipeline.process_pdf(self.test_pdf, extract_fields=True)
            
        print(f"✓ 完整管道测试")
        print(f"  处理文件: {Path(self.test_pdf).name}")
        
        result = asyncio.run(run_pipeline())
        
        print(f"  处理结果:")
        print(f"    成功: {result['success']}")
        print(f"    文档ID: {result['doc_id']}")
        print(f"    分块数: {result['chunks']}")
        print(f"    页数: {result['pages']}")
        print(f"    表格数: {result['tables']}")
        
        self.assertTrue(result['success'])
        self.assertGreater(result['chunks'], 0)
        
    def test_08_realtime_processing(self):
        """测试8: 实时处理性能"""
        if not Path(self.test_pdf).exists():
            self.skipTest("测试PDF不存在")
            
        pipeline = PDFPipeline()
        
        print(f"✓ 实时处理性能测试")
        
        start_time = time.time()
        
        async def process_with_timing():
            return await pipeline.process_pdf(self.test_pdf, extract_fields=False)
            
        result = asyncio.run(process_with_timing())
        
        elapsed = time.time() - start_time
        
        print(f"  处理时间: {elapsed:.2f} 秒")
        print(f"  处理速度: {result['pages']/elapsed:.1f} 页/秒")
        print(f"  分块速度: {result['chunks']/elapsed:.1f} 块/秒")
        
        # 性能基准：应该在30秒内完成
        self.assertLess(elapsed, 30)
        
    @classmethod
    def tearDownClass(cls):
        """清理测试环境"""
        # 保留测试数据供检查
        print("\n测试索引保留供检查")
        

def run_tests():
    """运行所有测试"""
    print("="*60)
    print("PDF处理管道测试")
    print("="*60)
    
    # 检查ES连接
    es = Elasticsearch(['http://localhost:9200'])
    if not es.ping():
        print("错误: ElasticSearch未运行")
        print("请先启动: docker-compose up -d")
        return False
        
    # 运行测试
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPDFPipeline)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*60)
    print(f"测试完成: {result.testsRun} 个测试")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print("="*60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)