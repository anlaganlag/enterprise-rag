"""
测试向量编码API功能
"""
import requests
import json

# 测试向量搜索
def test_vector_search():
    # 1. 先上传一个测试文档
    print("1. 测试文档上传...")
    
    # 创建测试PDF内容
    with open("test_doc.txt", "w", encoding="utf-8") as f:
        f.write("Sun Life RoyalFortune minimum premium USD 125,000")
    
    # 模拟PDF上传
    files = {'file': ('test.pdf', open('test_doc.txt', 'rb'), 'application/pdf')}
    response = requests.post('http://localhost:8000/api/upload', files=files)
    print(f"   上传响应: {response.status_code}")
    if response.status_code == 200:
        print(f"   结果: {response.json()}")
    else:
        print(f"   错误: {response.text}")
    
    # 2. 测试搜索
    print("\n2. 测试搜索功能...")
    search_data = {
        "query": "minimum premium",
        "size": 5
    }
    response = requests.post('http://localhost:8000/api/search', json=search_data)
    print(f"   搜索响应: {response.status_code}")
    if response.status_code == 200:
        results = response.json()
        print(f"   找到 {len(results)} 个结果")
        for r in results[:2]:
            print(f"   - {r.get('doc_name')}: 分数 {r.get('score', 0):.3f}")
    
    # 3. 测试问答
    print("\n3. 测试问答功能...")
    qa_data = {
        "question": "What is the minimum premium for RoyalFortune?",
        "include_sources": True
    }
    response = requests.post('http://localhost:8000/api/qa', json=qa_data)
    print(f"   问答响应: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"   答案: {result.get('answer', 'No answer')[:100]}...")
        print(f"   置信度: {result.get('confidence', 0):.2f}")

if __name__ == "__main__":
    test_vector_search()